"""
Test file for pipeline 'keyword_processing_gtr'.
"""

# pylint: skip-file
import pytest
from unittest.mock import MagicMock
import pandas as pd
from dsit_taxonomy.pipelines.keyword_processing_gtr.nodes import (
    dbp_keywords,
    rake_keywords,
    yake_keywords,
    keybert_keywords,
    concatenate_partitions,
    aggregate_keyword_annotators,
    _preprocess_keywords,
)


@pytest.fixture
def sample_projects_df():
    """Create a sample projects DataFrame for testing."""
    return pd.DataFrame(
        {
            "project_id": ["1", "2"],
            "title": ["AI Research Project", "Machine Learning Study"],
            "abstract_text": ["This is an AI study", "ML research paper"],
            "tech_abstract_text": ["Technical details", "More technical info"],
            "potential_impact": ["High impact", "Medium impact"],
        }
    )


@pytest.fixture
def empty_processed_df():
    """Create an empty DataFrame representing no previously processed projects."""
    return pd.DataFrame(columns=["project_id", "keywords"])


@pytest.fixture
def sample_dbp_keywords():
    """Create sample DBpedia keywords data."""
    return pd.DataFrame(
        {
            "project_id": ["1", "2", "3"],
            "dbp_keywords": [
                ["artificial intelligence", "machine learning"],
                ["deep learning", "neural networks"],
                ["machine learning", "data science"],
            ],
        }
    )


@pytest.fixture
def sample_rake_keywords():
    """Create sample RAKE keywords data."""
    return pd.DataFrame(
        {
            "project_id": ["1", "2", "3"],
            "rake_keywords": [
                ["machine learning", "AI research"],
                ["neural networks", "deep learning"],
                ["data analysis", "machine learning"],
            ],
        }
    )


@pytest.fixture
def sample_yake_keywords():
    """Create sample YAKE keywords data."""
    return pd.DataFrame(
        {
            "project_id": ["1", "2", "3"],
            "yake_keywords": [
                ["AI", "machine learning"],
                ["neural networks", "ML models"],
                ["data science", "analytics"],
            ],
        }
    )


@pytest.fixture
def sample_keybert_keywords():
    """Create sample KeyBERT keywords data."""
    return pd.DataFrame(
        {
            "project_id": ["1", "2", "3"],
            "keybert_keywords": [
                ["artificial intelligence", "ML"],
                ["neural networks", "deep learning"],
                ["machine learning", "data mining"],
            ],
        }
    )


def test_dbp_keywords(sample_projects_df, empty_processed_df):
    """Test DBpedia keyword extraction."""
    result = dbp_keywords(sample_projects_df, empty_processed_df)

    assert isinstance(result, pd.DataFrame)
    assert "project_id" in result.columns
    assert "dbp_keywords" in result.columns
    assert len(result) == len(sample_projects_df)


def test_rake_keywords(sample_projects_df, empty_processed_df):
    """Test RAKE keyword extraction."""
    result = rake_keywords(sample_projects_df, empty_processed_df)

    assert isinstance(result, pd.DataFrame)
    assert "project_id" in result.columns
    assert "rake_keywords" in result.columns
    assert len(result) == len(sample_projects_df)


def test_yake_keywords(sample_projects_df, empty_processed_df):
    """Test YAKE keyword extraction."""
    result = yake_keywords(sample_projects_df, empty_processed_df)

    assert isinstance(result, pd.DataFrame)
    assert "project_id" in result.columns
    assert "yake_keywords" in result.columns
    assert len(result) == len(sample_projects_df)


def test_keybert_keywords(sample_projects_df, empty_processed_df):
    """Test KeyBERT keyword extraction."""
    generator = keybert_keywords(sample_projects_df, empty_processed_df)
    first_batch = next(generator)

    assert isinstance(first_batch, dict)
    assert len(first_batch) == 1

    # Get the DataFrame from the first batch
    df = list(first_batch.values())[0]
    assert isinstance(df, pd.DataFrame)
    assert "project_id" in df.columns
    assert "keybert_keywords" in df.columns


def test_concatenate_partitions():
    """Test concatenation of partitioned results."""

    partition1 = pd.DataFrame(
        {"project_id": ["1"], "keybert_keywords": [["ai", "machine learning"]]}
    )
    partition2 = pd.DataFrame(
        {"project_id": ["2"], "keybert_keywords": [["data science", "analytics"]]}
    )

    partitioned_dataset = {
        "part1": MagicMock(return_value=partition1),
        "part2": MagicMock(return_value=partition2),
    }

    result = concatenate_partitions(partitioned_dataset)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "project_id" in result.columns
    assert "keybert_keywords" in result.columns


def test_aggregate_keyword_annotators(
    sample_dbp_keywords,
    sample_rake_keywords,
    sample_yake_keywords,
    sample_keybert_keywords,
):
    """Test keyword aggregation from multiple extractors."""
    result = aggregate_keyword_annotators(
        sample_dbp_keywords,
        sample_rake_keywords,
        sample_yake_keywords,
        sample_keybert_keywords,
    )

    assert isinstance(result, pd.DataFrame)
    assert all(
        col in result.columns
        for col in ["keyword", "num_annotators", "project_ids", "uuid"]
    )
    assert len(result) > 0
    # Check that keywords appearing in multiple extractors are kept
    assert all(result["num_annotators"] > 1)
    # Check that project_ids is a list
    assert all(isinstance(ids, list) for ids in result["project_ids"])


def test_preprocess_keywords():
    """Test keyword preprocessing."""
    input_series = pd.Series(
        [
            "Machine Learning",
            " artificial intelligence ",
            "DEEP LEARNING",
        ]
    )

    result = _preprocess_keywords(input_series)

    assert all(isinstance(kw, str) for kw in result)
    assert all(kw.islower() for kw in result)
    assert all(not kw.startswith(" ") and not kw.endswith(" ") for kw in result)
