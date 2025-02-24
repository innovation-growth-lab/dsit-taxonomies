"""
Test file for pipeline 'data_processing_taxonomies'.
"""

# pylint: skip-file
import pandas as pd
import pytest
from dsit_taxonomy.pipelines.data_processing_taxonomies.nodes import (
    preprocess_cwts_topics,
    preprocess_goscience_taxonomy,
    preprocess_oa_concepts,
    _preprocess_concepts,
)


@pytest.fixture
def sample_cwts_data():
    """Create sample CWTS taxonomy data."""
    return pd.DataFrame(
        {
            "domain_name": ["Science", "Science"],
            "domain_id": [1, 1],
            "field_name": ["Physics", "Physics"],
            "field_id": [10, 10],
            "subfield_name": ["Quantum Physics", "Particle Physics"],
            "subfield_id": [101, 102],
            "topic_name": ["Quantum Computing", "Particle Acceleration"],
            "topic_id": [1011, 1021],
        }
    )


@pytest.fixture
def sample_goscience_data():
    """Create sample GO-SCIENCE taxonomy data."""
    return pd.DataFrame(
        {
            "concept_name": ["Physics", "Quantum Physics", "Quantum Computing"],
            "parent_name": [None, "Physics", "Quantum Physics"],
            "taxonomy_level": ["L0", "L1", "L2"],
        }
    )


@pytest.fixture
def sample_oa_concepts_data():
    """Create sample OpenAlex concepts data."""
    return pd.DataFrame(
        {
            "display_name": ["Physics", "Quantum Physics", "Quantum Computing"],
            "openalex_id": ["C1234", "C5678", "C9012"],
            "level": [0, 1, 2],
            "parent_display_names": [None, "Physics", "Quantum Physics"],
            "parent_ids": [None, "C1234", "C5678"],
        }
    )


def test_preprocess_cwts_topics(sample_cwts_data):
    """Test CWTS taxonomy preprocessing."""
    taxonomy, bottom_level = preprocess_cwts_topics(sample_cwts_data)

    # Test full taxonomy
    assert isinstance(taxonomy, pd.DataFrame)
    assert all(col in taxonomy.columns for col in ["label", "id_path", "level", "uuid"])
    assert len(taxonomy) > len(bottom_level)  # Full taxonomy has more entries

    # Test bottom level
    assert isinstance(bottom_level, pd.DataFrame)
    assert all(
        col in bottom_level.columns for col in ["label", "id_path", "level", "uuid"]
    )
    assert all(bottom_level["level"] == bottom_level["level"].max())


def test_preprocess_goscience_taxonomy(sample_goscience_data):
    """Test GO-SCIENCE taxonomy preprocessing."""
    taxonomy, terminal_nodes = preprocess_goscience_taxonomy(sample_goscience_data)

    # Test full taxonomy
    assert isinstance(taxonomy, pd.DataFrame)
    assert all(
        col in taxonomy.columns for col in ["label", "level_path", "level", "uuid"]
    )

    # Test terminal nodes
    assert isinstance(terminal_nodes, pd.DataFrame)
    assert len(terminal_nodes) < len(taxonomy)
    assert (
        "Quantum Computing" in terminal_nodes["label"].str.split(" > ").str[-1].values
    )


def test_preprocess_oa_concepts(sample_oa_concepts_data):
    """Test OpenAlex concepts preprocessing."""
    taxonomy, bottom_level = preprocess_oa_concepts(sample_oa_concepts_data)

    # Test full taxonomy
    assert isinstance(taxonomy, pd.DataFrame)
    assert all(col in taxonomy.columns for col in ["label", "id_path", "level", "uuid"])

    # Test bottom level
    assert isinstance(bottom_level, pd.DataFrame)
    assert len(bottom_level) < len(taxonomy)
    assert all(bottom_level["label"].str.contains("Quantum Computing"))


def test_preprocess_concepts(sample_oa_concepts_data):
    """Test OpenAlex concepts preprocessing helper function."""
    processed_df = _preprocess_concepts(sample_oa_concepts_data)

    assert isinstance(processed_df, pd.DataFrame)
    assert "openalex_id" in processed_df.columns
    assert not processed_df["openalex_id"].str.contains("https://openalex.org/").any()
