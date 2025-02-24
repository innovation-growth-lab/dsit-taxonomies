"""
Integration test file for pipeline 'keyword_processing_gtr'.
"""

# pylint: skip-file
import logging
import pytest
import pandas as pd
import numpy as np
from kedro.io import MemoryDataset
from unittest.mock import MagicMock
from dsit_taxonomy.pipelines.keyword_processing_gtr.pipeline import (
    create_pipeline as create_keyword_pipeline,
)


@pytest.fixture
def sample_projects_data():
    """Create sample project data for testing."""
    return pd.DataFrame(
        {
            "project_id": ["1", "2", "3"],
            "title": ["AI Project", "ML Study", "Data Science Research"],
            "abstract_text": [
                "This is an AI study",
                "ML research paper",
                "Data science research",
            ],
            "tech_abstract_text": ["Technical details", "More technical info", ""],
            "potential_impact": ["High impact", "Medium impact", ""],
        }
    )

@pytest.fixture
def sample_keywords_data():
    """Create sample keywords data from different extractors."""
    dbp_data = pd.DataFrame(
        {
            "project_id": ["1", "2"],
            "dbp_keywords": [
                ["artificial intelligence", "machine learning"],
                ["neural networks", "deep learning"],
            ],
        }
    )

    rake_data = pd.DataFrame(
        {
            "project_id": ["1", "2"],
            "rake_keywords": [
                ["machine learning", "AI research"],
                ["neural networks", "ML models"],
            ],
        }
    )

    yake_data = pd.DataFrame(
        {
            "project_id": ["1", "2"],
            "yake_keywords": [
                ["AI", "machine learning"],
                ["neural networks", "deep learning"],
            ],
        }
    )

    keybert_data = pd.DataFrame(
        {
            "project_id": ["1", "2"],
            "keybert_keywords": [
                ["artificial intelligence", "ML"],
                ["neural networks", "deep learning"],
            ],
        }
    )

    return {
        "dbp": dbp_data,
        "rake": rake_data,
        "yake": yake_data,
        "keybert": keybert_data,
    }


@pytest.mark.integration
def test_annotation_pipelines(caplog, sample_projects_data, seq_runner, catalog):
    """Test the DBpedia, RAKE and YAKE annotation nodes of the pipeline."""
    # Set up logging
    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    # Create and populate catalog
    catalog.add_feed_dict(
        {
            "gtr.data_collection.projects.intermediate": sample_projects_data,
            "dbp.gtr_data.annotated.oracle": pd.DataFrame(
                columns=["project_id", "dbp_keywords"]
            ),
            "yake.gtr_data.annotated.oracle": pd.DataFrame(
                columns=["project_id", "yake_keywords"]
            ),
            "rake.gtr_data.annotated.oracle": pd.DataFrame(
                columns=["project_id", "rake_keywords"]
            ),
        }
    )

    # Create and run DBpedia pipeline
    dbp_pipeline = (
        create_keyword_pipeline()
        .from_nodes("dbp_annotate_data")
        .to_nodes("dbp_annotate_data")
    )
    seq_runner.run(dbp_pipeline, catalog)
    assert successful_run_msg in caplog.text

    # Create and run RAKE pipeline
    rake_pipeline = (
        create_keyword_pipeline()
        .from_nodes("rake_annotate_data")
        .to_nodes("rake_annotate_data")
    )
    seq_runner.run(rake_pipeline, catalog)
    assert successful_run_msg in caplog.text

    # Create and run YAKE pipeline
    yake_pipeline = (
        create_keyword_pipeline()
        .from_nodes("yake_annotate_data")
        .to_nodes("yake_annotate_data")
    )
    seq_runner.run(yake_pipeline, catalog)
    assert successful_run_msg in caplog.text


@pytest.mark.integration
def test_annotation_pipeline_keybert(caplog, sample_projects_data, seq_runner, catalog):
    """Test the KeyBERT annotation nodes of the pipeline."""
    # First test keybert_annotate_data node
    pipeline_keybert = (
        create_keyword_pipeline()
        .from_nodes("keybert_annotate_data")
        .to_nodes("keybert_annotate_data")
    )

    # Set up logging
    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    # Create and populate catalog for keybert annotation
    catalog.add_feed_dict(
        {
            "gtr.data_collection.projects.intermediate": sample_projects_data,
            "keybert.gtr_data.annotated.oracle": pd.DataFrame(
                columns=["project_id", "keybert_keywords"]
            ),
            "n_jobs": 1,
        }
    )

    # Run keybert annotation
    results = seq_runner.run(pipeline_keybert, catalog)

    # Convert the results to mock datasets using MagicMock
    partitioned_results = {
        k: MagicMock(return_value=v)
        for k, v in results["keybert.gtr_data.annotated.ptd"].items()
    }

    # Test concatenation node
    pipeline_concat = (
        create_keyword_pipeline()
        .from_nodes("concatenate_keybert_partitions")
        .to_nodes("concatenate_keybert_partitions")
    )

    # Update catalog with mock datasets
    catalog.add_feed_dict(
        {"keybert.gtr_data.annotated.ptd": MemoryDataset(partitioned_results)}
    )

    # Run concatenation
    seq_runner.run(pipeline_concat, catalog)

    assert successful_run_msg in caplog.text


@pytest.mark.integration
def test_keyword_processing_pipeline(caplog, sample_keywords_data, seq_runner, catalog):
    """Test the complete keyword processing pipeline."""
    # Set up logging
    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    # Create and populate catalog
    catalog.add_feed_dict(
        {
            "dbp.gtr_data.annotated": MemoryDataset(sample_keywords_data["dbp"]),
            "rake.gtr_data.annotated": MemoryDataset(sample_keywords_data["rake"]),
            "yake.gtr_data.annotated": MemoryDataset(sample_keywords_data["yake"]),
            "keybert.gtr_data.annotated": MemoryDataset(
                sample_keywords_data["keybert"]
            ),
        }
    )

    # Create and run pipeline
    pipeline = create_keyword_pipeline().only_nodes("aggregate_keyword_annotators")
    results = seq_runner.run(pipeline, catalog)

    # Check results
    assert "keywords.gtr_data.db" in results

    # Verify aggregated keywords structure
    keywords_db = results["keywords.gtr_data.db"]
    assert isinstance(keywords_db, pd.DataFrame)
    assert all(
        col in keywords_db.columns
        for col in ["keyword", "num_annotators", "project_ids", "uuid"]
    )

    assert successful_run_msg in caplog.text