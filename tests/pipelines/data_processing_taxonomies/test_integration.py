"""
Integration test file for pipeline 'data_processing_taxonomies'.
"""

# pylint: skip-file
import logging
import pytest
import pandas as pd
from kedro.io import MemoryDataset
from dsit_taxonomy.pipelines.data_processing_taxonomies.pipeline import (
    create_pipeline as create_taxonomy_pipeline,
)


@pytest.fixture
def sample_taxonomies_data():
    """Create sample data for all taxonomies."""
    cwts_data = pd.DataFrame(
        {
            "domain_name": ["Science"],
            "domain_id": [1],
            "field_name": ["Physics"],
            "field_id": [10],
            "subfield_name": ["Quantum Physics"],
            "subfield_id": [101],
            "topic_name": ["Quantum Computing"],
            "topic_id": [1011],
        }
    )

    goscience_data = pd.DataFrame(
        {
            "concept_name": ["Physics", "Quantum Physics"],
            "parent_name": [None, "Physics"],
            "taxonomy_level": ["L0", "L1"],
        }
    )

    oa_concepts_data = pd.DataFrame(
        {
            "display_name": ["Physics", "Quantum Physics"],
            "openalex_id": ["https://openalex.org/C1234", "https://openalex.org/C5678"],
            "level": [0, 1],
            "parent_display_names": [None, "Physics"],
            "parent_ids": [None, "https://openalex.org/C1234"],
        }
    )

    return {
        "cwts": cwts_data,
        "goscience": goscience_data,
        "oa_concepts": oa_concepts_data,
    }


@pytest.mark.integration
def test_taxonomy_processing_pipeline(
    caplog, sample_taxonomies_data, seq_runner, catalog
):
    """Test the complete taxonomy processing pipeline."""
    # Set up logging
    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    # Create and populate catalog
    catalog.add_feed_dict(
        {
            "taxonomy.cwts.raw": MemoryDataset(sample_taxonomies_data["cwts"]),
            "taxonomy.goscience.raw": MemoryDataset(
                sample_taxonomies_data["goscience"]
            ),
            "taxonomy.oa_concepts.raw": MemoryDataset(
                sample_taxonomies_data["oa_concepts"]
            ),
        }
    )

    # Create and run pipeline
    pipeline = create_taxonomy_pipeline()
    results = seq_runner.run(pipeline, catalog)

    # Check results
    assert "taxonomy.cwts.full.db" in results
    assert "taxonomy.cwts.bottom.db" in results
    assert "taxonomy.goscience.full.db" in results
    assert "taxonomy.goscience.bottom.db" in results
    assert "taxonomy.oa_concepts.full.db" in results
    assert "taxonomy.oa_concepts.bottom.db" in results

    assert successful_run_msg in caplog.text


@pytest.mark.integration
def test_individual_taxonomy_nodes(caplog, sample_taxonomies_data, seq_runner, catalog):
    """Test each taxonomy processing node individually."""
    caplog.set_level(logging.DEBUG, logger="kedro")

    # Test CWTS pipeline
    cwts_pipeline = create_taxonomy_pipeline().only_nodes("preprocess_cwts_topics")
    catalog.add_feed_dict(
        {"taxonomy.cwts.raw": MemoryDataset(sample_taxonomies_data["cwts"])}
    )
    results = seq_runner.run(cwts_pipeline, catalog)
    assert "taxonomy.cwts.full.db" in results
    assert "taxonomy.cwts.bottom.db" in results

    # Test GO-SCIENCE pipeline
    goscience_pipeline = create_taxonomy_pipeline().only_nodes(
        "preprocess_goscience_taxonomy"
    )
    catalog.add_feed_dict(
        {"taxonomy.goscience.raw": MemoryDataset(sample_taxonomies_data["goscience"])}
    )
    results = seq_runner.run(goscience_pipeline, catalog)
    assert "taxonomy.goscience.full.db" in results
    assert "taxonomy.goscience.bottom.db" in results

    # Test OpenAlex pipeline
    oa_pipeline = create_taxonomy_pipeline().only_nodes("preprocess_oa_concepts")
    catalog.add_feed_dict(
        {
            "taxonomy.oa_concepts.raw": MemoryDataset(
                sample_taxonomies_data["oa_concepts"]
            )
        }
    )
    results = seq_runner.run(oa_pipeline, catalog)
    assert "taxonomy.oa_concepts.full.db" in results
    assert "taxonomy.oa_concepts.bottom.db" in results
