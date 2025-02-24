"""
Integration test file for pipeline 'project_similarity_refinement'.
"""

# pylint: skip-file
import logging
import pytest
import pandas as pd
from kedro.io import MemoryDataset
from dsit_taxonomy.pipelines.project_similarity_refinement.pipeline import (
    create_pipeline as create_refinement_pipeline,
)


@pytest.fixture
def sample_granular_data():
    """Create sample granular scores data."""
    return pd.DataFrame({
        "project_id": ["proj1", "proj1", "proj1", "proj2", "proj2"],
        "sentence_id": ["sent1", "sent2", "sent3", "sent4", "sent5"],
        "taxonomy_label_id": ["tax1", "tax1", "tax2", "tax1", "tax2"],
        "taxonomy_label": ["Physics", "Physics", "Chemistry", "Physics", "Chemistry"],
        "similarity_score": [0.8, 0.7, 0.6, 0.75, 0.85],
        "similarity_score_global": [0.75, 0.7, 0.65, 0.7, 0.8],
        "similarity_score_key": [0.7, None, 0.6, 0.65, 0.75],
        "sentence_score": [0.85, 0.75, 0.65, 0.7, 0.8]
    })


@pytest.fixture
def sample_project_texts():
    """Create sample project texts data."""
    return pd.DataFrame({
        "project_id": ["proj1", "proj2"],
        "text": [
            "Research in physics and quantum mechanics",
            "Study of chemical reactions and compounds"
        ]
    })


@pytest.fixture
def refinement_parameters():
    """Create parameters for refinement pipeline."""
    return {
        "similarity_refinement": {
            "n_jobs": 1,
            "binning": {
                "global_q2": 0.5,
                "global_q3": 0.75,
                "local_q2": 0.5,
                "local_q3": 0.75
            },
            "normalise_by_matches": False,
            "zeroshot": {
                "batch_size": 1,
                "model_name": "tasksource/ModernBERT-large-nli"
            }
        }
    }


@pytest.mark.integration
def test_refinement_pipeline(
    caplog, sample_granular_data, sample_project_texts, 
    refinement_parameters, seq_runner, catalog
):
    """Test the complete refinement pipeline."""
    caplog.set_level(logging.DEBUG, logger="kedro")
    
    # Create pipeline for one taxonomy
    pipeline = create_refinement_pipeline().only_nodes(
        "aggregate_scores_to_labels_cwts",
        "enhance_zeroshot_cwts",
        "refine_confidence_bins_cwts"
    )

    # Add data to catalog
    catalog.add_feed_dict({
        "projects.gtr_data.cwts_scores.granular": MemoryDataset(sample_granular_data),
        "projects.gtr_data.db": MemoryDataset(sample_project_texts),
        "params:similarity_refinement.normalise_by_matches": refinement_parameters["similarity_refinement"]["normalise_by_matches"],
        "params:similarity_refinement.binning.global_q2": refinement_parameters["similarity_refinement"]["binning"]["global_q2"],
        "params:similarity_refinement.binning.global_q3": refinement_parameters["similarity_refinement"]["binning"]["global_q3"],
        "params:similarity_refinement.binning.local_q2": refinement_parameters["similarity_refinement"]["binning"]["local_q2"],
        "params:similarity_refinement.binning.local_q3": refinement_parameters["similarity_refinement"]["binning"]["local_q3"],
        "params:similarity_refinement.zeroshot.batch_size": refinement_parameters["similarity_refinement"]["zeroshot"]["batch_size"],
        "params:similarity_refinement.zeroshot.model_name": refinement_parameters["similarity_refinement"]["zeroshot"]["model_name"],
        "params:similarity_refinement.n_jobs": refinement_parameters["similarity_refinement"]["n_jobs"],
    })

    # Run pipeline
    results = seq_runner.run(pipeline, catalog)

    # Check results
    assert "projects.gtr_data.cwts_scores.final" in results

    # Verify structure of final results
    final_scores = results["projects.gtr_data.cwts_scores.final"]
    assert isinstance(final_scores, pd.DataFrame)
    assert all(col in final_scores.columns for col in [
        "project_id", "taxonomy_label", "relevance_score",
        "zeroshot_score", "max_confidence",
        "zeroshot_favouring_confidence", "sentence_favouring_confidence"
    ]) 