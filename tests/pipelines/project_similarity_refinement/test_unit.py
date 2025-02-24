"""
Test file for pipeline 'project_similarity_refinement'.
"""

# pylint: skip-file
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from dsit_taxonomy.pipelines.project_similarity_refinement.nodes import (
    aggregate_scores_to_labels,
    enhance_with_zeroshot,
    refine_confidence_bins,
)


@pytest.fixture
def sample_granular_scores():
    """Create sample granular scores data."""
    return pd.DataFrame({
        "project_id": ["proj1", "proj1", "proj1", "proj2", "proj2"],
        "sentence_id": ["sent1", "sent2", "sent3", "sent4", "sent5"],
        "taxonomy_label_id": ["tax1", "tax1", "tax2", "tax1", "tax2"],
        "taxonomy_label": ["Physics", "Physics", "Chemistry", "Physics", "Chemistry"],
        "similarity_score": [0.8, 0.7, 0.6, 0.75, 0.85],
        "similarity_score_global": [0.75, 0.7, 0.65, 0.7, 0.8],
        "similarity_score_key": [0.7, np.nan, 0.6, 0.65, 0.75],
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


def test_aggregate_scores_to_labels(sample_granular_scores):
    """Test score aggregation to project-label pairs."""
    result = aggregate_scores_to_labels(
        sample_granular_scores,
        normalise_by_matches=False,
        global_q2_threshold=0.5,
        global_q3_threshold=0.75,
        local_q2_threshold=0.5,
        local_q3_threshold=0.75
    )

    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in [
        "project_id", "taxonomy_label_id", "relevance_score",
        "global_bin", "local_bin", "confidence_bin",
        "num_matching_sentences", "num_sentences"
    ])
    assert len(result) == len(sample_granular_scores.groupby(["project_id", "taxonomy_label_id"]))


@patch("dsit_taxonomy.pipelines.project_similarity_refinement.nodes.pipeline")
def test_enhance_with_zeroshot(mock_pipeline, sample_project_texts):
    """Test zero-shot enhancement of taxonomy assignments."""
    # Create mock classifier
    mock_classifier = MagicMock()
    mock_classifier.return_value = {
        "labels": ["Physics", "Chemistry"],
        "scores": [0.8, 0.6]
    }
    mock_pipeline.return_value = mock_classifier

    # Create sample aggregated scores
    aggregated_scores = pd.DataFrame({
        "project_id": ["proj1", "proj1", "proj2"],
        "taxonomy_label": ["Physics", "Chemistry", "Physics"],
        "relevance_score": [0.8, 0.6, 0.7],
        "confidence_bin": ["high", "medium", "medium"]
    })

    result = enhance_with_zeroshot(
        aggregated_scores,
        sample_project_texts,
        batch_size=1,
        model_name="test-model"
    )

    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in [
        "project_id", "taxonomy_label", "zeroshot_score", "zeroshot_bin"
    ])
    assert len(result) == len(aggregated_scores)


def test_refine_confidence_bins():
    """Test confidence bin refinement."""
    zeroshot_scores = pd.DataFrame({
        "project_id": ["proj1", "proj1", "proj2"],
        "taxonomy_label_id": ["tax1", "tax2", "tax1"],
        "taxonomy_label": ["Physics > Quantum Physics", "Chemistry > Organic", "Physics > Mechanics"],
        "relevance_score": [0.8, 0.6, 0.7],
        "num_matching_sentences": [2, 1, 1],
        "similarity_score_global": [0.75, 0.65, 0.7],
        "similarity_score_key": [0.7, 0.6, 0.65],
        "similarity_score_sent": [0.85, 0.65, 0.7],
        "num_sentences": [3, 3, 2],
        "global_bin": ["high", "medium", "medium"],
        "local_bin": ["high", "low", "medium"],
        "confidence_bin": ["high", "low", "medium"],
        "zeroshot_score": [0.85, 0.3, 0.9],
        "zeroshot_bin": ["high", "very low", "very high"]
    })

    result = refine_confidence_bins(zeroshot_scores)

    assert isinstance(result, pd.DataFrame)
    # Check all required columns are present
    assert all(col in result.columns for col in [
        "project_id", "taxonomy_label_id", "taxonomy_label",
        "relevance_score", "zeroshot_score", 
        "max_confidence", "zeroshot_favouring_confidence", "sentence_favouring_confidence",
        "similarity_score_global", "similarity_score_key", "similarity_score_sent",
        "num_matching_sentences", "num_sentences",
        "global_bin", "local_bin"
    ])
    assert len(result) == len(zeroshot_scores)
    # Check that confidence bins are properly ordered
    assert all(result["max_confidence"].isin(["very high", "high", "medium", "low", "very low"]))
    # Verify bin logic
    assert result.iloc[0]["max_confidence"] == "high"  # Both high
    assert result.iloc[1]["zeroshot_favouring_confidence"] == "low"  # Favor lower zeroshot
    assert result.iloc[2]["zeroshot_favouring_confidence"] == "very high"  # Take highest 