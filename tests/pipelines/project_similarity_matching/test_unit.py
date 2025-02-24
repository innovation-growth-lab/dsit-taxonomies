"""
Test file for pipeline 'project_similarity_matching'.
"""

# pylint: skip-file
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from dsit_taxonomy.hooks import EmbeddingsHook
from dsit_taxonomy.pipelines.project_similarity_matching.nodes import (
    document_preprocessing,
    compute_similarities,
    combine_scores,
    add_metadata,
    prune_raw_matches,
)
from dsit_taxonomy.pipelines.project_similarity_matching.utils import (
    search_batch,
    split_sentences,
)
from .test_integration import sample_data, matches, matching_parameters


class MockNode:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def hooked_data(sample_data, catalog):
    """Create hooked data with embeddings."""
    catalog.add_feed_dict({"parameters": {"embeddings_model_name": "all-MiniLM-L6-v2"}})
    node = MockNode("compute_global_matches")
    hook = EmbeddingsHook("compute_global_matches")

    hooked_data = hook.before_node_run(
        node,
        catalog,
        {
            "taxonomy.cwts.full.db": sample_data["taxonomy"],
            "projects.gtr_data.db": sample_data["projects"],
            "sentences.gtr_data.db": sample_data["sentences"],
            "keywords.gtr_data.db": sample_data["keywords"],
        },
    )

    return hooked_data


def test_document_preprocessing(sample_data):
    """Test document preprocessing and sentence splitting."""
    projects_df, sentences_df = document_preprocessing(sample_data["raw_projects"])

    # Test projects output
    assert isinstance(projects_df, pd.DataFrame)
    assert all(col in projects_df.columns for col in ["project_id", "text"])
    assert len(projects_df) == len(sample_data["raw_projects"])

    # Test sentences output
    assert isinstance(sentences_df, pd.DataFrame)
    assert all(
        col in sentences_df.columns for col in ["project_id", "uuid", "sentence_text"]
    )
    assert len(sentences_df) > len(
        sample_data["raw_projects"]
    )  # Multiple sentences per project


def test_compute_similarities(hooked_data):
    """Test similarity computation."""
    result = compute_similarities(
        taxonomy=hooked_data["taxonomy.cwts.full.db"],
        documents=hooked_data["projects.gtr_data.db"],
        batch_size=1,
        top_n=2,
        n_jobs=1,
    )

    assert isinstance(result, pd.DataFrame)
    assert all(
        col in result.columns
        for col in ["document_id", "taxonomy_label_id", "similarity_score"]
    )
    assert len(result) <= len(hooked_data["projects.gtr_data.db"]) * 2  # top_n=2


def test_add_metadata(
    matches,
    sample_data,
):
    """Test metadata addition to matches."""
    sent_results, key_results = add_metadata(
        matches["sentence_matches"],
        matches["keyword_matches"],
        sample_data["sentences"],
        sample_data["keywords"],
        sample_data["taxonomy"],
    )

    assert isinstance(sent_results, pd.DataFrame)
    assert isinstance(key_results, pd.DataFrame)

    # Check sentence results
    assert all(
        col in sent_results.columns
        for col in [
            "project_id",
            "sentence_id",
            "taxonomy_label_id",
            "taxonomy_label",
            "similarity_score",
        ]
    )
    assert len(sent_results) == len(matches["sentence_matches"])

    # Check keyword results
    assert all(
        col in key_results.columns
        for col in [
            "keyword_id",
            "keyword",
            "num_annotators",
            "project_ids",
            "taxonomy_label_id",
            "taxonomy_label",
            "similarity_score",
        ]
    )
    assert len(key_results) == len(matches["keyword_matches"])

    # Verify data consistency
    assert all(sent_results["taxonomy_label"].isin(sample_data["taxonomy"]["label"]))
    assert all(key_results["taxonomy_label"].isin(sample_data["keywords"]["keyword"]))


def test_prune_matches(matches):
    """Test match pruning."""
    pruned_sent, pruned_global, pruned_key = prune_raw_matches(
        sentence_matches=matches["sentence_matches"],
        global_matches=matches["project_matches"],
        keyword_matches=matches["keyword_matches"],
        sentence_threshold=0.7,
        global_threshold=0.7,
        keyword_threshold=0.7,
        use_quantile=False,
    )

    assert isinstance(pruned_sent, pd.DataFrame)
    assert isinstance(pruned_global, pd.DataFrame)
    assert isinstance(pruned_key, pd.DataFrame)
    assert all(
        df["similarity_score"].min() >= 0.7
        for df in [pruned_sent, pruned_global, pruned_key]
    )

def test_search_batch():
    """Test batch similarity search."""
    # Create mock data
    doc_batch = [
        {"id": "1", "vector": np.random.rand(384)},
        {"id": "2", "vector": np.random.rand(384)},
    ]
    tax_embeddings = np.random.rand(3, 384)  # 3 taxonomy labels
    tax_ids = np.array(["t1", "t2", "t3"])

    result = search_batch(doc_batch, tax_embeddings, tax_ids, top_n=2)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(doc_batch) * 2  # top_n=2
    assert all(
        col in result.columns
        for col in ["document_id", "taxonomy_label_id", "similarity_score"]
    )


def test_split_sentences():
    """Test sentence splitting."""
    nlp = MagicMock()
    nlp.return_value.sents = [
        MagicMock(text="Sentence 1."),
        MagicMock(text="Sentence 2."),
    ]

    result = split_sentences("Sentence 1. Sentence 2.", nlp)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(sent, str) for sent in result)
