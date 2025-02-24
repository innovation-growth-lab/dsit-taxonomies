"""
Integration test file for pipeline 'project_similarity_matching'.
"""

# pylint: skip-file
import logging
import pytest
import pandas as pd
import numpy as np
from kedro.io import MemoryDataset
from unittest.mock import MagicMock
from dsit_taxonomy.hooks import EmbeddingsHook
from dsit_taxonomy.pipelines.project_similarity_matching.pipeline import (
    create_pipeline as create_matching_pipeline,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""

    # Raw projects data
    raw_projects = pd.DataFrame(
        {
            "project_id": ["proj1", "proj2", "proj3", "proj4", "proj5"],
            "title": [
                "AI Research",
                "ML Study",
                "Data Science Project",
                "Computer Vision Research",
                "NLP Study",
            ],
            "abstract_text": [
                "AI Research. Study on artificial intelligence. Technical implementation details.",
                "Machine learning study. Neural network research. Model deployment.",
                "Data science analytics. Big data processing. Statistical mining.",
                "Computer vision algorithms. Image recognition. CNN architectures.", 
                "Natural language models. Text classification. Transformer networks.",
            ],
            "tech_abstract_text": [
                "Deep learning advances. Reinforcement learning methods. AI optimization.",
                "Supervised learning approaches. Feature engineering. Cross-validation.",
                "Data pipeline design. ETL workflows. Distributed computing.",
                "Object detection systems. Video processing. Transfer learning.",
                "Language understanding. Named entity recognition. Sentiment analysis.",
            ],
            "potential_impact": [
                "High impact",
                "Medium impact",
                "High impact",
                "Medium impact",
                "High impact",
            ],
        }
    )

    # Projects data
    projects = pd.DataFrame(
        {
            "project_id": ["proj1", "proj2", "proj3", "proj4", "proj5"],
            "text": [
                "AI Research. Study on AI. Technical details. High impact",
                "ML Study. Machine learning research. ML implementation. Medium impact",
                "Data Science Project. Big data analytics. Data mining. High impact",
                "Computer Vision Research. Image processing. Deep learning. Medium impact",
                "NLP Study. Natural language processing. Text mining. High impact",
            ],
        }
    )

    sentences = pd.DataFrame(
        {
            "project_id": [
                "proj1",
                "proj1",
                "proj2",
                "proj2",
                "proj3",
                "proj4",
                "proj5",
            ],
            "uuid": ["sent1", "sent2", "sent3", "sent4", "sent5", "sent6", "sent7"],
            "sentence_text": [
                "AI Research. Study on AI. Technical details. High impact",
                "ML Study. Machine learning research. ML implementation. Medium impact",
                "Data Science Project. Big data analytics. Data mining. High impact",
                "Computer Vision Research. Image processing. Deep learning. Medium impact",
                "NLP Study. Natural language processing. Text mining. High impact",
                "ML Study 2. New Machine learning research. ML implementation 2. Medium impact 2",
                "Data Science Project 2. New Big data analytics. Data mining 2. High impact 2",
            ],
        }
    )

    keywords = pd.DataFrame(
        {
            "keyword": ["AI", "ML", "Data Science", "Computer Vision", "NLP"],
            "num_annotators": [3, 4, 3, 4, 3],
            "project_ids": [["proj1"], ["proj1"], ["proj2"], ["proj2"], ["proj3"]],
            "uuid": ["key1", "key2", "key3", "key4", "key5"],
        }
    )

    # Taxonomy data with mock embeddings
    taxonomy = pd.DataFrame(
        {
            "label": ["AI", "ML", "Data Science", "Computer Vision", "NLP"],
            "id": ["1", "2", "3", "4", "5"],
            "level": [0, 0, 0, 0, 0],
            "uuid": ["uuid1", "uuid2", "uuid3", "uuid4", "uuid5"],
        }
    )

    return {
        "raw_projects": raw_projects,
        "projects": projects,
        "sentences": sentences,
        "keywords": keywords,
        "taxonomy": taxonomy,
    }


@pytest.fixture
def matches():
    """Create mock matches data."""

    project_matches = pd.DataFrame(
        {
            "document_id": ["proj1", "proj1", "proj2", "proj2", "proj3"],
            # "uuid": ["uuid1", "uuid2", "uuid3", "uuid4", "uuid5"],
            "taxonomy_label_id": ["uuid1", "uuid2", "uuid3", "uuid4", "uuid5"],
            "similarity_score": [0.8, 0.7, 0.9, 0.6, 0.75],
        }
    )

    sentence_matches = pd.DataFrame(
        {
            "document_id": [
                "sent1",
                "sent2",
                "sent3",
                "sent4",
                "sent5",
                "sent6",
                "sent7",
            ],
            "taxonomy_label_id": [
                "uuid1",
                "uuid2",
                "uuid3",
                "uuid4",
                "uuid5",
                "uuid2",
                "uuid3",
            ],
            "similarity_score": [0.8, 0.7, 0.9, 0.6, 0.75, 0.8, 0.7],
        }
    )

    keyword_matches = pd.DataFrame(
        {
            "document_id": ["key1", "key2", "key3", "key4", "key5"],
            "taxonomy_label_id": ["uuid1", "uuid2", "uuid3", "uuid4", "uuid5"],
            "similarity_score": [0.8, 0.7, 0.9, 0.6, 0.75],
        }
    )

    return {
        "project_matches": project_matches,
        "sentence_matches": sentence_matches,
        "keyword_matches": keyword_matches,
    }


@pytest.fixture
def matching_parameters():
    """Create parameters for matching pipeline."""
    return {
        "similarity_matching": {
            "projects": {"batch_size": 2, "top_n": 2},
            "sentences": {"batch_size": 2, "top_n": 2},
            "keywords": {"batch_size": 2, "top_n": 2},
            "pruning": {
                "use_quantile": False,
                "sentence_threshold": 0.5,
                "global_threshold": 0.5,
                "keyword_threshold": 0.5,
            },
            "score_weights": {
                "sentence_weight": 0.6,
                "global_weight": 0.3,
            },
        }
    }


class MockNode:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def hooked_data(sample_data, catalog):
    """Create hooked data for testing."""
    # Recreate the embeddings hook
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


@pytest.mark.integration
def test_document_preprocessing_node(caplog, sample_data, seq_runner, catalog):
    """Test the document preprocessing node."""
    caplog.set_level(logging.DEBUG, logger="kedro")

    # Create pipeline for preprocessing only
    pipeline = create_matching_pipeline().only_nodes("document_preprocessing")

    # Add data to catalog
    catalog.add_feed_dict(
        {
            "gtr.projects.documents": MemoryDataset(sample_data["raw_projects"]),
        }
    )

    # Run pipeline
    results = seq_runner.run(pipeline, catalog)

    assert "projects.gtr_data.db" in results
    assert "sentences.gtr_data.db" in results


@pytest.mark.integration
def test_similarity_computation_pipeline(
    caplog, hooked_data, matching_parameters, seq_runner, catalog
):
    """Test the similarity computation nodes."""
    caplog.set_level(logging.DEBUG, logger="kedro")

    # Create pipeline for one taxonomy
    pipeline = create_matching_pipeline().only_nodes(
        "compute_global_matches_cwts",
        "compute_sentence_matches_cwts",
        "compute_keyword_matches_cwts",
    )

    # Add data to catalog
    catalog.add_feed_dict(
        {
            **hooked_data,
            "params:similarity_matching.projects.batch_size": matching_parameters[
                "similarity_matching"
            ]["projects"]["batch_size"],
            "params:similarity_matching.projects.top_n": matching_parameters[
                "similarity_matching"
            ]["projects"]["top_n"],
            "params:similarity_matching.sentences.batch_size": matching_parameters[
                "similarity_matching"
            ]["sentences"]["batch_size"],
            "params:similarity_matching.sentences.top_n": matching_parameters[
                "similarity_matching"
            ]["sentences"]["top_n"],
            "params:similarity_matching.keywords.batch_size": matching_parameters[
                "similarity_matching"
            ]["keywords"]["batch_size"],
            "params:similarity_matching.keywords.top_n": 1,
            "params:similarity_matching.n_jobs": 1,
        }
    )

    # Run pipeline
    results = seq_runner.run(pipeline, catalog)

    assert "projects.gtr_data.cwts_matches.raw" in results
    assert "sentences.gtr_data.cwts_matches.raw" in results
    assert "keywords.gtr_data.cwts_matches.raw" in results


@pytest.mark.integration
def test_scoring_pipeline(
    caplog, sample_data, matches, matching_parameters, seq_runner, catalog
):
    """Test the scoring nodes."""
    caplog.set_level(logging.DEBUG, logger="kedro")

    # Create pipeline for scoring
    pipeline = create_matching_pipeline().only_nodes(
        "add_metadata_cwts",
        "prune_raw_matches_cwts",
        "combine_scores_cwts",
    )

    # Add data to catalog
    catalog.add_feed_dict(
        {
            "sentences.gtr_data.cwts_matches.raw": MemoryDataset(
                matches["sentence_matches"]
            ),
            "keywords.gtr_data.cwts_matches.raw": MemoryDataset(
                matches["keyword_matches"]
            ),
            "projects.gtr_data.cwts_matches.raw": MemoryDataset(
                matches["project_matches"]
            ),
            "sentences.gtr_data.db": MemoryDataset(sample_data["sentences"]),
            "keywords.gtr_data.db": MemoryDataset(sample_data["keywords"]),
            "taxonomy.cwts.full.db": MemoryDataset(sample_data["taxonomy"]),
            "params:similarity_matching.pruning.sentence_threshold": matching_parameters[
                "similarity_matching"
            ][
                "pruning"
            ][
                "sentence_threshold"
            ],
            "params:similarity_matching.pruning.global_threshold": matching_parameters[
                "similarity_matching"
            ]["pruning"]["global_threshold"],
            "params:similarity_matching.pruning.keyword_threshold": matching_parameters[
                "similarity_matching"
            ]["pruning"]["keyword_threshold"],
            "params:similarity_matching.pruning.use_quantile": matching_parameters[
                "similarity_matching"
            ]["pruning"]["use_quantile"],
            "params:similarity_matching.score_weights.sentence_weight": matching_parameters[
                "similarity_matching"
            ][
                "score_weights"
            ][
                "sentence_weight"
            ],
            "params:similarity_matching.score_weights.global_weight": matching_parameters[
                "similarity_matching"
            ][
                "score_weights"
            ][
                "global_weight"
            ],
        }
    )

    # Run pipeline
    results = seq_runner.run(pipeline, catalog)

    assert "projects.gtr_data.cwts_scores.granular" in results
