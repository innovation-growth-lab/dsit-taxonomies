"""
This module contains nodes for the keyword similarity matching pipeline.

The pipeline performs semantic similarity matching between research project texts
and taxonomy labels at multiple levels:
- Global project-level matching using full document text
- Sentence-level matching to identify relevant passages 
- Keyword-level matching to boost scores based on key terms

The nodes handle:
- Document preprocessing and sentence splitting
- Computing similarity scores between texts and taxonomy labels
- Pruning low-confidence matches
- Combining and weighting scores from different matching approaches
"""

import logging
import uuid
import pandas as pd
from spacy.lang.en import English
from joblib import Parallel, delayed
import numpy as np
from .utils import search_batch, split_sentences

logger = logging.getLogger(__name__)


def document_preprocessing(documents: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame of documents.

    Args:
        documents: DataFrame containing the GtR documents with text columns

    Returns:
        Tuple containing:
            - DataFrame with project_id and combined text
            - DataFrame with project_id, uuid and individual sentences
    """
    nlp = English()
    nlp.add_pipe("sentencizer")

    # create a unique column combining all text columns
    documents["text"] = documents.apply(
        lambda row: ". ".join(
            row[col] if row[col] is not None else ""
            for col in [
                "title",
                "abstract_text",
                "tech_abstract_text",
                "potential_impact",
            ]
        ),
        axis=1,
    )

    # split documents into sentences
    documents["sentence_text"] = documents["text"].apply(split_sentences, nlp=nlp)

    # explode sentences into separate rows
    sentences = documents.explode("sentence_text")

    # add uuids
    sentences["uuid"] = sentences["sentence_text"].apply(
        lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, x))
    )

    # [HACK] drop duplicate rows to avoid non-informational matches
    # This of course has the downside that it may remove some valid matches.
    sentences = sentences.drop_duplicates(subset=["sentence_text"], keep=False)

    return (
        documents[["project_id", "text"]],
        sentences[["project_id", "uuid", "sentence_text"]],
    )


def compute_similarities(
    taxonomy: pd.DataFrame,
    documents: pd.DataFrame,
    batch_size: int = 1000,
    top_n: int = 10,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """
    Compute similarity scores between documents and taxonomy labels.

    Args:
        taxonomy: DataFrame containing taxonomy labels and their embeddings
        documents: DataFrame containing documents to match
        batch_size: Number of documents to process in each batch
        top_n: Number of top matches to retain per document
        n_jobs: Number of parallel jobs to run

    Returns:
        DataFrame with similarity scores between documents and labels
    """
    # Convert to numpy arrays
    taxonomy_embeddings = np.vstack(taxonomy["vector"].values)
    taxonomy_ids = taxonomy["id"].values
    documents_dict = documents.to_dict(orient="records")

    # Divide documents into batches
    document_batches = [
        documents_dict[i : i + batch_size]
        for i in range(0, len(documents_dict), batch_size)
    ]

    # Process batches in parallel
    logger.info("Processing %d batches in parallel", len(document_batches))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(search_batch)(batch, taxonomy_embeddings, taxonomy_ids, top_n)
        for batch in document_batches
    )

    logger.info("Flattening results")
    return pd.concat(results, ignore_index=True)


def add_metadata(
    sentence_scores: pd.DataFrame,
    keyword_scores: pd.DataFrame,
    sentence_db: pd.DataFrame,
    keyword_db: pd.DataFrame,
    taxonomy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add metadata to the sentence and keyword similarity scores.

    Args:
        sentence_scores: Raw similarity scores for sentences
        keyword_scores: Raw similarity scores for keywords
        sentence_db: DataFrame with sentence metadata
        keyword_db: DataFrame with keyword metadata
        taxonomy: DataFrame with taxonomy label metadata

    Returns:
        Tuple containing:
            - DataFrame with enriched sentence scores
            - DataFrame with enriched keyword scores
    """
    logger.info("Adding metadata to sentence scores")
    sentence_scores = (
        pd.merge(
            sentence_scores,
            sentence_db[["uuid", "project_id"]],
            left_on="document_id",
            right_on="uuid",
            how="inner",
        )
        .rename(columns={"document_id": "sentence_id"})[
            ["project_id", "sentence_id", "taxonomy_label_id", "similarity_score"]
        ]
        .merge(
            taxonomy[["uuid", "label"]],
            left_on="taxonomy_label_id",
            right_on="uuid",
            how="left",
        )
        .rename(columns={"label": "taxonomy_label"})[
            [
                "project_id",
                "sentence_id",
                "taxonomy_label_id",
                "taxonomy_label",
                "similarity_score",
            ]
        ]
    )

    logger.info("Adding metadata to keyword scores")

    keyword_scores = (
        pd.merge(
            keyword_scores,
            keyword_db,
            left_on="document_id",
            right_on="uuid",
            how="left",
        )
        .rename(columns={"document_id": "keyword_id"})
        .merge(
            taxonomy[["uuid", "label"]],
            left_on="taxonomy_label_id",
            right_on="uuid",
            how="left",
        )
        .rename(columns={"label": "taxonomy_label"})[
            [
                "keyword_id",
                "keyword",
                "num_annotators",
                "project_ids",
                "taxonomy_label_id",
                "taxonomy_label",
                "similarity_score",
            ]
        ]
    )

    return sentence_scores, keyword_scores


def prune_raw_matches(
    sentence_matches: pd.DataFrame,
    global_matches: pd.DataFrame,
    keyword_matches: pd.DataFrame,
    sentence_threshold: float,
    global_threshold: float,
    keyword_threshold: float,
    use_quantile: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prune low-scoring matches from all three sources based on quantile thresholds or
    direct score values.

    Args:
        sentence_matches: DataFrame with raw sentence similarity matches
        global_matches: DataFrame with raw project-level matches
        keyword_matches: DataFrame with raw keyword matches
        sentence_threshold: Quantile threshold or score value for sentence matches
        global_threshold: Quantile threshold or score value for global matches
        keyword_threshold: Quantile threshold or score value for keyword matches
        use_quantile: Boolean flag to use thresholds as quantiles (True) or direct score
            values (False)

    Returns:
        Tuple of (pruned_sentence_matches, pruned_global_matches, pruned_keyword_matches)
    """
    logger.info(
        "Pruning matches with thresholds - Sentence: %.2f, Global: %.2f, Keyword: %.2f",
        sentence_threshold,
        global_threshold,
        keyword_threshold,
    )

    if use_quantile:
        # Prune sentence matches using quantile
        sent_threshold = sentence_matches["similarity_score"].quantile(
            sentence_threshold
        )
        pruned_sentences = sentence_matches[
            sentence_matches["similarity_score"] >= sent_threshold
        ]

        # Prune global matches using quantile
        global_threshold_val = global_matches["similarity_score"].quantile(
            global_threshold
        )
        pruned_global = global_matches[
            global_matches["similarity_score"] >= global_threshold_val
        ]

        # Prune keyword matches using quantile
        key_threshold = keyword_matches["similarity_score"].quantile(keyword_threshold)
        pruned_keywords = keyword_matches[
            keyword_matches["similarity_score"] >= key_threshold
        ]
    else:
        # Prune sentence matches using direct score value
        pruned_sentences = sentence_matches[
            sentence_matches["similarity_score"] >= sentence_threshold
        ]

        # Prune global matches using direct score value
        pruned_global = global_matches[
            global_matches["similarity_score"] >= global_threshold
        ]

        # Prune keyword matches using direct score value
        pruned_keywords = keyword_matches[
            keyword_matches["similarity_score"] >= keyword_threshold
        ]

    logger.info(
        "Pruning results:\n"
        "Sentences: %d -> %d (%.1f%%)\n"
        "Global: %d -> %d (%.1f%%)\n"
        "Keywords: %d -> %d (%.1f%%)",
        len(sentence_matches),
        len(pruned_sentences),
        100 * len(pruned_sentences) / len(sentence_matches),
        len(global_matches),
        len(pruned_global),
        100 * len(pruned_global) / len(global_matches),
        len(keyword_matches),
        len(pruned_keywords),
        100 * len(pruned_keywords) / len(keyword_matches),
    )

    return pruned_sentences, pruned_global, pruned_keywords


def combine_scores(
    sentence_scores: pd.DataFrame,
    keyword_scores: pd.DataFrame,
    global_scores: pd.DataFrame,
    sentence_weight: float,
    global_weight: float,
) -> pd.DataFrame:
    """
    Compute granular sentence-level scores with weighted boosts.

    Args:
        sentence_scores: DataFrame with sentence-level similarity scores
        keyword_scores: DataFrame with keyword-level similarity scores
        global_scores: DataFrame with project-level similarity scores
        sentence_weight: Weight for sentence-level scores (alpha)
        global_weight: Weight for global-level scores (beta)
            Note: Keyword weight is (1-alpha-beta)

    Returns:
        DataFrame with combined scores including:
            - Base sentence score
            - Global boost from project matches
            - Keyword boost from keyword matches
    """
    logger.info(
        "Computing granular scores with weights - Sentence: %.2f, Global: %.2f, Keyword: %.2f",
        sentence_weight,
        global_weight,
        1 - sentence_weight - global_weight,
    )

    global_scores.rename(columns={"document_id": "project_id"}, inplace=True)

    # Start with sentence scores
    granular_scores = sentence_scores.copy()

    # Add global boost
    granular_scores = pd.merge(
        granular_scores,
        global_scores[["project_id", "taxonomy_label_id", "similarity_score"]],
        on=["project_id", "taxonomy_label_id"],
        how="left",
        suffixes=("", "_global"),
    )

    # Add keyword boost (max similarity across project keywords)
    keyword_scores_exploded = keyword_scores.explode("project_ids").rename(
        columns={"project_ids": "project_id"}
    )
    keyword_max_scores = (
        keyword_scores_exploded.groupby(["project_id", "taxonomy_label_id"])[
            "similarity_score"
        ]
        .max()
        .reset_index()
        .rename(columns={"similarity_score": "similarity_score_key"})
    )

    granular_scores = pd.merge(
        granular_scores,
        keyword_max_scores,
        on=["project_id", "taxonomy_label_id"],
        how="left",
    )

    # Compute sentence-level score with weighted boosts
    granular_scores["sentence_score"] = (
        sentence_weight * granular_scores["similarity_score"]
        + global_weight * granular_scores["similarity_score_global"].fillna(0)
        + (1 - sentence_weight - global_weight)
        * granular_scores["similarity_score_key"].fillna(0)
    )

    return granular_scores
