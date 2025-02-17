import logging
import uuid
from typing import List, Dict
import pandas as pd
from spacy.lang.en import English
from joblib import Parallel, delayed
from functools import partial
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)


def document_preprocessing(documents: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame of documents.

    Args:
        documents: DataFrame containing the GtR documents.

    Returns:
        pd.DataFrame: DataFrame containing the preprocessed text column.
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
    documents["sentence_text"] = documents["text"].apply(_split_sentences, nlp=nlp)

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
) -> pd.DataFrame:
    """Compute similarity scores using parallel processing."""
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
    results = Parallel(n_jobs=8, verbose=10)(
        delayed(_search_batch)(batch, taxonomy_embeddings, taxonomy_ids, top_n)
        for batch in document_batches
    )

    logger.info("Flattening results")
    return pd.concat(results, ignore_index=True)


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
    Prune low-scoring matches from all three sources based on quantile thresholds or direct score values.

    Args:
        sentence_matches: DataFrame with raw sentence similarity matches
        global_matches: DataFrame with raw project-level matches
        keyword_matches: DataFrame with raw keyword matches
        sentence_threshold: Quantile threshold or score value for sentence matches
        global_threshold: Quantile threshold or score value for global matches
        keyword_threshold: Quantile threshold or score value for keyword matches
        use_quantile: Boolean flag to use thresholds as quantiles (True) or direct score values (False)

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


def add_metadata(
    sentence_scores: pd.DataFrame,
    keyword_scores: pd.DataFrame,
    sentence_db: pd.DataFrame,
    keyword_db: pd.DataFrame,
    taxonomy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add metadata to the combined scores.
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


def combine_scores(
    sentence_scores: pd.DataFrame,
    keyword_scores: pd.DataFrame,
    global_scores: pd.DataFrame,
    sentence_weight: float,  # sentence weight
    global_weight: float,  # global weight
) -> pd.DataFrame:
    """
    Compute granular sentence-level scores with boosts.

    For each (sentence, label) pair, computes:
    - Base sentence score (weight alpha)
    - Global boost if label appears in project's top candidates (weight beta)
    - Keyword boost if label appears in project's keyword matches (weight 1-alpha-beta)
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


def aggregate_scores_to_labels(
    granular_scores: pd.DataFrame, normalise_by_matches: bool = False, **binning_params
) -> pd.DataFrame:
    """
    Aggregate sentence-level scores to project-label pairs.

    Args:
        granular_scores: DataFrame with sentence-level scores
        normalise_by_matches: If True, weight scores by ratio of matching sentences
        **binning_params: Parameters for confidence binning
    """
    logger.info("Aggregating sentence-level scores to project-label pairs")

    # Get total sentences per project
    project_sentence_counts = (
        granular_scores.groupby("project_id")["sentence_id"]
        .nunique()
        .reset_index()
        .rename(columns={"sentence_id": "num_sentences"})
    )

    # Aggregate to project-label level
    aggregated = (
        granular_scores.groupby(["project_id", "taxonomy_label_id"])
        .agg(
            {
                "sentence_score": ["max", "count"],
                "taxonomy_label": "first",
                "similarity_score_global": "first",
                "similarity_score_key": "first",
                "similarity_score": "first",
            }
        )
        .reset_index()
    )

    # Rename columns
    aggregated.columns = [
        "project_id",
        "taxonomy_label_id",
        "relevance_score",
        "num_matching_sentences",
        "taxonomy_label",
        "similarity_score_global",
        "similarity_score_key",
        "similarity_score_sent",
    ]

    # Add total sentence count per project
    aggregated = pd.merge(
        aggregated, project_sentence_counts, on="project_id", how="left"
    )

    # Apply global binning first
    q2 = aggregated["relevance_score"].quantile(binning_params["global_q2_threshold"])
    q3 = aggregated["relevance_score"].quantile(binning_params["global_q3_threshold"])
    aggregated["global_bin"] = aggregated["relevance_score"].apply(
        lambda x: "high" if x > q3 else ("medium" if x > q2 else "low")
    )

    # Normalise scores within projects
    def normalise_project_scores(group):
        # Weight by matching sentence ratio
        group["relevance_score"] = group["relevance_score"] * (
            group["num_matching_sentences"] / group["num_sentences"]
        )
        return group

    if normalise_by_matches:
        logger.info("Normalising scores within projects")
        aggregated = (
            aggregated.groupby("project_id")
            .apply(normalise_project_scores)
            .reset_index(drop=True)
        )

    logger.info(
        "Score normalisation summary:\n"
        "Mean project score sum: %.3f\n"
        "Score distribution:\n%s",
        aggregated.groupby("project_id")["relevance_score"].sum().mean(),
        aggregated["relevance_score"].describe().to_string(),
    )

    # Complete binning with local thresholds only
    return _assign_local_bins(aggregated, **binning_params)


def _search_batch(
    document_batch: List[Dict[str, str]],
    taxonomy_embeddings: np.ndarray,
    taxonomy_ids: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Perform vectorized similarity search for a batch of documents.

    Args:
        document_batch: List of document embeddings
        taxonomy_embeddings: Array of taxonomy label embeddings (n_labels, embedding_dim)
        taxonomy_ids: Array of taxonomy label IDs
        top_n: Number of top matches to retain

    Returns:
        DataFrame with similarity matches
    """
    # Stack document embeddings
    doc_embeddings = np.vstack([doc["vector"] for doc in document_batch])
    doc_ids = [doc["id"] for doc in document_batch]

    # Compute cosine similarity matrix
    # (num_docs, embedding_dim) @ (embedding_dim, num_labels) = (num_docs, num_labels)
    similarities = doc_embeddings @ taxonomy_embeddings.T

    # Normalise for cosine similarity
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    tax_norms = np.linalg.norm(taxonomy_embeddings, axis=1, keepdims=True).T
    similarities = similarities / (doc_norms @ tax_norms)

    # Get top N indices and scores for each document
    top_indices = np.argpartition(-similarities, top_n, axis=1)[:, :top_n]

    results = []
    for i, doc_id in enumerate(doc_ids):
        top_idx = top_indices[i]
        scores = similarities[i, top_idx]

        # Sort by score
        sort_idx = np.argsort(-scores)
        top_idx = top_idx[sort_idx]
        scores = scores[sort_idx]

        results.append(
            pd.DataFrame(
                {
                    "document_id": doc_id,
                    "taxonomy_label_id": taxonomy_ids[top_idx],
                    "similarity_score": scores,
                }
            )
        )

    return pd.concat(results, ignore_index=True)


def _split_sentences(document: str, nlp: English) -> List[str]:
    """Split a document into sentences."""
    doc = nlp(document)
    return [sent.text for sent in doc.sents]


def _process_project_group(
    group: pd.DataFrame,
    local_q2_threshold: float,
    local_q3_threshold: float,
) -> pd.Series:
    """Process a single project group for local binning."""
    n_labels = len(group)

    # Handle edge cases for small groups
    if n_labels == 1:
        return pd.Series(["high"], index=group.index)
    elif n_labels == 2:
        sorted_idx = group["relevance_score"].sort_values(ascending=False).index
        return pd.Series(["high", "medium"], index=sorted_idx)

    # Sort scores in descending order
    sorted_scores = group["relevance_score"].sort_values(ascending=False)

    # Compute project-specific quantiles
    q2_local = sorted_scores.quantile(local_q2_threshold)
    q3_local = sorted_scores.quantile(local_q3_threshold)

    def _assign_quantile_bin(score):
        if score > q3_local:
            return "high"
        elif score > q2_local:
            return "medium"
        return "low"

    quantile_bins = pd.Series(
        [_assign_quantile_bin(score) for score in sorted_scores],
        index=sorted_scores.index,
    )

    # Compute relative gaps
    relative_gaps = []
    for i in range(1, len(sorted_scores)):
        prev_score = sorted_scores.iloc[i - 1]
        curr_score = sorted_scores.iloc[i]
        relative_gap = (prev_score - curr_score) / prev_score if prev_score > 0 else 0
        relative_gaps.append((i - 1, relative_gap))

    # Find two largest relative gaps
    relative_gaps.sort(key=lambda x: x[1], reverse=True)

    if len(relative_gaps) >= 2:
        gap_positions = sorted([x[0] for x in relative_gaps[:2]])
        i_star, j_star = gap_positions[0], gap_positions[1]

        dropoff_bins = pd.Series("low", index=sorted_scores.index, dtype="string")
        dropoff_bins.iloc[: i_star + 1] = "high"
        dropoff_bins.iloc[i_star + 1 : j_star + 1] = "medium"
    else:
        dropoff_bins = quantile_bins

    # Take minimum of quantile and dropoff bins
    bin_order = {"high": 3, "medium": 2, "low": 1}
    final_local_bins = pd.Series(index=sorted_scores.index, dtype="string")

    for idx in sorted_scores.index:
        quantile_val = bin_order[quantile_bins[idx]]
        dropoff_val = bin_order[dropoff_bins[idx]]
        min_val = min(quantile_val, dropoff_val)
        final_local_bins[idx] = {3: "high", 2: "medium", 1: "low"}[min_val]

    return final_local_bins


def prune_global_matches(
    matches: pd.DataFrame,
    global_embedding_threshold: float,
) -> pd.DataFrame:
    """
    Prune low-scoring matches based on a global quantile threshold.

    Args:
        matches: DataFrame with raw similarity matches
        global_embedding_threshold: Quantile threshold below which matches are dropped

    Returns:
        DataFrame with pruned matches
    """
    logger.info(
        "Pruning global matches with threshold: %0.2f\n" "Initial matches: %d",
        global_embedding_threshold,
        len(matches),
    )

    # Compute threshold
    score_threshold = matches["similarity_score"].quantile(global_embedding_threshold)

    # Filter matches
    pruned_matches = matches[matches["similarity_score"] >= score_threshold]

    logger.info(
        "Score threshold: %0.3f\n" "Remaining matches: %d (%0.1f%%)",
        score_threshold,
        len(pruned_matches),
        100 * len(pruned_matches) / len(matches),
    )

    return pruned_matches


def _assign_local_bins(
    df: pd.DataFrame,
    local_q2_threshold: float,
    local_q3_threshold: float,
    n_jobs: int = 8,
    **unused_params
) -> pd.DataFrame:
    """Assign confidence bins using only local thresholds."""
    # Process groups in parallel
    project_groups = [group for _, group in df.groupby("project_id")]

    _assign_local_bins_partial = partial(
        _process_project_group,
        local_q2_threshold=local_q2_threshold,
        local_q3_threshold=local_q3_threshold,
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_assign_local_bins_partial)(group) for group in project_groups
    )

    df["local_bin"] = pd.concat(results)

    # Final bin (minimum of global and local)
    bin_order = {"high": 3, "medium": 2, "low": 1}
    df["final_bin"] = df.apply(
        lambda row: {3: "high", 2: "medium", 1: "low"}[
            min(bin_order[row["global_bin"]], bin_order[row["local_bin"]])
        ],
        axis=1,
    )

    return df


def validate_with_zeroshot(
    aggregated_scores: pd.DataFrame,
    project_texts: pd.DataFrame,
    batch_size: int = 32,
    model_name: str = "tasksource/ModernBERT-large-nli",
) -> pd.DataFrame:
    """
    Validate taxonomy assignments using zero-shot classification.

    Args:
        aggregated_scores: DataFrame with project-label pairs and confidence bins
        project_texts: DataFrame with project_id and text columns
        batch_size: Number of projects to process at once
        model_name: HuggingFace model to use for zero-shot classification
    """
    logger.info("Initializing zero-shot classifier with model: %s", model_name)

    # Initialize with memory-efficient settings
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        multi_label=True,
        batch_size=batch_size,
        truncation=True,
        torch_dtype=torch.float16,
    )

    # Merge project texts efficiently
    scores_with_text = pd.merge(
        aggregated_scores[["project_id", "taxonomy_label"]],
        project_texts[["project_id", "text"]],
        on="project_id",
        how="inner",
    )

    # Group and prepare batches
    project_groups = (
        scores_with_text.groupby("project_id")
        .agg({"text": "first", "taxonomy_label": list})
        .reset_index()
        .rename(columns={"taxonomy_label": "candidate_labels"})
    )

    logger.info("Running zero-shot classification for %d projects", len(project_groups))
    results = []

    # Process in smaller chunks to manage memory
    chunk_size = 100  # Process 100 projects at a time
    for chunk_start in range(0, len(project_groups), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(project_groups))
        chunk = project_groups.iloc[chunk_start:chunk_end]
        
        logger.info(
            "Processing projects %d to %d of %d",
            chunk_start, chunk_end, len(project_groups)
        )

        for _, row in tqdm(chunk.iterrows(), total=len(chunk)):
            try:
                # Get predictions
                prediction = classifier(
                    row["text"],
                    candidate_labels=row["candidate_labels"],
                    hypothesis_template="This research project is about {}.",
                )

                # Store results
                for label, score in zip(prediction["labels"], prediction["scores"]):
                    results.append({
                        "project_id": row["project_id"],
                        "taxonomy_label": label,
                        "zeroshot_score": score,
                    })

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Failed to classify project %s: %s",
                    row["project_id"],
                    str(e),
                )
                continue

        # Clear memory after each chunk
        torch.cuda.empty_cache()
        
        # Log progress
        if len(results) > 0:
            logger.info(
                "Processed %d projects. Current memory usage: %.1f GB",
                len(results),
                torch.cuda.max_memory_allocated() / 1e9
            )

    # Process results
    zeroshot_df = pd.DataFrame(results)
    final_scores = pd.merge(
        aggregated_scores,
        zeroshot_df,
        on=["project_id", "taxonomy_label"],
        how="left",
    )

    final_scores["zeroshot_bin"] = final_scores["zeroshot_score"].apply(
        lambda x: (
            "very high" if x >= 0.9
            else "high" if x >= 0.7
            else "medium" if x >= 0.5
            else "low" if x >= 0.25
            else "very low"
        )
    )

    return final_scores
