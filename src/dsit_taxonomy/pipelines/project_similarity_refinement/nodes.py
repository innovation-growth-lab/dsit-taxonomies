"""
This module contains nodes for refining and validating taxonomy assignments.

The module provides functionality for:
- Aggregating granular scores to project-label pairs
- Assigning confidence bins based on score distributions 
- Validating assignments using zero-shot classification

The nodes in this module are used in the keyword_similarity_refinement pipeline
to process the raw similarity scores and produce final taxonomy assignments.
"""

import logging
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import torch
from .utils import assign_local_bins

logger = logging.getLogger(__name__)


def aggregate_scores_to_labels(
    granular_scores: pd.DataFrame, normalise_by_matches: bool = False, **binning_params
) -> pd.DataFrame:
    """
    Aggregate sentence-level scores to project-label pairs.

    Args:
        granular_scores: DataFrame with sentence-level scores and metadata
        normalise_by_matches: If True, weight scores by ratio of matching sentences
        **binning_params: Parameters for confidence binning including:
            - global_q2_threshold: Quantile threshold for medium confidence
            - global_q3_threshold: Quantile threshold for high confidence
            - local_q2_threshold: Project-level threshold for medium confidence
            - local_q3_threshold: Project-level threshold for high confidence

    Returns:
        DataFrame with aggregated scores and confidence bins containing:
            - project_id: ID of the research project
            - taxonomy_label_id: ID of the taxonomy label
            - relevance_score: Combined relevance score
            - global_bin: Confidence bin based on global thresholds
            - local_bin: Confidence bin based on project-level thresholds
            - confidence_bin: Combined confidence bin (minimum of global and local)
            - num_matching_sentences: Count of sentences matching this label
            - num_sentences: Total sentences in the project
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
    return assign_local_bins(aggregated, **binning_params)


def prune_global_matches(
    matches: pd.DataFrame,
    global_embedding_threshold: float,
) -> pd.DataFrame:
    """
    Prune low-scoring matches based on a global quantile threshold.

    Args:
        matches: DataFrame with raw similarity matches containing:
            - similarity_score: Score to threshold on
        global_embedding_threshold: Quantile threshold below which matches are dropped

    Returns:
        DataFrame with pruned matches, containing only matches above the
        computed score threshold
    """
    logger.info(
        "Pruning global matches with threshold: %0.2f\nInitial matches: %d",
        global_embedding_threshold,
        len(matches),
    )

    # Compute threshold
    score_threshold = matches["similarity_score"].quantile(global_embedding_threshold)

    # Filter matches
    pruned_matches = matches[matches["similarity_score"] >= score_threshold]

    logger.info(
        "Score threshold: %0.3f\nRemaining matches: %d (%0.1f%%)",
        score_threshold,
        len(pruned_matches),
        100 * len(pruned_matches) / len(matches),
    )

    return pruned_matches


def enhance_with_zeroshot(
    aggregated_scores: pd.DataFrame,
    project_texts: pd.DataFrame,
    batch_size: int = 32,
    model_name: str = "tasksource/ModernBERT-large-nli",
) -> pd.DataFrame:
    """
    Enhance taxonomy assignments using zero-shot classification.

    This node validates the similarity-based matches using a zero-shot classifier
    to assess whether each project text actually discusses the matched taxonomy labels.

    Args:
        aggregated_scores: DataFrame with project-label pairs and confidence bins
        project_texts: DataFrame with project_id and text columns
        batch_size: Number of projects to process in each batch
        model_name: HuggingFace model to use for zero-shot classification

    Returns:
        DataFrame with original scores plus zero-shot confidence scores:
            - project_id: ID of the research project
            - taxonomy_label: Label being validated
            - zeroshot_score: Classification confidence score
            - zeroshot_bin: Confidence bin based on score thresholds
    """
    logger.info("Initialising zero-shot classifier with model: %s", model_name)

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
            chunk_start,
            chunk_end,
            len(project_groups),
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
                    results.append(
                        {
                            "project_id": row["project_id"],
                            "taxonomy_label": label,
                            "zeroshot_score": score,
                        }
                    )

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
                torch.cuda.max_memory_allocated() / 1e9,
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
            "very high"
            if x >= 0.9
            else (
                "high"
                if x >= 0.7
                else "medium" if x >= 0.5 else "low" if x >= 0.25 else "very low"
            )
        )
    )

    return final_scores
