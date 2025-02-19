"""
Utility functions for the project similarity refinement pipeline.

This module provides helper functions for:
- Processing project groups for confidence binning
- Assigning confidence bins based on score distributions
- Computing relative gaps between scores
"""

from functools import partial
import pandas as pd
from joblib import Parallel, delayed


def assign_local_bins(
    df: pd.DataFrame,
    local_q2_threshold: float,
    local_q3_threshold: float,
    n_jobs: int = 8,
    **unused_params
) -> pd.DataFrame:
    """
    Assign confidence bins using project-level thresholds.

    For each project, this function:
    1. Computes local quantile thresholds based on score distribution
    2. Assigns initial bins based on quantiles
    3. Refines bins based on relative score gaps
    4. Takes minimum of global and local bins for confidence assignment

    Args:
        df: DataFrame with project-label scores containing:
            - project_id: ID of the research project
            - relevance_score: Score to bin
            - global_bin: Already computed global confidence bin
        local_q2_threshold: Quantile threshold for medium confidence (e.g., 0.5)
        local_q3_threshold: Quantile threshold for high confidence (e.g., 0.75)
        n_jobs: Number of parallel jobs for processing
        **unused_params: Additional parameters (ignored)

    Returns:
        DataFrame with additional columns:
            - local_bin: Confidence bin based on project-level thresholds
            - confidence_bin: Combined confidence bin (minimum of global and local)
    """
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
    bin_order = {"high": 3, "medium": 2, "low": 1}
    df["confidence_bin"] = df.apply(
        lambda row: {3: "high", 2: "medium", 1: "low"}[
            min(bin_order[row["global_bin"]], bin_order[row["local_bin"]])
        ],
        axis=1,
    )

    return df


def _process_project_group(
    group: pd.DataFrame,
    local_q2_threshold: float,
    local_q3_threshold: float,
) -> pd.Series:
    """
    Process a single project group for local confidence binning.

    This function assigns confidence bins based on both:
    1. Local quantile thresholds
    2. Relative gaps between consecutive scores

    Args:
        group: DataFrame slice containing scores for one project
        local_q2_threshold: Quantile threshold for medium confidence
        local_q3_threshold: Quantile threshold for high confidence

    Returns:
        Series with confidence bins ('high', 'medium', 'low') for each label,
        using the minimum of quantile-based and gap-based assignments
    """
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
