"""
Utility functions for the keyword similarity tuning pipeline.

This module provides helper functions for:
- Computing validation metrics
- Analysing agreement between predictions
- Processing expert feedback
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_per_project_metrics(
    expert_df: pd.DataFrame, assessment_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute true positives, false positives, and false negatives for each project.

    Args:
        expert_df: DataFrame with expert labels
        assessment_df: DataFrame with assessment predictions

    Returns:
        DataFrame with per-project metrics
    """
    # Remove hallucinated labels and convert to lowercase
    expert_df["likelihood"] = expert_df["likelihood"].str.lower()

    # Merge expert and assessment predictions
    data = pd.merge(
        assessment_df,
        expert_df,
        on=["project_id", "taxonomy_label_id"],
        how="outer",
    )

    data["taxonomy_label"] = data["taxonomy_label_x"].fillna(data["taxonomy_label_y"])
    data = data.drop(columns=["taxonomy_label_x", "taxonomy_label_y"])

    project_metrics = []

    for project_id in data["project_id"].unique():
        project_data = data[data["project_id"] == project_id]

        # Define conditions for true/false positives/negatives
        algo_high = project_data["confidence_bin"] == "high"
        expert_agreement = (project_data["positive"] is True) | (
            project_data["likelihood"] == "high"
        )
        expert_disagreement = (project_data["positive"] is False) | (
            project_data["likelihood"] != "high"
        )

        # Calculate metrics
        true_positives = project_data[algo_high & expert_agreement][
            "taxonomy_label"
        ].tolist()
        false_positives = project_data[algo_high & expert_disagreement][
            "taxonomy_label"
        ].tolist()
        false_negatives = project_data[
            (~algo_high | algo_high.isna()) & expert_agreement
        ]["taxonomy_label"].tolist()

        project_metrics.append(
            {
                "project_id": project_id,
                "num_true_positives": len(true_positives),
                "true_positives": true_positives,
                "num_false_positives": len(false_positives),
                "false_positives": false_positives,
                "num_false_negatives": len(false_negatives),
                "false_negatives": false_negatives,
            }
        )

    return pd.DataFrame(project_metrics)


def validate_predictions(
    expert_df: pd.DataFrame,
    assessment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate predictions by comparing expert high likelihood labels with assessment predictions.
    Computes metrics for both strict (high only) and relaxed (high+medium) assessment confidence.

    Args:
        expert_df: DataFrame with expert labels
        assessment_df: DataFrame with assessment predictions

    Returns:
        DataFrame with validation metrics for each threshold
    """
    logger.info("Validating predictions against expert high likelihood labels")

    # Remove hallucinated labels and convert to lowercase
    expert_df["likelihood"] = expert_df["likelihood"].str.lower()

    # Merge expert and assessment predictions
    data = pd.merge(
        assessment_df,
        expert_df,
        on=["project_id", "taxonomy_label_id"],
        how="outer",
    )

    # Clean up labels
    data["taxonomy_label"] = data["taxonomy_label_x"].fillna(data["taxonomy_label_y"])
    data = data.drop(columns=["taxonomy_label_x", "taxonomy_label_y"])

    metrics = []

    # Calculate metrics for both strict and relaxed thresholds
    for threshold in ["strict", "relaxed"]:
        # Define assessment condition based on threshold
        algo_condition = (
            (data["confidence_bin"] == "high")
            if threshold == "strict"
            else (data["confidence_bin"].isin(["high", "medium"]))
        )

        # True positives: Algorithm predicts high AND expert agrees
        expert_agreement = (data["positive"] is True) | (
            data["likelihood"].isin(["high", "medium"])
        )
        true_positives = sum(algo_condition & expert_agreement)

        # False positives: Algorithm predicts high BUT expert disagrees
        expert_disagreement = (data["positive"] is False) | (
            ~data["likelihood"].isin(["high", "medium"])
        )
        false_positives = sum(algo_condition & expert_disagreement)

        # False negatives: Algorithm doesn't predict high (or is missing) BUT
        # expert thinks it should
        false_negatives = sum(
            (~algo_condition | algo_condition.isna()) & expert_agreement
        )

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics.append(
            {
                "threshold": threshold,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        # Format threshold description for logging
        threshold_desc = "high only" if threshold == "strict" else "high+medium"

        logger.info(
            "%s Threshold Metrics (algo: %s):\n"
            "Precision: %0.3f\n"
            "Recall: %0.3f\n"
            "F1 Score: %0.3f\n"
            "True Positives: %d\n"
            "False Positives: %d\n"
            "False Negatives: %d",
            threshold.title(),
            threshold_desc,
            precision,
            recall,
            f1,
            true_positives,
            false_positives,
            false_negatives,
        )

    return pd.DataFrame(metrics)


def compute_likelihood_agreement(
    row: pd.Series, bin_column: str = "zeroshot_bin"
) -> str:
    """
    Compute agreement level between model predictions and expert likelihood ratings.

    Args:
        row: Series containing:
            - likelihood: Expert assigned likelihood
            - bin_column: Column name for model predictions (zeroshot_bin, confidence_bin, or combined_bin)

    Returns:
        Agreement level ('strong', 'weak', or 'disagree')
    """
    if pd.isna(row["likelihood"]) or pd.isna(row[bin_column]):
        return None

    # Map bins to numeric values
    model_strength = {
        "very high": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
        "very low": 0,
    }
    expert_strength = {"high": 3, "medium": 2, "low": 1}

    # Get model prediction value
    m_val = model_strength[row[bin_column]]
    # Get expert value (convert to lowercase to handle any inconsistencies)
    e_val = expert_strength[row["likelihood"].lower()]

    # Strong agreement: Difference ≤ 1
    # Weak agreement: Difference = 2
    # Disagreement: Difference > 2
    diff = abs(m_val - e_val)
    if diff <= 1:
        return "strong"
    elif diff == 2:
        return "weak"
    return "disagree"


def compute_validation_metrics(
    predictions: pd.DataFrame,
    expert_df: pd.DataFrame,
    assessment_df: pd.DataFrame,
    bin_column: str,
    positive_bins: list,
) -> dict:
    """
    Compute validation metrics for a set of predictions against expert labels.

    Args:
        predictions: DataFrame with predicted labels and confidence bins
        expert_df: DataFrame with expert-suggested labels and likelihood
        assessment_df: DataFrame with binary assessment of proposed labels
        bin_column: Name of the confidence bin column
        positive_bins: List of bin values to consider as positive predictions

    Returns:
        Dictionary with computed metrics
    """
    # Get positive predictions
    pred_positives = predictions[
        predictions[bin_column].isin(positive_bins)
    ][["project_id", "taxonomy_label_id"]]
    
    # Get true positives from assessment_df
    true_pos = len(
        pred_positives.merge(
            assessment_df[assessment_df["positive"]],
            on=["project_id", "taxonomy_label_id"],
            how="inner"
        )
    )
    
    # Get false positives from assessment_df
    false_pos = len(
        pred_positives.merge(
            assessment_df[~assessment_df["positive"]],
            on=["project_id", "taxonomy_label_id"],
            how="inner"
        )
    )
    
    # Get expert-suggested labels (excluding low confidence)
    expert_labels = expert_df[
        expert_df["likelihood"].isin(["high"])
    ][["project_id", "taxonomy_label_id"]]
    
    # False negatives are expert-suggested labels we missed
    false_neg = len(
        expert_labels[
            ~expert_labels.apply(
                lambda x: (
                    (x["project_id"], x["taxonomy_label_id"]) in 
                    zip(pred_positives["project_id"], pred_positives["taxonomy_label_id"])
                ),
                axis=1
            )
        ]
    )
    
    # Compute metrics
    total_pred = true_pos + false_pos
    total_true = true_pos + false_neg
    
    precision = true_pos / total_pred if total_pred > 0 else 0.0
    recall = true_pos / total_true if total_true > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "false_negatives": false_neg,
    }
