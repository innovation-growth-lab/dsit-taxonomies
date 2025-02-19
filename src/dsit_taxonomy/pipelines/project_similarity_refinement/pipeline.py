"""
This pipeline refines and validates taxonomy assignments using score aggregation
and zero-shot classification.

The pipeline performs two main steps:
1. Score Aggregation
   - Combines sentence-level scores into project-level assignments
   - Assigns initial confidence bins based on score distributions
   - Uses both global and project-level thresholds

2. Zero-Shot Validation
   - Validates initial confidence assignments
   - Provides additional validation scores
   - Helps filter out spurious matches
   - Produces final confidence assignments

Dependencies:
    - pandas
    - transformers
    - torch
    - joblib

Example:
    Run the refinement pipeline:
    ```
    kedro run --pipeline project_similarity_refinement
    ```
    Or run specific tags:
    ```
    kedro run --pipeline project_similarity_refinement --tags aggregate
    ```
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    aggregate_scores_to_labels,
    enhance_with_zeroshot,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=C0116,W0613
    def refinement_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                # Aggregate scores to labels
                node(
                    func=aggregate_scores_to_labels,
                    inputs={
                        "granular_scores": f"projects.gtr_data.{taxonomy_name}_scores.granular",
                        "normalise_by_matches": "params:similarity_refinement.normalise_by_matches",
                        "global_q2_threshold": "params:similarity_refinement.binning.global_q2",
                        "global_q3_threshold": "params:similarity_refinement.binning.global_q3",
                        "local_q2_threshold": "params:similarity_refinement.binning.local_q2",
                        "local_q3_threshold": "params:similarity_refinement.binning.local_q3",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_scores.aggregated",
                    name=f"aggregate_scores_to_labels_{taxonomy_name}",
                    tags=["refinement", taxonomy_name],
                ),
                node(
                    func=enhance_with_zeroshot,
                    inputs={
                        "aggregated_scores": f"projects.gtr_data.{taxonomy_name}_scores.aggregated",
                        "project_texts": "projects.gtr_data.db",
                        "batch_size": "params:similarity_refinement.zeroshot.batch_size",
                        "model_name": "params:similarity_refinement.zeroshot.model_name",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_scores.zeroshot",
                    name=f"enhance_zeroshot_{taxonomy_name}",
                    tags=["refinement", taxonomy_name],
                ),
            ],
            tags=[
                "refinement",
            ],
        )

    return sum(refinement_pipeline(tax) for tax in ["cwts", "goscience", "oa_concepts"])
