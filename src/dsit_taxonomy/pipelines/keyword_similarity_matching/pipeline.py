"""
This pipeline fetches data from the GtR API and preprocesses it into a format
that can be used by the rest of the project.

Pipelines:
    - data_collection_gtr:
        Fetches and preprocesses data from the GtR API.

Dependencies:
    - Kedro
    - pandas
    - requests
    - logging

Usage:
    Run the pipeline to fetch and preprocess data from the GtR API.

Command Line Example:
    ```
    kedro run --pipeline data_collection_gtr
    ```
    Alternatively, you can run this pipeline for a single endpoint:
    ```
    kedro run --pipeline ___ --tags projects
    ```

Note:
    In regards to the use of namespaces, note that these are appended as
    prefixes to the outputs of the nodes in the pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    document_preprocessing,
    compute_similarities_and_entropy,
    aggregate_sentence_matches,
    combine_sentence_and_keyword_scores,
    add_metadata,
    aggregate_scores_to_labels,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=C0116,W0613
    doc_preprocess_pipeline = pipeline(
        [
            node(
                func=document_preprocessing,
                inputs="gtr.projects.documents",
                outputs="sentences.gtr_data.db",
            )
        ]
    )

    def compute_raw_matches_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                # Compute sentence matches
                node(
                    func=compute_similarities_and_entropy,
                    inputs={
                        "taxonomy": f"taxonomy.{taxonomy_name}.bottom.db",
                        "documents": "sentences.gtr_data.db",
                        "batch_size": "params:similarity_matching.batch_size",
                        "top_n": "params:similarity_matching.top_n",
                        "number_returns": "params:similarity_matching.number_returns",
                    },
                    outputs=f"sentences.gtr_data.{taxonomy_name}_matches.raw",
                    name=f"compute_sentence_matches_{taxonomy_name}",
                    tags=[
                        f"raw_matches_{taxonomy_name}",
                        f"similarity_matching_{taxonomy_name}",
                    ],
                ),
                # Compute keyword matches
                node(
                    func=compute_similarities_and_entropy,
                    inputs={
                        "taxonomy": f"taxonomy.{taxonomy_name}.bottom.db",
                        "documents": "keywords.gtr_data.db",
                        "batch_size": "params:similarity_matching.batch_size",
                        "top_n": "params:similarity_matching.top_n",
                        "number_returns": "params:similarity_matching.number_returns",
                    },
                    outputs=f"keywords.gtr_data.{taxonomy_name}_matches.intermediate",
                    name=f"compute_keyword_matches_{taxonomy_name}",
                    tags=[
                        f"raw_matches_{taxonomy_name}",
                        f"similarity_matching_{taxonomy_name}",
                    ],
                ),
            ]
        )

    def scoring_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                # Aggregate sentence matches
                node(
                    func=aggregate_sentence_matches,
                    inputs={
                        "sentences": "sentences.gtr_data.db",
                        "sentence_matches": f"sentences.gtr_data.{taxonomy_name}_matches.raw",
                        "top_k_per_sentence": "params:similarity_matching.sentence_matches.top_k_per_sentence",
                        "min_score_quantile": "params:similarity_matching.sentence_matches.min_score_quantile",
                        "min_similarity_score": "params:similarity_matching.sentence_matches.min_similarity_score",
                    },
                    outputs=f"sentences.gtr_data.{taxonomy_name}_matches.intermediate",
                    name=f"aggregate_sentence_matches_{taxonomy_name}",
                    tags=[
                        f"sentence_matches_{taxonomy_name}",
                        f"similarity_matching_{taxonomy_name}",
                    ],
                ),
                # Combine sentence and keyword scores
                node(
                    func=combine_sentence_and_keyword_scores,
                    inputs={
                        "sentence_scores": f"sentences.gtr_data.{taxonomy_name}_matches.intermediate",
                        "keyword_scores": f"keywords.gtr_data.{taxonomy_name}_matches.intermediate",
                        "keyword_data": "keywords.gtr_data.db",
                        "sentence_weight": "params:similarity_matching.score_weights.sentence_weight",
                        "keyword_weight": "params:similarity_matching.score_weights.keyword_weight",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_scores.intermediate",
                    name=f"combine_scores_{taxonomy_name}",
                    tags=[
                        f"scores_{taxonomy_name}",
                        f"similarity_matching_{taxonomy_name}",
                    ],
                ),
                # Add metadata
                node(
                    func=add_metadata,
                    inputs={
                        "combined_scores": f"projects.gtr_data.{taxonomy_name}_scores.intermediate",
                        "keyword_data": "keywords.gtr_data.db",
                        "taxonomy": f"taxonomy.{taxonomy_name}.full.db",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_scores.detailed",
                    name=f"add_metadata_{taxonomy_name}",
                ),
                # Final aggregation
                node(
                    func=aggregate_scores_to_labels,
                    inputs=f"projects.gtr_data.{taxonomy_name}_scores.detailed",
                    outputs=f"projects.gtr_data.{taxonomy_name}_scores.aggregated",
                    name=f"aggregate_final_scores_{taxonomy_name}",
                    tags=[
                        f"scores_{taxonomy_name}",
                        f"similarity_matching_{taxonomy_name}",
                    ],
                ),
            ]
        )

    return (
        doc_preprocess_pipeline
        + sum(
            compute_raw_matches_pipeline(tax)
            for tax in ["cwts", "goscience", "oa_concepts"]
        )
        + sum(scoring_pipeline(tax) for tax in ["cwts", "goscience", "oa_concepts"])
    )
