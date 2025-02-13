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
    compute_similarities,
    aggregate_sentence_matches,
    combine_sentence_and_keyword_scores,
    add_metadata,
    aggregate_scores_to_labels,
    prune_global_matches,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=C0116,W0613
    doc_preprocess_pipeline = pipeline(
        [
            node(
                func=document_preprocessing,
                inputs="gtr.projects.documents",
                outputs=["projects.gtr_data.db", "sentences.gtr_data.db"],
                name="document_preprocessing",
            )
        ]
    )

    def compute_raw_matches_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                node(
                    func=compute_similarities,
                    inputs={
                        "taxonomy": f"taxonomy.{taxonomy_name}.full.db",
                        "documents": "projects.gtr_data.db",
                        "batch_size": "params:similarity_matching.projects.batch_size",
                        "top_n": "params:similarity_matching.projects.top_n",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_matches.raw",
                    name=f"compute_global_matches_{taxonomy_name}",
                ),
                # Compute sentence matches
                node(
                    func=compute_similarities,
                    inputs={
                        "taxonomy": f"taxonomy.{taxonomy_name}.full.db",
                        "documents": "sentences.gtr_data.db",
                        "batch_size": "params:similarity_matching.sentences.batch_size",
                        "top_n": "params:similarity_matching.sentences.top_n",
                    },
                    outputs=f"sentences.gtr_data.{taxonomy_name}_matches.raw",
                    name=f"compute_sentence_matches_{taxonomy_name}",
                ),
                # Compute keyword matches
                node(
                    func=compute_similarities,
                    inputs={
                        "taxonomy": f"taxonomy.{taxonomy_name}.full.db",
                        "documents": "keywords.gtr_data.db",
                        "batch_size": "params:similarity_matching.keywords.batch_size",
                        "top_n": "params:similarity_matching.keywords.top_n",
                    },
                    outputs=f"keywords.gtr_data.{taxonomy_name}_matches.intermediate",
                    name=f"compute_keyword_matches_{taxonomy_name}",
                ),
            ],
            tags=[
                f"raw_matches_{taxonomy_name}",
                f"similarity_matching_{taxonomy_name}",
                "raw_matches",
            ],
        )

    def scoring_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                # Prune global matches
                node(
                    func=prune_global_matches,
                    inputs={
                        "matches": f"projects.gtr_data.{taxonomy_name}_matches.raw",
                        "global_embedding_threshold": "params:similarity_matching.global_embedding_threshold",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_matches.pruned",
                    name=f"prune_global_matches_{taxonomy_name}",
                ),
                # Aggregate sentence matches
                node(
                    func=aggregate_sentence_matches,
                    inputs={
                        "sentences": "sentences.gtr_data.db",
                        "sentence_matches": f"sentences.gtr_data.{taxonomy_name}_matches.raw",
                        "min_score_quantile": "params:similarity_matching.sentence_matches.min_score_quantile",
                    },
                    outputs=f"sentences.gtr_data.{taxonomy_name}_matches.intermediate",
                    name=f"aggregate_sentence_matches_{taxonomy_name}",
                    tags=[
                        f"sentence_matches_{taxonomy_name}",
                        "combine_scores_and_add_metadata",
                    ],
                ),
                # Combine sentence and keyword scores
                node(
                    func=combine_sentence_and_keyword_scores,
                    inputs={
                        "sentence_scores": f"sentences.gtr_data.{taxonomy_name}_matches.intermediate",
                        "keyword_scores": f"keywords.gtr_data.{taxonomy_name}_matches.intermediate",
                        "keyword_data": "keywords.gtr_data.db",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_scores.intermediate",
                    name=f"combine_scores_{taxonomy_name}",
                    tags=[
                        f"scores_{taxonomy_name}",
                        "combine_scores_and_add_metadata",
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
                    tags=[
                        f"scores_{taxonomy_name}",
                        "combine_scores_and_add_metadata",
                    ],
                ),
                # Final aggregation
                node(
                    func=aggregate_scores_to_labels,
                    inputs={
                        "scores": f"projects.gtr_data.{taxonomy_name}_scores.detailed",
                        "sentence_weight": "params:similarity_matching.score_weights.sentence_weight",
                        "similarity_quantile_threshold": "params:similarity_matching.similarity_quantile_threshold",
                        "global_q2_threshold": "params:similarity_matching.binning.global_q2_threshold",
                        "global_q3_threshold": "params:similarity_matching.binning.global_q3_threshold",
                        "local_q2_threshold": "params:similarity_matching.binning.local_q2_threshold",
                        "local_q3_threshold": "params:similarity_matching.binning.local_q3_threshold",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_scores.aggregated",
                    name=f"aggregate_final_scores_{taxonomy_name}",
                    tags=[
                        f"scores_{taxonomy_name}",
                        "aggregate_scores",
                    ],
                ),
            ],
            tags=[
                f"similarity_matching_{taxonomy_name}",
            ],
        )

    return (
        doc_preprocess_pipeline
        + sum(
            compute_raw_matches_pipeline(tax)
            for tax in ["cwts", "goscience", "oa_concepts"]
        )
        + sum(scoring_pipeline(tax) for tax in ["cwts", "goscience", "oa_concepts"])
    )
