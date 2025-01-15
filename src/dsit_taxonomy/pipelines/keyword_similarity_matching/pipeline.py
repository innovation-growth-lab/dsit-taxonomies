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
    compute_similarities_and_entropy,
    document_preprocessing,
    compute_document_scores,
    aggregate_scores_to_labels,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    taxonomy_keywords_pipeline = pipeline(
        [
            node(
                func=compute_similarities_and_entropy,
                inputs={
                    "taxonomy": f"taxonomy.{tax}.bottom.db",
                    "documents": "keywords.gtr_data.db",
                    "batch_size": "params:batch_size",
                    "top_n": "params:top_n",
                    "number_returns": "params:number_returns",
                },
                outputs=f"keywords.gtr_data.{tax}_matches.intermediate",
                name=f"compute_keyword_matches_{tax}",
            )
            for tax in ["cwts", "oa_concepts", "goscience"]
        ]
    )

    document_weights_pipeline = pipeline(
        [
            node(
                func=document_preprocessing,
                inputs="gtr.projects.documents",
                outputs="sentences.gtr_data.db",
                name="document_preprocessing",
            )
        ]
        + [
            node(
                func=compute_similarities_and_entropy,
                inputs={
                    "taxonomy": f"taxonomy.{tax}.bottom.db",
                    "documents": "sentences.gtr_data.db",
                    "batch_size": "params:batch_size",
                    "top_n": "params:top_n",
                    "number_returns": "params:number_returns",
                },
                outputs=f"sentences.gtr_data.{tax}_matches.intermediate",
                name=f"compute_document_matches_{tax}",
            )
            for tax in ["cwts", "oa_concepts", "goscience"]
        ]
        + [
            node(
                func=compute_document_scores,
                inputs={
                    "documents": "sentences.gtr_data.db",
                    "document_matches": f"sentences.gtr_data.{tax}_matches.intermediate",
                },
                outputs=f"projects.gtr_data.{tax}_matches.intermediate",
                name=f"compute_document_scores_{tax}",
            )
            for tax in ["cwts", "oa_concepts", "goscience"]
        ]
        + [
            node(
                func=aggregate_scores_to_labels,
                inputs={
                    "document_scores": f"projects.gtr_data.{tax}_matches.intermediate",
                    "keyword_scores": f"keywords.gtr_data.{tax}_matches.intermediate",
                    "keyword_data": "keywords.gtr_data.db",
                },
                outputs=f"projects.gtr_data.{tax}_scores",
                name=f"aggregate_scores_to_labels_{tax}",
            )
            for tax in ["cwts", "oa_concepts", "goscience"]
        ],
        tags=["compute_document_weights"],
    )

    return taxonomy_keywords_pipeline + document_weights_pipeline
