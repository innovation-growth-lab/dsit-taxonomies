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
    kedro run --pipeline data_collection_gtr --tags projects
    ```

Note:
    In regards to the use of namespaces, note that these are appended as
    prefixes to the outputs of the nodes in the pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import aggregate_keyword_annotators, generate_keyword_embeddings


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    aggregate_keywords_pipeline = pipeline(
        [
            node(
                func=aggregate_keyword_annotators,
                inputs=[
                    "dbp.gtr_data.annotated",
                    "rake.gtr_data.annotated",
                    "yake.gtr_data.annotated",
                    "keybert.gtr_data.annotated",
                ],
                outputs="keywords.gtr_data.preprocessed",
                name="aggregate_keyword_annotators",
            ),
            node(
                func=generate_keyword_embeddings,
                inputs="keywords.gtr_data.preprocessed",
                outputs="keywords.gtr_data.embeddings",
                name="generate_keyword_embeddings",
            ),
        ]
    )

    return aggregate_keywords_pipeline
