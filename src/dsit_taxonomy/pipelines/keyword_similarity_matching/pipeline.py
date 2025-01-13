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
    compute_similarities_and_entropy
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
                    "taxonomy": "taxonomy.cwts.bottom.db", 
                    "keywords": "keywords.gtr_data.db",
                    "batch_size": "params:batch_size"
                },
                outputs="keywords.gtr_data.cwts_matches.intermediate",
                name="compute_matches_cwts",
            ),
        ]
    )

    return taxonomy_keywords_pipeline
