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
    preprocess_cwts_topics,
    preprocess_goscience_taxonomy,
    preprocess_oa_concepts,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    taxonomy_keywords_pipeline = pipeline(
        [
            node(
                func=preprocess_cwts_topics,
                inputs="taxonomy.cwts.raw",
                outputs=["taxonomy.cwts.full.db", "taxonomy.cwts.bottom.db"],
                name="preprocess_cwts_topics",
            ),
            node(
                func=preprocess_goscience_taxonomy,
                inputs="taxonomy.goscience.raw",
                outputs=["taxonomy.goscience.full.db", "taxonomy.goscience.bottom.db"],
                name="preprocess_goscience_taxonomy",
            ),
            node(
                func=preprocess_oa_concepts,
                inputs="taxonomy.oa_concepts.raw",
                outputs=[
                    "taxonomy.oa_concepts.full.db",
                    "taxonomy.oa_concepts.bottom.db",
                ],
                name="preprocess_oa_concepts",
            ),
        ]
    )

    return taxonomy_keywords_pipeline
