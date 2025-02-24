"""
This pipeline processes and combines keywords extracted from research projects
using multiple keyword extraction methods.

The pipeline performs two main steps:
1. Keyword Aggregation
   - Combines keywords from multiple extractors (DBP, RAKE, YAKE, KeyBERT)
   - Counts appearances across different extractors
   - Filters keywords based on extractor agreement
   - Maps keywords to their source projects

2. Embedding Generation
   - Generates semantic embeddings for filtered keywords
   - Uses sentence transformers for embedding computation
   - Prepares keywords for similarity matching

Dependencies:
    - pandas
    - numpy
    - sentence-transformers
    - uuid

Example:
    Run the complete keyword processing pipeline:
    ```
    kedro run --pipeline keyword_processing_gtr
    ```
    Or run specific nodes:
    ```
    kedro run --pipeline keyword_processing_gtr --nodes aggregate_keyword_annotators
    ```
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    dbp_keywords,
    rake_keywords,
    yake_keywords,
    keybert_keywords,
    concatenate_partitions,
    aggregate_keyword_annotators,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """
    Creates a pipeline for processing and embedding keywords.

    The pipeline aggregates keywords from multiple extractors and generates
    embeddings for downstream similarity matching.

    Returns:
        Pipeline: A pipeline containing keyword aggregation and embedding nodes
    """
    annotation_pipeline = pipeline(
        [
            node(
                func=dbp_keywords,
                inputs={
                    "dataframe": "gtr.data_collection.projects.intermediate",
                    "processed_projects": "dbp.gtr_data.annotated.oracle",
                },
                outputs="dbp.gtr_data.annotated",
                name="dbp_annotate_data",
            ),
            node(
                func=rake_keywords,
                inputs={
                    "dataframe": "gtr.data_collection.projects.intermediate",
                    "processed_projects": "rake.gtr_data.annotated.oracle",
                },
                outputs="rake.gtr_data.annotated",
                name="rake_annotate_data",
            ),
            node(
                func=yake_keywords,
                inputs={
                    "dataframe": "gtr.data_collection.projects.intermediate",
                    "processed_projects": "yake.gtr_data.annotated.oracle",
                },
                outputs="yake.gtr_data.annotated",
                name="yake_annotate_data",
            ),
            node(
                func=keybert_keywords,
                inputs={
                    "dataframe": "gtr.data_collection.projects.intermediate",
                    "processed_projects": "keybert.gtr_data.annotated.oracle",
                },
                outputs="keybert.gtr_data.annotated.ptd",
                name="keybert_annotate_data",
            ),
            node(
                func=concatenate_partitions,
                inputs={"partitioned_dataset": "keybert.gtr_data.annotated.ptd"},
                outputs="keybert.gtr_data.annotated",
                name="concatenate_keybert_partitions",
            ),
        ],
        tags="annotation",
    )

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
                outputs="keywords.gtr_data.db",
                name="aggregate_keyword_annotators",
            )
        ],
        tags="keyword_processing",
    )

    return annotation_pipeline + aggregate_keywords_pipeline
