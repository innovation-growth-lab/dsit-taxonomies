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
from .nodes import aggregate_keyword_annotators, generate_keyword_embeddings


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """
    Creates a pipeline for processing and embedding keywords.

    The pipeline aggregates keywords from multiple extractors and generates
    embeddings for downstream similarity matching.

    Returns:
        Pipeline: A pipeline containing keyword aggregation and embedding nodes
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
                outputs="keywords.gtr_data.db",
                name="aggregate_keyword_annotators",
            ),
            node(
                func=generate_keyword_embeddings,
                inputs="keywords.gtr_data.db",
                outputs="keywords.gtr_data.embeddings",
                name="generate_keyword_embeddings",
            ),
        ],
        tags="keyword_processing"
    )

    return aggregate_keywords_pipeline
