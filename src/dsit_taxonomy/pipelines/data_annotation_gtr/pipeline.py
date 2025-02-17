"""
This pipeline extracts and processes keywords from research project texts
using multiple keyword extraction methods.

The pipeline uses four different extractors:
1. DBpedia Spotlight
   - Entity linking to DBpedia concepts
   - Identifies domain-specific terminology
   - Provides structured knowledge base links

2. RAKE (Rapid Automatic Keyword Extraction)
   - Statistical approach using word co-occurrences
   - Identifies multi-word phrases
   - Scores based on word frequency and co-occurrence

3. YAKE (Yet Another Keyword Extractor)
   - Unsupervised approach for multilingual keyword extraction
   - Uses text features like word position and case
   - Handles domain-specific content well

4. KeyBERT
   - Transformer-based keyword extraction
   - Uses semantic similarity with BERT embeddings
   - Identifies contextually relevant terms

The pipeline processes each project incrementally and handles:
- Batched processing for memory efficiency
- Parallel extraction where possible
- Progress tracking and checkpointing
- Result aggregation and deduplication

Dependencies:
    - pandas
    - keybert
    - dbpedia-spotlight
    - rake-nltk
    - yake
    - concurrent.futures

Example:
    Run the complete annotation pipeline:
    ```
    kedro run --pipeline data_annotation_gtr
    ```
    Or run a specific extractor:
    ```
    kedro run --pipeline data_annotation_gtr --nodes dbp_annotate_data
    ```
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    dbp_keywords,
    rake_keywords,
    yake_keywords,
    keybert_keywords,
    concatenate_partitions,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for extracting keywords from research project texts.

    The pipeline runs multiple keyword extractors in parallel and combines
    their results. It includes checkpointing to handle large datasets
    efficiently.

    Returns:
        Pipeline: A pipeline containing nodes for each keyword extractor
        and result aggregation
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
        ]
    )

    return annotation_pipeline
