"""
This pipeline processes and standardises multiple research taxonomies for
use in project classification.

The pipeline processes three taxonomies:
1. CWTS Topics
   - Research topics from the CWTS Leiden Ranking
   - Hierarchical structure with multiple levels
   - Includes topic descriptions and keywords

2. GO-SCIENCE Areas
   - UK government research classification
   - Structured hierarchy of research areas
   - Contains detailed area descriptions

3. Open Alex Concepts
   - Research concept ontology from OpenAlex
   - Covers broad range of academic disciplines
   - Includes concept relationships and levels

For each taxonomy, the pipeline:
- Standardises the format and structure
- Extracts relevant metadata
- Generates two versions:
  * Full hierarchy with all levels
  * Bottom-level terms only for direct matching

Dependencies:
    - pandas
    - numpy
    - uuid

Example:
    Run the complete taxonomy processing:
    ```
    kedro run --pipeline data_processing_taxonomies
    ```
    Or process a specific taxonomy:
    ```
    kedro run --pipeline data_processing_taxonomies --nodes preprocess_cwts_topics
    ```
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocess_cwts_topics,
    preprocess_goscience_taxonomy,
    preprocess_oa_concepts,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=W0613
    """
    Creates a pipeline for processing multiple research taxonomies.

    The pipeline standardises three different taxonomies (CWTS, GO-SCIENCE,
    OpenAlex) into a consistent format for downstream matching.

    Returns:
        Pipeline: A pipeline containing nodes for processing each taxonomy
    """
    taxonomy_keywords_pipeline = pipeline(
        [
            node(
                func=preprocess_cwts_topics,
                inputs="taxonomy.cwts.raw",
                outputs=["taxonomy.cwts.full.db", "taxonomy.cwts.bottom.db"],
                name="preprocess_cwts_topics",
                tags=["cwts"],
            ),
            node(
                func=preprocess_goscience_taxonomy,
                inputs="taxonomy.goscience.raw",
                outputs=["taxonomy.goscience.full.db", "taxonomy.goscience.bottom.db"],
                name="preprocess_goscience_taxonomy",
                tags=["goscience"],
            ),
            node(
                func=preprocess_oa_concepts,
                inputs="taxonomy.oa_concepts.raw",
                outputs=[
                    "taxonomy.oa_concepts.full.db",
                    "taxonomy.oa_concepts.bottom.db",
                ],
                name="preprocess_oa_concepts",
                tags=["oa_concepts"],
            ),
        ],
        tags="taxonomy_processing"
    )

    return taxonomy_keywords_pipeline
