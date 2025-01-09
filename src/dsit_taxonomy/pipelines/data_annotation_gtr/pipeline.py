"""
This is a boilerplate pipeline 'data_annotation_gtr'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    dbp_keywords,
    rake_keywords,
    yake_keywords,
    keybert_keywords,
    concatenate_partitions,
)


def create_pipeline(  # pylint: disable=unused-argument&missing-function-docstring
    **kwargs,
) -> Pipeline:
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
