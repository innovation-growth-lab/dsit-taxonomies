"""
This is a boilerplate pipeline 'data_annotation_gtr'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import dbp_keywords, rake_keywords, yake_keywords, keybert_keywords


def create_pipeline(  # pylint: disable=unused-argument&missing-function-docstring
    **kwargs,
) -> Pipeline:
    annotation_pipeline = pipeline(
        [
            node(
                func=dbp_keywords,
                inputs="gtr.data_collection.projects.intermediate",
                outputs="dbp.gtr_data.annotated",
                name="dbp_annotate_data",
            ),
            node(
                func=rake_keywords,
                inputs="gtr.data_collection.projects.intermediate",
                outputs="rake.gtr_data.annotated",
                name="rake_annotate_data",
            ),
            node(
                func=yake_keywords,
                inputs="gtr.data_collection.projects.intermediate",
                outputs="yake.gtr_data.annotated",
                name="yake_annotate_data",
            ),
            node(
                func=keybert_keywords,
                inputs="gtr.data_collection.projects.intermediate",
                outputs="keybert.gtr_data.annotated",
                name="keybert_annotate_data",
            ),
        ]
    )

    return annotation_pipeline
