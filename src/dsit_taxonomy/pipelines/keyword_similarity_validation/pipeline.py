"""
This is a boilerplate pipeline 'keyword_similarity_validation'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import select_sample_projects, get_expert_labels


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=C0116,W0613
    sample_selection_pipeline = pipeline(
        [
            node(
                func=select_sample_projects,
                inputs="gtr.projects.documents",
                outputs="gtr.projects.sample",
                name="select_sample_projects",
            ),
            node(
                func=get_expert_labels,
                inputs={
                    "taxonomy": "taxonomy.goscience.bottom.db",
                    "data": "gtr.projects.sample",
                },
                outputs="gtr.projects.sample.expert_labels",
                name="get_expert_labels",
            ),
        ]
    )
    return sample_selection_pipeline
