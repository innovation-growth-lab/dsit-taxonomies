"""
This is a boilerplate pipeline 'keyword_similarity_validation'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import select_sample_projects, get_expert_labels


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=C0116,W0613
    return pipeline(
        [
            node(
                func=select_sample_projects,
                inputs={
                    "data": "gtr.projects.documents",
                    "sample_size": "params:expert_validation.sample_size",
                    "sample_random_state": "params:expert_validation.sample_random_state"
                },
                outputs="gtr.projects.sample",
                name="select_sample_projects",
            ),
            node(
                func=get_expert_labels,
                inputs={
                    "taxonomy": "taxonomy.goscience.bottom.db",
                    "data": "gtr.projects.sample",
                    "llm_model": "params:expert_validation.llm_model",
                    "embedding_model": "params:expert_validation.embedding_model",
                    "retriever_k": "params:expert_validation.retriever_k",
                    "max_retries": "params:expert_validation.max_retries",
                    "system_prompt": "params:expert_validation.system_prompt",
                    "question_prompt": "params:expert_validation.question_prompt"
                },
                outputs="gtr.projects.sample.expert_labels.goscience",
                name="get_expert_labels_goscience",
            ),
            node(
                func=get_expert_labels,
                inputs={
                    "taxonomy": "taxonomy.cwts.bottom.db",
                    "data": "gtr.projects.sample",
                    "llm_model": "params:expert_validation.llm_model",
                    "embedding_model": "params:expert_validation.embedding_model",
                    "retriever_k": "params:expert_validation.retriever_k",
                    "max_retries": "params:expert_validation.max_retries",
                    "system_prompt": "params:expert_validation.system_prompt",
                    "question_prompt": "params:expert_validation.question_prompt"
                },
                outputs="gtr.projects.sample.expert_labels.cwts",
                name="get_expert_labels_cwts",
            ),
        ]
    )
