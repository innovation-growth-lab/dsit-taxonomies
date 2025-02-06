"""Pipeline for expert labeling and validation of taxonomy matching."""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    select_sample_projects,
    get_expert_labels,
    prepare_validation_data,
    validate_top_predictions,
)


def create_pipeline(**kwargs) -> Pipeline:
    # Expert labeling pipeline
    expert_labeling = pipeline(
        [
            node(
                func=select_sample_projects,
                inputs={
                    "data": "gtr.projects.documents",
                    "sample_size": "params:expert_validation.sample_size",
                    "sample_random_state": "params:expert_validation.sample_random_state",
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
                    "question_prompt": "params:expert_validation.question_prompt",
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
                    "question_prompt": "params:expert_validation.question_prompt",
                },
                outputs="gtr.projects.sample.expert_labels.cwts",
                name="get_expert_labels_cwts",
            ),
        ],
        tags=["expert_labels"],
    )

    # Performance validation pipeline
    def validation_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                node(
                    func=prepare_validation_data,
                    inputs={
                        "expert_labels": f"gtr.projects.sample.expert_labels.{taxonomy_name}",
                        "scores": f"projects.gtr_data.{taxonomy_name}_scores.aggregated",
                    },
                    outputs=[
                        f"validation.{taxonomy_name}.expert_labels.processed",
                        f"validation.{taxonomy_name}.scores.processed",
                    ],
                    name=f"prepare_validation_data_{taxonomy_name}",
                    tags=[f"dev_{taxonomy_name}"],
                ),
                node(
                    func=validate_top_predictions,
                    inputs={
                        "expert_df": f"validation.{taxonomy_name}.expert_labels.processed",
                        "algorithm_df": f"validation.{taxonomy_name}.scores.processed",
                        "top_n": "params:validation.top_n_predictions",
                    },
                    outputs=f"validation.{taxonomy_name}.prediction_validation",
                    name=f"validate_predictions_{taxonomy_name}",
                    tags=[f"dev_{taxonomy_name}"],
                ),
            ]
        )

    return expert_labeling + sum(
        validation_pipeline(tax) for tax in ["cwts", "goscience"]
    )
