"""Pipeline for expert labeling and validation of taxonomy matching."""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    select_sample_projects,
    get_expert_labels,
    prepare_validation_data,
    validate_predictions,
    validate_algorithmic_assignments,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=C0116,W0613
    sample_projects_pipeline = pipeline(
        [
            node(
                func=select_sample_projects,
                inputs={
                    "data": "gtr.projects.documents",
                    "sample_size": "params:sample.size",
                    "sample_random_state": "params:sample.random_state",
                },
                outputs="gtr.projects.sample",
                name="select_sample_projects",
            ),
        ]
    )

    # Expert labeling pipeline
    def expert_labeling_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                node(
                    func=get_expert_labels,
                    inputs={
                        "taxonomy": f"taxonomy.{taxonomy_name}.bottom.db",
                        "data": "gtr.projects.sample",
                        "llm_model": "params:llm.model",
                        "embedding_model": "params:llm.embedding_model",
                        "retriever_k": "params:expert_labeling.retriever_k",
                        "max_retries": "params:llm.max_retries",
                        "system_prompt": "params:expert_labeling.system_prompt",
                        "question_prompt": "params:expert_labeling.question_prompt",
                    },
                    outputs=f"gtr.projects.sample.expert_labels.{taxonomy_name}",
                    name=f"get_expert_labels_{taxonomy_name}",
                ),
                node(
                    func=validate_algorithmic_assignments,
                    inputs={
                        "scores": f"projects.gtr_data.{taxonomy_name}_scores.aggregated",
                        "data": "gtr.projects.sample",
                        "llm_model": "params:llm.model",
                        "max_retries": "params:llm.max_retries",
                        "system_prompt": "params:algo_validation.system_prompt",
                        "question_prompt": "params:algo_validation.question_prompt",
                    },
                    outputs=f"gtr.projects.sample.expert_validation.{taxonomy_name}",
                    name=f"validate_algorithmic_assignments_{taxonomy_name}",
                    # tags=[f"dev_{taxonomy_name}", "dev"],
                ),
            ],
            tags=["expert_labels"],
        )

    # Performance validation pipeline
    def analysis_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                node(
                    func=prepare_validation_data,
                    inputs={
                        "expert_labels": f"gtr.projects.sample.expert_labels.{taxonomy_name}",
                        "expert_validation": f"gtr.projects.sample.expert_validation.{taxonomy_name}",
                        "scores": f"projects.gtr_data.{taxonomy_name}_scores.aggregated",
                    },
                    outputs=[
                        f"validation.{taxonomy_name}.expert_labels.processed",
                        f"validation.{taxonomy_name}.scores.processed",
                    ],
                    name=f"prepare_validation_data_{taxonomy_name}",
                    # tags=[f"dev_{taxonomy_name}", "dev"],
                ),
                node(
                    func=validate_predictions,
                    inputs={
                        "expert_df": f"validation.{taxonomy_name}.expert_labels.processed",
                        "algorithm_df": f"validation.{taxonomy_name}.scores.processed",
                    },
                    outputs=f"validation.{taxonomy_name}.prediction_validation",
                    name=f"validate_predictions_{taxonomy_name}",
                    tags=[f"dev_{taxonomy_name}", "dev", "validation"],
                ),
            ]
        )

    return (
        sample_projects_pipeline
        + sum(expert_labeling_pipeline(tax) for tax in ["cwts", "goscience"])
        + sum(analysis_pipeline(tax) for tax in ["cwts", "goscience"])
    )
