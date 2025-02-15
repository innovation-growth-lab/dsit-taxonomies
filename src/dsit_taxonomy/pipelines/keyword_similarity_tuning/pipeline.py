"""Pipeline for expert labeling and fine-tuning of taxonomy matching parameters."""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    select_sample_projects,
    get_expert_labels,
    get_expert_assessment,
    # prepare_tuning_data,
    # tune_matching_parameters,
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
                    func=get_expert_assessment,
                    inputs={
                        "aggregated_scores": f"projects.gtr_data.{taxonomy_name}_scores.aggregated",
                        "data": "gtr.projects.sample",
                        "llm_model": "params:llm.model",
                        "max_retries": "params:llm.max_retries",
                        "system_prompt": "params:expert_assessment.system_prompt",
                        "question_prompt": "params:expert_assessment.question_prompt",
                    },
                    outputs=f"gtr.projects.sample.expert_assessment.{taxonomy_name}",
                    name=f"evaluate_algorithmic_assignments_{taxonomy_name}",
                    tags=[f"dev_{taxonomy_name}", "evaluate_algorithms", "get_expert_assessment"],
                ),
            ],
            tags=["expert_labels"],
        )

    # # Parameter tuning pipeline
    # def tuning_pipeline(taxonomy_name: str) -> Pipeline:
    #     return pipeline(
    #         [
    #             node(
    #                 func=prepare_tuning_data,
    #                 inputs={
    #                     "expert_labels": f"gtr.projects.sample.expert_labels.{taxonomy_name}",
    #                     "expert_tuning": f"gtr.projects.sample.expert_tuning.{taxonomy_name}",
    #                     "taxonomy": f"taxonomy.{taxonomy_name}.bottom.db",
    #                 },
    #                 outputs=[
    #                     f"tuning.{taxonomy_name}.expert_labels.processed",
    #                     f"tuning.{taxonomy_name}.scores.processed",
    #                 ],
    #                 name=f"prepare_tuning_data_{taxonomy_name}",
    #                 tags=[f"dev_{taxonomy_name}", "dev"],
    #             ),
    #             node(
    #                 func=tune_matching_parameters,
    #                 inputs={
    #                     "scores": f"projects.gtr_data.{taxonomy_name}_scores.detailed",
    #                     "expert_df": f"tuning.{taxonomy_name}.expert_labels.processed",
    #                     "algorithm_df": f"tuning.{taxonomy_name}.scores.processed",
    #                     "param_grid": "params:tuning.parameter_tuning.param_grid",
    #                 },
    #                 outputs=[
    #                     f"tuning.{taxonomy_name}.parameter_tuning_results",
    #                     f"tuning.{taxonomy_name}.project_results",
    #                 ],
    #                 name=f"tune_matching_parameters_{taxonomy_name}",
    #                 tags=[f"tuning_{taxonomy_name}", "tuning"],
    #             ),
    #         ]
    #     )

    return (
        sample_projects_pipeline
        + sum(expert_labeling_pipeline(tax) for tax in ["cwts", "goscience"])
        # + sum(tuning_pipeline(tax) for tax in ["cwts", "goscience"])
    )
