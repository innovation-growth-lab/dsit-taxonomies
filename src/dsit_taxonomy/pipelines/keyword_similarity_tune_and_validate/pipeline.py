"""Pipeline for expert labeling and fine-tuning of taxonomy matching parameters."""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    select_sample_projects,
    get_expert_labels,
    get_expert_assessment,
    prepare_tuning_data,
    tune_matching_parameters,
    evaluate_scoring_quality,
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
                    name=f"evaluate_assessment_assignments_{taxonomy_name}",
                    tags=[
                        f"dev_{taxonomy_name}",
                        "evaluate_assessments",
                        "get_expert_assessment",
                    ],
                ),
            ],
            tags=["expert_labels"],
        )

    # Parameter tuning pipeline
    def tuning_confidence_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                node(
                    func=prepare_tuning_data,
                    inputs={
                        "expert_labels": f"gtr.projects.sample.expert_labels.{taxonomy_name}",
                        "expert_assessment": f"gtr.projects.sample.expert_assessment.{taxonomy_name}",
                        "taxonomy": f"taxonomy.{taxonomy_name}.bottom.db",
                    },
                    outputs=[
                        f"tuning.{taxonomy_name}.expert_labels.processed",
                        f"tuning.{taxonomy_name}.expert_assessment.processed",
                    ],
                    name=f"prepare_tuning_data_{taxonomy_name}",
                    tags=[f"tuning_{taxonomy_name}", "dev", "tuning"],
                ),
                node(
                    func=tune_matching_parameters,
                    inputs={
                        "sentence_matches": f"sentences.gtr_data.{taxonomy_name}_matches.intermediate",
                        "keyword_matches": f"keywords.gtr_data.{taxonomy_name}_matches.intermediate",
                        "global_matches": f"projects.gtr_data.{taxonomy_name}_matches.raw",
                        "expert_df": f"tuning.{taxonomy_name}.expert_labels.processed",
                        "assessment_df": f"tuning.{taxonomy_name}.expert_assessment.processed",
                        "param_grid": "params:tuning.parameter_tuning.param_grid",
                    },
                    outputs=[
                        f"tuning.{taxonomy_name}.parameter_tuning_results",
                        f"tuning.{taxonomy_name}.project_results",
                    ],
                    name=f"tune_matching_parameters_{taxonomy_name}",
                    tags=[f"tuning_{taxonomy_name}", "tuning"],
                ),
            ]
        )

    def evaluate_zeroshot_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                node(
                    func=evaluate_scoring_quality,
                    inputs={
                        "zeroshot_scores": f"projects.gtr_data.{taxonomy_name}_scores.zeroshot",
                        "expert_df": f"tuning.{taxonomy_name}.expert_labels.processed",
                        "assessment_df": f"tuning.{taxonomy_name}.expert_assessment.processed",
                    },
                    outputs=f"validate.{taxonomy_name}.zeroshot_quality_metrics",
                    name=f"evaluate_scoring_quality_{taxonomy_name}",
                    tags=[f"validate_{taxonomy_name}", "evaluate_scoring_quality"],
                ),
            ]
        )

    return (
        sample_projects_pipeline
        + sum(expert_labeling_pipeline(tax) for tax in ["cwts", "goscience"])
        + sum(tuning_confidence_pipeline(tax) for tax in ["cwts", "goscience"])
        + sum(evaluate_zeroshot_pipeline(tax) for tax in ["cwts", "goscience"])
    )
