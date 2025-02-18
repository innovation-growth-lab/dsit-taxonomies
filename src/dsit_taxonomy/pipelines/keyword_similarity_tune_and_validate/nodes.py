"""
This module contains nodes for tuning and validating taxonomy matching parameters.

The nodes handle:
- Sampling projects for expert validation
- Getting expert labels using RAG models
- Parameter tuning using expert feedback
- Computing validation metrics

The module provides functionality for:
1. Sample Selection
   - Selecting representative projects for validation
   - Balancing across different project types

2. Expert Labeling
   - Using retrieval-augmented generation
   - Providing structured taxonomy assignments
   - Handling multiple rounds of validation

3. Confidence Bin Parameter Optimisation
   - Grid search over parameter space
   - Computing metrics for each parameter set
   - Finding optimal weights and thresholds

4. Validation Metrics
   - Computing precision, recall, F1 scores
   - Tracking true/false positives/negatives
   - Providing per-project and global metrics
"""

import ast, json
import logging
from typing import Generator, Tuple
from itertools import product
import pandas as pd
from kedro.io import AbstractDataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from ..keyword_similarity_matching.nodes import (
    prune_raw_matches,
    combine_scores,
)
from ..keyword_similarity_refinement.nodes import aggregate_scores_to_labels
from .utils import (
    compute_likelihood_agreement,
    compute_per_project_metrics,
    validate_predictions,
)

logger = logging.getLogger(__name__)


def select_sample_projects(
    data: pd.DataFrame, sample_size: int, sample_random_state: int
) -> pd.DataFrame:
    """Select a sample of projects for expert labeling."""
    return data.sample(n=sample_size, random_state=sample_random_state)


def get_expert_labels(
    taxonomy: pd.DataFrame,
    data: pd.DataFrame,
    llm_model: str,
    embedding_model: str,
    retriever_k: int,
    max_retries: int,
    system_prompt: str,
    question_prompt: str,
) -> Generator:
    """
    Get expert labels for a sample of projects using a retrieval-augmented generative model.

    Args:
        taxonomy: The taxonomy dataframe.
        data: The sample of projects to label.
        llm_model: Model name for ChatOpenAI
        embedding_model: Model name for embeddings
        retriever_k: Number of labels to retrieve
        max_retries: Maximum number of retry attempts
        system_prompt: System prompt template
        question_prompt: Question prompt template

    Yields:
        A dictionary containing the project ID and the expert labels.
    """
    logger.info("Getting expert labels")

    model = ChatOpenAI(model=llm_model)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = InMemoryVectorStore(embeddings)

    loader = DataFrameLoader(taxonomy, page_content_column="taxonomy_label")
    _ = vector_store.add_documents(loader.load())
    retriever = vector_store.as_retriever(search_kwargs={"k": retriever_k})

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    question_answer_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Combine project text fields
    data["text"] = data.apply(
        lambda row: ". ".join(
            row[col] if row[col] is not None else ""
            for col in [
                "title",
                "abstract_text",
                "tech_abstract_text",
                "potential_impact",
            ]
        ),
        axis=1,
    )

    for i, (_, row) in enumerate(data.iterrows()):
        logger.info(
            "Processing project: %s. Number: %d/%d", row["title"], i + 1, len(data)
        )

        question = question_prompt.format(abstract=row["text"])

        for attempt in range(max_retries):
            try:
                response = rag_chain.invoke({"input": question})
                response_json = ast.literal_eval(response["answer"])
                yield {row["project_id"]: response_json}
                break
            except (ValueError, SyntaxError) as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "Failed to decode JSON response after %d attempts for project %s: %s",
                        max_retries,
                        row["project_id"],
                        e,
                    )
                else:
                    logger.warning(
                        "Attempt %d failed for project %s, retrying...",
                        attempt + 1,
                        row["project_id"],
                    )


def prepare_tuning_data(
    expert_labels: AbstractDataset,
    expert_assessment: AbstractDataset,
    taxonomy: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare expert labels and assessment scores for parameter tuning.

    Args:
        expert_labels: Partitioned dataset with expert labels.
        expert_validation: Partitioned dataset with expert validation.

    Returns:
        Tuple of processed expert labels and assessment scores
    """
    taxonomy.rename(columns={"uuid": "taxonomy_label_id"}, inplace=True)
    logger.info("Preparing validation data")

    # Read the jsons from the expert_labels dataset and flatten into DataFrame
    labels_data = []
    for i, (project_id, loader_func) in enumerate(expert_labels.items()):
        logger.info("Processing label: %d / %d", i + 1, len(expert_labels))
        # Each project can have multiple labels
        for label_dict in loader_func():
            if "error" in label_dict:
                logger.warning(
                    "Error in expert labels for project %s: %s", project_id, label_dict
                )
                continue
            labels_data.append(
                {
                    "project_id": project_id,
                    "label": label_dict.get("taxonomy_label", label_dict.get("label")),
                    "likelihood": label_dict["likelihood"],
                }
            )

    expert_df = pd.DataFrame(labels_data)

    # map the labels to the taxonomy_label_id
    expert_df = expert_df.merge(
        taxonomy.drop_duplicates(subset=["label", "taxonomy_label_id"])[
            ["label", "taxonomy_label_id"]
        ],
        on="label",
        how="left",
    )

    # rename the label column
    expert_df.rename(columns={"label": "taxonomy_label"}, inplace=True)

    # remove hallucinated labels
    expert_df = expert_df.dropna(subset=["taxonomy_label_id"])

    scores_data = []
    for i, (project_id, loader_func) in enumerate(expert_assessment.items()):
        logger.info("Processing validation: %d / %d", i + 1, len(expert_assessment))
        for label_dict in loader_func():
            scores_data.append(
                {
                    "project_id": project_id,
                    "taxonomy_label_id": label_dict["taxonomy_label_id"],
                    "positive": label_dict["positive"],
                    "explanation": label_dict["explanation"],
                }
            )

    assessment_df = pd.DataFrame(scores_data)

    # map the id to the label
    assessment_df = assessment_df.merge(
        taxonomy.drop_duplicates(subset=["label", "taxonomy_label_id"])[
            ["label", "taxonomy_label_id"]
        ],
        on="taxonomy_label_id",
        how="left",
    )

    # rename the label column
    assessment_df.rename(columns={"label": "taxonomy_label"}, inplace=True)

    return expert_df, assessment_df


def get_expert_assessment(
    aggregated_scores: pd.DataFrame,
    data: pd.DataFrame,
    llm_model: str,
    max_retries: int,
    system_prompt: str,
    question_prompt: str,
) -> Generator:
    """
    Evaluate assessment assignments using OpenAI.

    Args:
        scores: DataFrame with assessment scores (project_id, label, relevance_score)
        data: DataFrame with project data (title, abstract, etc.)
        llm_model: Model name for ChatOpenAI
        max_retries: Maximum number of retry attempts
        system_prompt: System prompt for the LLM
        question_prompt: Question prompt template

    Yields:
        Dictionary mapping project_id to list of validation results
    """

    logger.info("Validating assessment assignments using LLM")

    model = ChatOpenAI(model=llm_model)

    # Combine project text fields
    project_data = data.copy()
    project_data["text"] = project_data.apply(
        lambda row: "\n".join(
            filter(
                None,
                [
                    f"TITLE: {row['title']}",
                    (
                        f"ABSTRACT: {row['abstract_text']}"
                        if pd.notna(row["abstract_text"])
                        else None
                    ),
                    (
                        f"TECHNICAL ABSTRACT: {row['tech_abstract_text']}"
                        if pd.notna(row["tech_abstract_text"])
                        else None
                    ),
                    (
                        f"POTENTIAL IMPACT: {row['potential_impact']}"
                        if pd.notna(row["potential_impact"])
                        else None
                    ),
                ],
            )
        ),
        axis=1,
    )

    # get scores for the sample projects
    aggregated_scores = aggregated_scores[
        aggregated_scores["project_id"].isin(project_data["project_id"])
    ]

    for i, project_id in enumerate(aggregated_scores["project_id"].unique()):

        logger.info(
            "Getting project details. Project %d / %d",
            i + 1,
            aggregated_scores.project_id.nunique(),
        )
        # Get project details
        project_text = project_data[project_data["project_id"] == project_id][
            "text"
        ].iloc[0]
        project_predictions = aggregated_scores[
            aggregated_scores["project_id"] == project_id
        ]

        # Format labels text with IDs
        labels_text = "\n".join(
            [
                f"- {row["taxonomy_label"]} (ID: {row["taxonomy_label_id"]})"
                for _, row in project_predictions.iterrows()
            ]
        )

        formatted_question = question_prompt.format(
            project_text=project_text, labels_text=labels_text
        )

        for attempt in range(max_retries):
            try:
                response = model.invoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": formatted_question},
                    ]
                )

                # if "```json" is in the response, remove it
                if "```json" in response.content:
                    response.content = response.content.replace("```json", "").replace(
                        "```", ""
                    )

                response_json = json.loads(response.content)

                # Validate response format
                for item in response_json:
                    if not all(
                        k in item
                        for k in ["taxonomy_label_id", "positive", "explanation"]
                    ):
                        raise ValueError("Missing required fields in response")

                yield {project_id: response_json}
                break
            except (ValueError, SyntaxError) as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "Failed to decode JSON response after %d attempts for project %s: %s",
                        max_retries,
                        project_id,
                        e,
                    )
                else:
                    logger.warning(
                        "Attempt %d failed for project %s, retrying...",
                        attempt + 1,
                        project_id,
                    )


def tune_matching_parameters(
    sentence_matches: pd.DataFrame,
    global_matches: pd.DataFrame,
    keyword_matches: pd.DataFrame,
    assessment_df: pd.DataFrame,
    expert_df: pd.DataFrame,
    param_grid: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tune parameters for aggregate_scores_to_labels to maximise validation metrics.

    Args:
        sentence_matches: DataFrame with sentence-level matches
        global_matches: DataFrame with global matches
        keyword_matches: DataFrame with keyword matches
        expert_df: Expert validation DataFrame
        param_grid: Dictionary of parameters to try, e.g.:
            {
                use_quantile: [False, True]
                sentence_threshold: [0.6, 0.7, 0.8]
                global_threshold: [0.6, 0.7, 0.8]
                keyword_threshold: [0.6, 0.7, 0.8]
                sentence_weight: [0.5, 0.6, 0.7]
                global_weight: [0.1, 0.2, 0.3]
                global_q2_threshold: [0.4, 0.5, 0.6]
                global_q3_threshold: [0.7, 0.8, 0.9]
                local_q2_threshold: [0.4, 0.5, 0.6]
                local_q3_threshold: [0.7, 0.8, 0.9]
            }

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame with parameter combinations and their validation metrics
            - DataFrame with per-project metrics for each parameter combination
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(product(*param_grid.values()))

    results = []
    project_results = []
    total_combinations = len(param_values)

    for i, values in enumerate(param_values, 1):
        params = dict(zip(param_names, values))
        logger.info("Testing combination %d/%d: %s", i, total_combinations, params)

        # prune raw matches
        pruned_sentences, pruned_globals, pruned_keywords = prune_raw_matches(
            sentence_matches=sentence_matches.copy(),
            global_matches=global_matches.copy(),
            keyword_matches=keyword_matches.copy(),
            sentence_threshold=params["sentence_threshold"],
            global_threshold=params["global_threshold"],
            keyword_threshold=params["keyword_threshold"],
            use_quantile=params["use_quantile"],
        )

        granular_scores = combine_scores(
            sentence_scores=pruned_sentences,
            keyword_scores=pruned_keywords,
            global_scores=pruned_globals,
            sentence_weight=params["sentence_weight"],
            global_weight=params["global_weight"],
        )

        # Aggregate scores with current parameters
        aggregated_scores = aggregate_scores_to_labels(
            granular_scores,
            normalise_by_matches=params["normalise_by_matches"],
            global_q2_threshold=params["global_q2_threshold"],
            global_q3_threshold=params["global_q3_threshold"],
            local_q2_threshold=params["local_q2_threshold"],
            local_q3_threshold=params["local_q3_threshold"],
        )

        # map the id to the label
        grid_assessment_df = assessment_df.merge(
            aggregated_scores[["project_id", "taxonomy_label_id", "confidence_bin"]],
            on=["project_id", "taxonomy_label_id"],
            how="left",
        )

        # Get per-project metrics
        project_metrics = compute_per_project_metrics(
            expert_df.copy(), grid_assessment_df.copy()
        )

        # Add parameters to project results
        for param_name, param_value in params.items():
            project_metrics[param_name] = param_value
        project_results.append(project_metrics)

        # Validate predictions for overall metrics
        validation_metrics = validate_predictions(
            expert_df.copy(), grid_assessment_df.copy()
        )

        # Add parameters to results
        result = params.copy()
        for _, row in validation_metrics.iterrows():
            threshold = row["threshold"]
            result.update(
                {
                    f"{threshold}_precision": row["precision"],
                    f"{threshold}_recall": row["recall"],
                    f"{threshold}_f1": row["f1_score"],
                    f"{threshold}_tp": row["true_positives"],
                    f"{threshold}_fp": row["false_positives"],
                    f"{threshold}_fn": row["false_negatives"],
                }
            )

        results.append(result)

        # Log current best results
        results_df = pd.DataFrame(results)
        best_strict = results_df.nlargest(1, "strict_f1").iloc[0]
        best_relaxed = results_df.nlargest(1, "relaxed_f1").iloc[0]

        logger.info(
            "Current best results:\n"
            "Strict (F1=%0.3f):\n%s\n"
            "Relaxed (F1=%0.3f):\n%s",
            best_strict["strict_f1"],
            {
                k: v
                for k, v in best_strict.items()
                if not k.startswith(("strict_", "relaxed_"))
            },
            best_relaxed["relaxed_f1"],
            {
                k: v
                for k, v in best_relaxed.items()
                if not k.startswith(("strict_", "relaxed_"))
            },
        )

    # Combine all project results
    project_results_df = pd.concat(project_results, ignore_index=True)

    return pd.DataFrame(results), project_results_df


def evaluate_zeroshot_quality(
    zeroshot_scores: pd.DataFrame,
    expert_df: pd.DataFrame,
    assessment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate the quality of zero-shot confidence scores against expert labels.

    This function:
    1. Compares zero-shot bins with expert likelihood ratings
    2. Compares zero-shot bins with expert assessment validations
    3. Computes agreement metrics and confusion matrices
    4. Analyses false positives and negatives

    Args:
        zeroshot_scores: DataFrame containing:
            - project_id: Project identifier
            - taxonomy_label_id: Label identifier
            - zeroshot_score: Raw zero-shot confidence score
            - zeroshot_bin: Binned confidence ('very high', 'high', etc.)
        expert_df: DataFrame with expert labels containing:
            - project_id: Project identifier
            - taxonomy_label_id: Label identifier
            - likelihood: Expert assigned likelihood ('high', 'medium', 'low')
        assessment_df: DataFrame with expert assessments containing:
            - project_id: Project identifier
            - taxonomy_label_id: Label identifier
            - positive: Boolean indicating expert validation

    Returns:
        DataFrame containing quality metrics:
            - metric_type: Type of metric (agreement, precision, recall, etc.)
            - threshold: Confidence threshold used ('strict', 'relaxed')
            - value: Metric value
            - details: Additional information about the metric
    """
    logger.info("Evaluating zero-shot classification quality")

    # Merge datasets
    merged_expert = pd.merge(
        zeroshot_scores,
        expert_df,
        on=["project_id", "taxonomy_label_id"],
        how="inner",
        suffixes=("", "_expert"),
    )

    merged_assessment = pd.merge(
        zeroshot_scores,
        assessment_df,
        on=["project_id", "taxonomy_label_id"],
        how="inner",
        suffixes=("", "_assessment"),
    )

    metrics = []

    merged_expert["agreement"] = merged_expert.apply(
        compute_likelihood_agreement, axis=1
    )

    agreement_stats = merged_expert["agreement"].value_counts(normalize=True)
    for agreement_type in ["strong", "weak", "disagree"]:
        metrics.append(
            {
                "metric_type": "expert_likelihood_agreement",
                "threshold": agreement_type,
                "value": agreement_stats.get(agreement_type, 0),
                "details": f"Proportion of {agreement_type} agreement with expert likelihood",
            }
        )

    # Analyse agreement with expert assessment
    for threshold in ["strict", "relaxed"]:
        # Define what counts as a positive prediction
        if threshold == "strict":
            zeroshot_positive = merged_assessment["zeroshot_bin"].isin(
                ["very high", "high"]
            )
        else:
            zeroshot_positive = merged_assessment["zeroshot_bin"].isin(
                ["very high", "high", "medium"]
            )

        true_pos = sum(zeroshot_positive & merged_assessment["positive"])
        false_pos = sum(zeroshot_positive & ~merged_assessment["positive"])
        false_neg = sum(~zeroshot_positive & merged_assessment["positive"])
        true_neg = sum(~zeroshot_positive & ~merged_assessment["positive"])

        precision = (
            true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        )
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics.extend(
            [
                {
                    "metric_type": "precision",
                    "threshold": threshold,
                    "value": precision,
                    "details": f"Precision using {threshold} threshold",
                },
                {
                    "metric_type": "recall",
                    "threshold": threshold,
                    "value": recall,
                    "details": f"Recall using {threshold} threshold",
                },
                {
                    "metric_type": "f1",
                    "threshold": threshold,
                    "value": f1,
                    "details": f"F1 score using {threshold} threshold",
                },
            ]
        )

        # Log detailed results
        logger.info(
            "%s threshold metrics:\n"
            "Precision: %.3f\n"
            "Recall: %.3f\n"
            "F1: %.3f\n"
            "True Positives: %d\n"
            "False Positives: %d\n"
            "False Negatives: %d\n"
            "True Negatives: %d",
            threshold.title(),
            precision,
            recall,
            f1,
            true_pos,
            false_pos,
            false_neg,
            true_neg,
        )

    return pd.DataFrame(metrics)
