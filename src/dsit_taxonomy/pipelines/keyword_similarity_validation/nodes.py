"""
This is a boilerplate pipeline 'keyword_similarity_validation'
generated using Kedro 0.19.10
"""

import ast, json
import logging
from typing import Generator, Tuple
import pandas as pd
from kedro.io import AbstractDataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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

    loader = DataFrameLoader(taxonomy, page_content_column="label")
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


def validate_algorithmic_assignments(
    scores: pd.DataFrame,
    data: pd.DataFrame,
    llm_model: str,
    max_retries: int,
    system_prompt: str,
    question_prompt: str,
) -> Generator:
    """
    Validate algorithmic assignments using OpenAI.

    Args:
        scores: DataFrame with algorithmic scores (project_id, label, relevance_score)
        data: DataFrame with project data (title, abstract, etc.)
        llm_model: Model name for ChatOpenAI
        max_retries: Maximum number of retry attempts
        system_prompt: System prompt for the LLM
        question_prompt: Question prompt template

    Yields:
        Dictionary mapping project_id to list of validation results
    """
    logger.info("Validating algorithmic assignments using LLM")

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
    scores = scores[scores["project_id"].isin(project_data["project_id"])]

    for i, project_id in enumerate(scores["project_id"].unique()):

        logger.info(
            "Getting project details. Project %d / %d",
            i + 1,
            scores.project_id.nunique(),
        )
        # Get project details
        project_text = project_data[project_data["project_id"] == project_id][
            "text"
        ].iloc[0]
        project_predictions = scores[scores["project_id"] == project_id]

        # Format labels text with IDs
        labels_text = "\n".join(
            [
                f"- {row['label']} (ID: {row['taxonomy_label_id']})"
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


def prepare_validation_data(
    expert_labels: AbstractDataset,
    expert_validation: AbstractDataset,
    scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare expert labels and algorithmic scores for comparison.

    Args:
        expert_labels: Partitioned dataset with expert labels.
        scores: DataFrame with algorithm-assigned scores (containing sentence and keyword scores)

    Returns:
        Tuple of processed expert labels and algorithmic scores
    """
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
                    "label": label_dict["label"],
                    "likelihood": label_dict["likelihood"],
                }
            )

    expert_df = pd.DataFrame(labels_data)

    # map the labels to the taxonomy_label_id
    expert_df = expert_df.merge(
        scores.drop_duplicates(subset=["label", "taxonomy_label_id"])[
            ["label", "taxonomy_label_id"]
        ],
        on="label",
        how="left",
    )

    scores_data = []
    for i, (project_id, loader_func) in enumerate(expert_validation.items()):
        logger.info("Processing validation: %d / %d", i + 1, len(expert_validation))
        for label_dict in loader_func():
            scores_data.append(
                {
                    "project_id": project_id,
                    "taxonomy_label_id": label_dict["taxonomy_label_id"],
                    "positive": label_dict["positive"],
                    "explanation": label_dict["explanation"],
                }
            )

    scores_df = pd.DataFrame(scores_data)

    # map the id to the label
    algorithmic_df = scores_df.merge(
        scores.drop_duplicates(subset=["project_id", "taxonomy_label_id", "label"])[
            ["project_id", "taxonomy_label_id", "label", "final_bin"]
        ],
        on=["project_id", "taxonomy_label_id"],
        how="left",
    )

    return expert_df, algorithmic_df


def validate_predictions(
    expert_df: pd.DataFrame,
    algorithm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate predictions by comparing expert high likelihood labels with algorithmic predictions.
    Computes metrics for both strict (high only) and relaxed (high+medium) algorithmic confidence.

    Args:
        expert_df: DataFrame with expert labels (columns: project_id, label, likelihood)
        algorithm_df: DataFrame with algorithm predictions (columns: project_id, label, final_bin, positive)

    Returns:
        DataFrame with performance metrics for both strict and relaxed confidence thresholds
    """
    logger.info("Validating predictions against expert high likelihood labels")

    # Remove hallucinated labels and convert to lowercase
    expert_df = expert_df.dropna(subset=["taxonomy_label_id"])
    expert_df["likelihood"] = expert_df["likelihood"].str.lower()

    # Merge expert and algorithm predictions
    data = pd.merge(
        algorithm_df,
        expert_df,
        on=["project_id", "taxonomy_label_id"],
        how="outer",
    )

    # Clean up labels
    data["label"] = data["label_x"].fillna(data["label_y"])
    data = data.drop(columns=["label_x", "label_y"])

    metrics = []

    # Calculate metrics for both strict and relaxed thresholds
    for threshold in ["strict", "relaxed"]:
        # True positives: Algorithm predicted high (or high/medium) AND expert gave high likelihood
        algo_condition = (
            (data["final_bin"] == "high")
            if threshold == "strict"
            else (data["final_bin"].isin(["high", "medium"]))
        )

        true_positives = sum(
            algo_condition & (data["likelihood"] == "high") & (data["positive"] == True)
        )

        # False positives: Algorithm predicted high (or high/medium) but expert either:
        # - Didn't label it at all
        # - Gave medium/low likelihood
        false_positives = sum(
            algo_condition
            & ((data["likelihood"] != "high") | pd.isna(data["likelihood"]))
            & (data["positive"] == True)
        )

        # False negatives: Expert gave high likelihood but algorithm either:
        # - Missed it completely
        # - Assigned lower confidence
        false_negatives = sum(
            (data["likelihood"] == "high")
            & ((~algo_condition) | pd.isna(data["final_bin"]))
        )

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics.append(
            {
                "threshold": threshold,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        logger.info(
            "%s Threshold Metrics (algo: %s, expert: high):\n"
            "Precision: %0.3f\n"
            "Recall: %0.3f\n"
            "F1 Score: %0.3f\n"
            "True Positives: %d\n"
            "False Positives: %d\n"
            "False Negatives: %d",
            threshold.title(),
            "high only" if threshold == "strict" else "high+medium",
            precision,
            recall,
            f1,
            true_positives,
            false_positives,
            false_negatives,
        )

    return pd.DataFrame(metrics)


def tune_matching_parameters(
    scores: pd.DataFrame,
    expert_df: pd.DataFrame,
    param_grid: dict,
) -> pd.DataFrame:
    """
    Tune parameters for aggregate_scores_to_labels to maximize validation metrics.

    Args:
        scores: Raw scores DataFrame
        expert_df: Expert validation DataFrame
        param_grid: Dictionary of parameters to try, e.g.:
            {
                "sentence_weight": [0.5, 0.66, 0.75],
                "keyword_weight": [0.25, 0.33, 0.5],
                "similarity_quantile_threshold": [0.7, 0.8, 0.9],
                "global_q2_threshold": [0.4, 0.5, 0.6],
                "global_q3_threshold": [0.7, 0.75, 0.8],
                "local_q2_threshold": [0.4, 0.5, 0.6],
                "local_q3_threshold": [0.7, 0.75, 0.8]
            }

    Returns:
        DataFrame with parameter combinations and their validation metrics
    """
    from itertools import product
    from ..keyword_similarity_matching.nodes import aggregate_scores_to_labels

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(product(*param_grid.values()))

    results = []
    total_combinations = len(param_values)

    for i, values in enumerate(param_values, 1):
        params = dict(zip(param_names, values))
        logger.info("Testing combination %d/%d: %s", i, total_combinations, params)

        # Aggregate scores with current parameters
        aggregated_scores = aggregate_scores_to_labels(scores.copy(), **params)

        # Validate predictions
        validation_metrics = validate_predictions(expert_df.copy(), aggregated_scores)

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

    return pd.DataFrame(results)
