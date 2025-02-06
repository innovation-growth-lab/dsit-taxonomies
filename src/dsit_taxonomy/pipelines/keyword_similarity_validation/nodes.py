"""
This is a boilerplate pipeline 'keyword_similarity_validation'
generated using Kedro 0.19.10
"""

import ast
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


def prepare_validation_data(
    expert_labels: AbstractDataset,
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
            labels_data.append({
                "project_id": project_id,
                "label": label_dict["label"],
                "likelihood": label_dict["likelihood"]
            })
    
    expert_df = pd.DataFrame(labels_data)
    
    # Only keep predicted scores for the project_ids in expert_df
    algorithmic_df = scores[scores["project_id"].isin(expert_df["project_id"])]

    return expert_df, algorithmic_df


def validate_top_predictions(
    expert_df: pd.DataFrame,
    algorithm_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Validate whether top algorithm predictions match expert high-likelihood labels.
    
    Args:
        expert_df: DataFrame with expert labels (columns: project_id, label, likelihood)
        algorithm_df: DataFrame with algorithm predictions (columns: project_id, label, relevance_score)
        top_n: Number of top predictions to consider per project
        
    Returns:
        DataFrame with validation results per project
    """
    logger.info("Validating top %d predictions against expert labels", top_n)
    
    # Get high-likelihood expert labels
    high_conf_expert = expert_df[expert_df["likelihood"] == "high"].copy()
    
    # Get top N predictions per project
    top_predictions = (
        algorithm_df
        .sort_values(["project_id", "relevance_score"], ascending=[True, False])
        .groupby("project_id")
        .head(top_n)
    )
    
    # Validate predictions project by project
    results = []
    for project_id in top_predictions["project_id"].unique():
        # Get expert and algorithm labels for this project
        expert_labels = set(
            high_conf_expert[high_conf_expert["project_id"] == project_id]["label"]
        )
        algo_labels = set(
            top_predictions[top_predictions["project_id"] == project_id]["label"]
        )
        
        # Check for matches
        correct_predictions = expert_labels & algo_labels
        
        results.append({
            "project_id": project_id,
            "num_expert_labels": len(expert_labels),
            "num_correct_predictions": len(correct_predictions),
            "has_correct_prediction": len(correct_predictions) > 0,
            "correct_labels": list(correct_predictions),
        })
    
    results_df = pd.DataFrame(results)
    
    # Log summary statistics
    total_projects = len(results_df)
    projects_with_match = results_df["has_correct_prediction"].sum()
    
    logger.info(
        "Validation results:\n"
        "Total projects: %d\n"
        "Projects with correct prediction: %d (%0.1f%%)\n"
        "Average correct predictions per project: %0.2f",
        total_projects,
        projects_with_match,
        100 * projects_with_match / total_projects,
        results_df["num_correct_predictions"].mean(),
    )
    
    return results_df