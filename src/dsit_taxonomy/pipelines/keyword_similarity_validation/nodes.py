"""
This is a boilerplate pipeline 'keyword_similarity_validation'
generated using Kedro 0.19.10
"""

import ast
import logging
from typing import Generator
import pandas as pd
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
