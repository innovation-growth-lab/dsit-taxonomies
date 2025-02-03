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


SYSTEM_PROMPT = """
You are an expert research analyst with deep domain knowledge across multiple scientific disciplines. Your task is to assign taxonomy labels to a research project, based on its abstract, technical description, and potential impact. You will be provided with:
- The project details (title, abstract, technical summary, potential impact).
- A retrieval-augmented (RAG) set of taxonomy labels, retrieved based on keyword and document similarity: {context}

### Your Task:
You must evaluate the retrieved taxonomy labels and classify them into three categories:
1. True Positive Labels → These labels correctly describe the project's field.
2. False Positive Labels → These labels appear relevant due to keyword or semantic overlap but are actually incorrect.
3. Likelihood Assignment → Assign a likelihood of "high", "medium", or "low" to each label based on how well it matches the project.

### Output Format:
Your final response should be a JSON-readable list of dictionaries, following this structure:

[
  {{"label": "Physics > Quantum Mechanics > Quantum Optics", "likelihood": "high", "positive": "true"}},
  {{"label": "Engineering > Materials Science > Nanophotonics", "likelihood": "medium", "positive": "true"}},
  {{"label": "Computer Science > Artificial Intelligence > Machine Learning", "likelihood": "low", "positive": "false"}}
]

- 0-6 true positive labels should be selected.
- 0-3 false positive labels should be included (if applicable).
Ensure the response is structured as JSON (not a sentence, no explanation).
False positive labels should only be included if they could be incorrectly assigned by an automated approach.

Guidelines for Label Selection:
- High likelihood: The label is a direct and precise match for the project's field.
- Medium likelihood: The label is relevant but secondary, or the project partially overlaps with this field.
- Low likelihood: The label has some minor connection but is not a correct categorization.
- False positives: These are labels that seem plausible based on keywords but are actually incorrect.
"""

QUESTION = """
How would you classify the following project based on the provided taxonomy labels?

{abstract}

Recall: 
- 0-6 true positive labels should be selected. Be critical in the "likelihood", only set "high" if confident.
- 0-3 false positive labels should be included (if applicable).
Ensure the response is structured as JSON (not a sentence, no explanation).
False positive labels should only be included if they could be incorrectly assigned by an automated approach.

Guidelines for Label Selection:
- High likelihood: The label is a direct and precise match for the project's field.
- Medium likelihood: The label is relevant but secondary, or the project partially overlaps with this field.
- Low likelihood: The label has some minor connection but is not a correct categorization.
- False positives: These are labels that seem plausible based on keywords but are actually incorrect.

"""

logger = logging.getLogger(__name__)


def select_sample_projects(data: pd.DataFrame) -> pd.DataFrame:
    """Select a sample of projects for expert labeling."""
    return data.sample(150, random_state=42)


def get_expert_labels(taxonomy: pd.DataFrame, data: pd.DataFrame) -> Generator:
    """
    Get expert labels for a sample of projects using a retrieval-augmented generative model.

    Args:
        taxonomy: The taxonomy dataframe.
        data: The sample of projects to label.

    Yields:
        A dictionary containing the project ID and the expert labels.
    """
    logger.info("Getting expert labels")

    model = ChatOpenAI(model="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = InMemoryVectorStore(embeddings)

    loader = DataFrameLoader(taxonomy, page_content_column="label")
    _ = vector_store.add_documents(loader.load())
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{input}")]
    )

    question_answer_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
        question = QUESTION.format(abstract=row["text"])
        response = rag_chain.invoke({"input": question})
        try:
            response_json = ast.literal_eval(response["answer"])
            yield {row["project_id"]: response_json}
        except (ValueError, SyntaxError) as e:
            logger.error("Failed to decode JSON response: %s", e)
