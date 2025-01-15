import logging
import uuid
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger(__name__)


def aggregate_keyword_annotators(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the number of distinct annotator classes each keyword appears in.

    Args:
        dataframes (pd.DataFrame): The dataframes to process.

    Returns:
        pd.DataFrame: A dataframe with three columns:
            - 'label': The unique keyword.
            - 'num_annotators': The number of distinct annotator classes the keyword appears in.
            - 'project_ids': A list of project_ids each keyword appears in.
    """
    keyword_to_annotators = {}
    keyword_to_projects = {}

    for df in dataframes:
        logger.info("Processing dataframe with columns: %s", df.columns)
        keyword_column = next(col for col in df.columns if "_keywords" in col)
        annotator_class = keyword_column.split("_")[0]

        # explode the keywords and drop NaN values
        exploded = df.explode(keyword_column).dropna(subset=[keyword_column])

        # preprocess the keywords
        exploded[keyword_column] = _preprocess_keywords(exploded[keyword_column])

        # aggregate the keywords with annotator classes and project_ids
        for keyword, project_id in zip(
            exploded[keyword_column], exploded["project_id"]
        ):
            keyword_to_annotators.setdefault(keyword, set()).add(annotator_class)
            keyword_to_projects.setdefault(keyword, set()).add(project_id)

    # prepare the output dataframe
    output_df = pd.DataFrame(
        {
            "keyword": keyword_to_annotators.keys(),
            "num_annotators": [
                len(annotators) for annotators in keyword_to_annotators.values()
            ],
            "project_ids": [
                list(project_ids) for project_ids in keyword_to_projects.values()
            ],
        }
    )

    # filter out keywords that appear in only one annotator class
    output_df = output_df[output_df["num_annotators"] > 1].reset_index(drop=True)

    # add uuids
    output_df["uuid"] = output_df["keyword"].apply(
        lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, x))
    )

    # sort the dataframe by num_annotators in descending order, then by keyword alphabetically
    return output_df.sort_values(
        by=["num_annotators", "keyword"], ascending=[False, True]
    )


def _preprocess_keywords(keywords: pd.Series) -> pd.Series:
    """Preprocess the keywords by lowercasing and removing trailing spaces."""
    return keywords.str.lower().str.strip()


def generate_keyword_embeddings(keyword_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate embeddings for each keyword in the dataframe.

    Args:
        keyword_dataframe: The dataframe containing the keywords.

    Returns:
        pd.DataFrame: A dataframe with two columns:
            - 'keyword': The unique keyword.
            - 'embedding': The corresponding embedding as a list of float32 values.
    """

    # generate embeddings for the keywords
    embeddings = model.encode(
        keyword_dataframe["keyword"].tolist(),
        show_progress_bar=True,
        convert_to_tensor=False,
    )

    # convert embeddings to float32
    embeddings = np.array(embeddings, dtype=np.float32)

    logger.info("Generated embeddings for %s keywords", len(embeddings))
    keyword_dataframe["embedding"] = embeddings.tolist()

    # select only the 'keyword' and 'embedding' columns
    result_df = keyword_dataframe[["keyword", "embedding"]]

    return result_df


def _preprocess_keywords(keywords):
    # Assuming this function is defined elsewhere in your code
    return keywords
