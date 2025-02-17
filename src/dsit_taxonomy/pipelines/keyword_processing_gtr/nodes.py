"""
This module contains nodes for processing and embedding keywords extracted
from research projects.

The nodes handle:
- Aggregating keywords from multiple extraction methods
- Filtering keywords based on extractor agreement
- Generating semantic embeddings for keywords
"""

import logging
import uuid
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger(__name__)


def aggregate_keyword_annotators(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate keywords from multiple extractors and count their appearances.

    This function:
    1. Processes keywords from each extractor
    2. Counts how many extractors found each keyword
    3. Maps keywords to their source projects
    4. Filters out keywords found by only one extractor

    Args:
        *dataframes: Variable number of DataFrames, each containing:
            - project_id: ID of the research project
            - {extractor}_keywords: List of keywords from each extractor

    Returns:
        DataFrame containing:
            - keyword: The unique keyword
            - num_annotators: Number of extractors that found this keyword
            - project_ids: List of projects where the keyword appears
            - uuid: Unique identifier for the keyword
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


def generate_keyword_embeddings(keyword_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate semantic embeddings for keywords using sentence transformers.

    This function converts keywords into dense vector representations that
    can be used for similarity matching.

    Args:
        keyword_dataframe: DataFrame containing:
            - keyword: Text of the keyword to embed
            - Other metadata columns (ignored)

    Returns:
        DataFrame containing:
            - keyword: Original keyword text
            - embedding: Dense vector representation as float32 array
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


def _preprocess_keywords(keywords: pd.Series) -> pd.Series:
    """
    Preprocess keywords for consistency.

    Args:
        keywords: Series of keyword strings

    Returns:
        Series of preprocessed keywords (lowercase, stripped)
    """
    return keywords.str.lower().str.strip()
