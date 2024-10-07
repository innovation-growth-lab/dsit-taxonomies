"""
This is a boilerplate pipeline 'data_annotation_gtr'
generated using Kedro 0.19.6
"""

import logging
import pandas as pd
from keybert import KeyBERT
from joblib import Parallel, delayed
from .utils import (
    get_dbp_annotation,
    get_rake_keywords,
    get_yake_keywords,
    get_keybert_keywords,
)

logger = logging.getLogger(__name__)


def dbp_keywords(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates the text data with DBpedia keywords.

    Args:
        dataframe (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The annotated data.

    """
    dataframe = dataframe.copy()
    dataframe["input_text"] = dataframe["title"] + " " + dataframe["abstract_text"]
    dataframe["dbp_keywords"] = Parallel(n_jobs=8, verbose=10)(
        delayed(get_dbp_annotation)(text) for text in dataframe["input_text"]
    )
    return dataframe[["project_id", "dbp_keywords"]]


def rake_keywords(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates the text data with RAKE keywords.

    Args:
        dataframe (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The annotated data.

    """
    dataframe = dataframe.copy()
    dataframe["input_text"] = dataframe["title"] + " " + dataframe["abstract_text"]
    dataframe["rake_keywords"] = Parallel(n_jobs=8, verbose=10)(
        delayed(get_rake_keywords)(text) for text in dataframe["input_text"]
    )
    return dataframe[["project_id", "rake_keywords"]]


def yake_keywords(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates the text data with YAKE keywords.

    Args:
        dataframe (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The annotated data.

    """
    dataframe = dataframe.copy()
    dataframe["input_text"] = dataframe["title"] + " " + dataframe["abstract_text"]
    dataframe["yake_keywords"] = Parallel(n_jobs=8, verbose=10)(
        delayed(get_yake_keywords)(text) for text in dataframe["input_text"]
    )
    return dataframe[["project_id", "yake_keywords"]]


def keybert_keywords(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates the text data with KeyBERT keywords.

    Args:
        dataframe (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The annotated data.

    """
    kw_extractor = KeyBERT("all-MiniLM-L6-v2")
    dataframe = dataframe.copy()
    dataframe["input_text"] = dataframe["title"] + " " + dataframe["abstract_text"]

    for start in range(0, len(dataframe), 100):
        logger.info(
            "Processing batch %d to %d. This is number: %d / %d",
            start,
            start + 100,
            start // 100,
            len(dataframe) // 100,
        )
        end = start + 100
        batch_df = dataframe.iloc[start:end]
        batch_df.loc[:, "keybert_keywords"] = Parallel(n_jobs=8, verbose=10)(
            delayed(get_keybert_keywords)(text, extractor=kw_extractor)
            for text in batch_df["input_text"]
        )
        yield {f"s{int(start/100)}": batch_df[["project_id", "keybert_keywords"]]}
