"""
This is a boilerplate pipeline 'data_annotation_gtr'
generated using Kedro 0.19.6
"""

import logging
from typing import Generator, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from keybert import KeyBERT
from kedro.io import AbstractDataset
from joblib import Parallel, delayed
from .utils import (
    get_dbp_annotation,
    get_rake_keywords,
    get_yake_keywords,
    get_keybert_keywords,
)

logger = logging.getLogger(__name__)


def dbp_keywords(
    dataframe: pd.DataFrame, processed_projects: pd.DataFrame
) -> pd.DataFrame:
    """
    Annotates the text data with DBpedia keywords.

    Args:
        dataframe (pd.DataFrame): The input GtR data.
        processed_projects (pd.DataFrame): The projects that have already been processed.

    Returns:
        pd.DataFrame: The annotated data.

    """
    dataframe = _filter_processed_projects(dataframe, processed_projects)
    dataframe["input_text"] = dataframe["title"] + " " + dataframe["abstract_text"]
    dataframe["dbp_keywords"] = Parallel(n_jobs=8, verbose=10)(
        delayed(get_dbp_annotation)(text) for text in dataframe["input_text"]
    )

    # concatenate the processed projects with the new ones
    output_dataframe = pd.concat(
        [processed_projects, dataframe[["project_id", "dbp_keywords"]]],
        ignore_index=True,
    )

    # drop duplicates, keeping the first occurrence
    output_dataframe.drop_duplicates(subset="project_id", keep="first", inplace=True)

    return output_dataframe


def rake_keywords(
    dataframe: pd.DataFrame, processed_projects: pd.DataFrame
) -> pd.DataFrame:
    """
    Annotates the text data with RAKE keywords.

    Args:
        dataframe (pd.DataFrame): The input GtR data.
        processed_projects (pd.DataFrame): The projects that have already been processed.

    Returns:
        pd.DataFrame: The annotated data.

    """
    dataframe = _filter_processed_projects(dataframe, processed_projects)
    dataframe["input_text"] = dataframe["title"] + " " + dataframe["abstract_text"]
    dataframe["rake_keywords"] = Parallel(n_jobs=8, verbose=10)(
        delayed(get_rake_keywords)(text) for text in dataframe["input_text"]
    )

    # concatenate the processed projects with the new ones
    output_dataframe = pd.concat(
        [processed_projects, dataframe[["project_id", "rake_keywords"]]],
        ignore_index=True,
    )

    # drop duplicates, keeping the first occurrence
    output_dataframe.drop_duplicates(subset="project_id", keep="first", inplace=True)

    return output_dataframe


def yake_keywords(
    dataframe: pd.DataFrame, processed_projects: pd.DataFrame
) -> pd.DataFrame:
    """
    Annotates the text data with YAKE keywords.

    Args:
        dataframe (pd.DataFrame): The input GtR data.
        processed_projects (pd.DataFrame): The projects that have already been processed.

    Returns:
        pd.DataFrame: The annotated data.

    """
    dataframe = _filter_processed_projects(dataframe, processed_projects)
    dataframe["input_text"] = dataframe["title"] + " " + dataframe["abstract_text"]
    dataframe["yake_keywords"] = Parallel(n_jobs=8, verbose=10)(
        delayed(get_yake_keywords)(text) for text in dataframe["input_text"]
    )

    # concatenate the processed projects with the new ones
    output_dataframe = pd.concat(
        [processed_projects, dataframe[["project_id", "yake_keywords"]]],
        ignore_index=True,
    )

    # drop duplicates, keeping the first occurrence
    output_dataframe.drop_duplicates(subset="project_id", keep="first", inplace=True)

    return output_dataframe


def keybert_keywords(
    dataframe: pd.DataFrame, processed_projects: pd.DataFrame
) -> Generator[pd.DataFrame, None, None]:
    """
    Annotates the text data with KeyBERT keywords.

    Args:
        dataframe (pd.DataFrame): The input GtR data.
        processed_projects (pd.DataFrame): The projects that have already been processed.

    Returns:
        pd.DataFrame: The annotated data.

    """
    kw_extractor = KeyBERT("all-MiniLM-L6-v2")
    dataframe = _filter_processed_projects(dataframe, processed_projects)
    dataframe["input_text"] = dataframe["title"] + " " + dataframe["abstract_text"]
    day_timestamp = str(datetime.now().strftime("%y%m%d"))

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
        yield {
            f"{day_timestamp}/s{int(start/100)}": batch_df[
                ["project_id", "keybert_keywords"]
            ]
        }


def concatenate_partitions(
    partitioned_dataset: Dict[str, AbstractDataset]
) -> pd.DataFrame:
    """
    Concatenate the partitions from the given inputs.

    Args:
        partitioned_dataset (Dict[str, AbstractDataset]): The partitioned dataset.

    Returns:
        pd.DataFrame: The concatenated dataset.
    """

    def load_dataset(dataset: AbstractDataset) -> pd.DataFrame:
        return dataset()

    datasets = []
    with ThreadPoolExecutor() as executor:
        future_to_dataset = {
            executor.submit(load_dataset, dataset): i
            for i, dataset in enumerate(partitioned_dataset.values())
        }
        for future in as_completed(future_to_dataset):
            i = future_to_dataset[future]
            try:
                data = future.result()
                logger.info(
                    "Concatenating partition %d / %d", i + 1, len(partitioned_dataset)
                )
                datasets.append(data)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Partition %d generated an exception: %s", i + 1, exc)

    concat_data = pd.concat(datasets, ignore_index=True)
    concat_data.drop_duplicates(subset="project_id", keep="first", inplace=True)

    return concat_data


def _filter_processed_projects(
    gtr_data: pd.DataFrame, processed_projects: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter out projects that have already been processed.

    Args:
        gtr_data (pd.DataFrame): The input GtR data.
        processed_projects (pd.DataFrame): The projects that have already been processed.

    Returns:
        pd.DataFrame: The filtered data.

    """
    if processed_projects.empty:
        return gtr_data

    filtered_df = gtr_data[
        ~gtr_data["project_id"].isin(processed_projects["project_id"])
    ]
    return filtered_df
