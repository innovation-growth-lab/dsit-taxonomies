"""
This module contains nodes for extracting keywords from research project texts
using multiple extraction methods.

The nodes implement:
- DBpedia Spotlight for entity linking
- RAKE for statistical keyword extraction
- YAKE for unsupervised keyword extraction
- KeyBERT for transformer-based extraction
- Utilities for result aggregation and processing
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
    Extract keywords using DBpedia Spotlight entity linking.

    This function:
    1. Combines text fields from each project
    2. Links text entities to DBpedia concepts
    3. Filters and processes the linked entities
    4. Handles incremental processing with checkpointing

    Args:
        dataframe: Input data containing:
            - project_id: Unique project identifier
            - title: Project title
            - abstract_text: Main abstract
            - tech_abstract_text: Technical abstract
            - potential_impact: Impact statement
        processed_projects: Previously processed results for incremental updates

    Returns:
        DataFrame containing:
            - project_id: Project identifier
            - dbp_keywords: List of extracted DBpedia concepts
    """
    dataframe = _filter_processed_projects(dataframe, processed_projects)
    dataframe["input_text"] = (
        dataframe["title"].fillna("")
        + ". "
        + dataframe["abstract_text"].fillna("")
        + ". "
        + dataframe["tech_abstract_text"].fillna("")
        + ". "
        + dataframe["potential_impact"].fillna("")
    )
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
    Extract keywords using the RAKE algorithm.

    This function:
    1. Combines title and abstract text
    2. Applies RAKE to identify key phrases
    3. Scores and ranks the extracted phrases
    4. Handles incremental processing

    Args:
        dataframe: Input data with text fields
        processed_projects: Previously processed results

    Returns:
        DataFrame with project_id and rake_keywords columns
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
    Extract keywords using the YAKE algorithm.

    This function:
    1. Combines title and abstract text
    2. Applies YAKE for keyword extraction
    3. Processes results with custom parameters
    4. Handles incremental updates

    Args:
        dataframe: Input data with text fields
        processed_projects: Previously processed results

    Returns:
        DataFrame with project_id and yake_keywords columns
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
    Extract keywords using KeyBERT transformer model.

    This function:
    1. Initialises KeyBERT with specified model
    2. Processes texts in batches for memory efficiency
    3. Uses semantic similarity for keyword ranking
    4. Yields results incrementally with timestamps

    Args:
        dataframe: Input data with text fields
        processed_projects: Previously processed results

    Yields:
        Dictionary mapping partition ID to DataFrame with:
            - project_id: Project identifier
            - keybert_keywords: Extracted keywords
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
    Combine partitioned KeyBERT results into a single dataset.

    This function:
    1. Loads partitioned results in parallel
    2. Concatenates all results
    3. Removes duplicate project entries
    4. Handles errors in individual partitions

    Args:
        partitioned_dataset: Dictionary mapping partition IDs to datasets

    Returns:
        DataFrame containing combined and deduplicated results
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
    Filter out already processed projects for incremental updates.

    Args:
        gtr_data: Complete dataset to process
        processed_projects: Previously processed results

    Returns:
        DataFrame containing only unprocessed projects
    """
    if processed_projects.empty:
        return gtr_data

    filtered_df = gtr_data[
        ~gtr_data["project_id"].isin(processed_projects["project_id"])
    ]
    return filtered_df
