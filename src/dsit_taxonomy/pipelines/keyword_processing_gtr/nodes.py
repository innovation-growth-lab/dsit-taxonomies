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
from typing import Generator, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from kedro.io import AbstractDataset
from joblib import Parallel, delayed
from .utils import (
    get_dbp_annotation,
    get_rake_keywords,
    get_yake_keywords,
    get_keybert_keywords_standalone,
)
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)


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
    dataframe["dbp_keywords"] = Parallel(
        n_jobs=8,
        verbose=10,
        timeout=180,  # 3 minutes timeout for API calls
        max_nbytes=None,  # Disable memory limit
    )(delayed(get_dbp_annotation)(text) for text in dataframe["input_text"])

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
    dataframe["rake_keywords"] = Parallel(
        n_jobs=8,
        verbose=10,
        timeout=120,  # 2 minutes timeout for RAKE processing
        max_nbytes=None,  # Disable memory limit
    )(delayed(get_rake_keywords)(text) for text in dataframe["input_text"])

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
    dataframe["yake_keywords"] = Parallel(
        n_jobs=8,
        verbose=10,
        timeout=120,  # 2 minutes timeout for YAKE processing
        max_nbytes=None,  # Disable memory limit
    )(delayed(get_yake_keywords)(text) for text in dataframe["input_text"])

    # concatenate the processed projects with the new ones
    output_dataframe = pd.concat(
        [processed_projects, dataframe[["project_id", "yake_keywords"]]],
        ignore_index=True,
    )

    # drop duplicates, keeping the first occurrence
    output_dataframe.drop_duplicates(subset="project_id", keep="first", inplace=True)

    return output_dataframe


def keybert_keywords(
    dataframe: pd.DataFrame, processed_projects: pd.DataFrame, n_jobs: int = 8
) -> Generator[pd.DataFrame, None, None]:
    """
    Extract keywords using KeyBERT transformer model.

    This function:
    1. Uses multiprocessing.Pool with spawn method to handle CUDA properly
    2. Processes texts in batches for memory efficiency
    3. Uses semantic similarity for keyword ranking
    4. Yields results incrementally with timestamps

    Args:
        dataframe: Input data with text fields
        processed_projects: Previously processed results
        n_jobs: Number of parallel jobs for keyword extraction

    Yields:
        Dictionary mapping partition ID to DataFrame with:
            - project_id: Project identifier
            - keybert_keywords: Extracted keywords
    """

    
    dataframe = _filter_processed_projects(dataframe, processed_projects)
    dataframe = dataframe.copy()
    dataframe["input_text"] = dataframe["title"] + " " + dataframe["abstract_text"]
    day_timestamp = str(datetime.now().strftime("%y%m%d"))

    # Process in batches to manage memory
    batch_size = 100
    for start in range(0, len(dataframe), batch_size):
        logger.info(
            "Processing batch %d to %d. This is number: %d / %d",
            start,
            start + batch_size,
            start // batch_size,
            len(dataframe) // batch_size,
        )
        end = start + batch_size
        batch_df = dataframe.iloc[start:end]
        batch_df_copy = batch_df.copy()
        
        # Use multiprocessing.Pool with spawn method to handle CUDA properly
        try:
            # Set start method to spawn to avoid CUDA issues
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
            
            with mp.Pool(processes=n_jobs) as pool:
                # Process texts in parallel
                results = pool.map(
                    get_keybert_keywords_standalone, 
                    batch_df["input_text"].tolist(),
                    chunksize=max(1, len(batch_df) // n_jobs)
                )
                batch_df_copy.loc[:, "keybert_keywords"] = results
                
        except Exception as e:
            logger.error("Multiprocessing failed: %s", e)
            # Fallback to sequential processing if multiprocessing fails
            logger.info("Falling back to sequential processing")
            results = []
            for text in batch_df["input_text"]:
                try:
                    keywords = get_keybert_keywords_standalone(text)
                    results.append(keywords)
                except Exception as text_error:
                    logger.error("Error processing text: %s", text_error)
                    results.append([])
            batch_df_copy.loc[:, "keybert_keywords"] = results
        
        yield {
            f"{day_timestamp}/s{int(start/batch_size)}": batch_df_copy[
                ["project_id", "keybert_keywords"]
            ]
        }


def concatenate_partitions(
    partitioned_dataset: Dict[str, AbstractDataset], n_jobs: int = 8
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
        n_jobs: Number of parallel jobs for loading datasets. Defaults to 8.

    Returns:
        DataFrame containing combined and deduplicated results
    """

    def load_dataset(dataset: AbstractDataset) -> pd.DataFrame:
        return dataset()

    datasets = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
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


def _preprocess_keywords(keywords: pd.Series) -> pd.Series:
    """
    Preprocess keywords for consistency.

    Args:
        keywords: Series of keyword strings

    Returns:
        Series of preprocessed keywords (lowercase, stripped)
    """
    return keywords.str.lower().str.strip()
