import logging
import uuid
from typing import List, Dict
import lancedb
import pandas as pd
from scipy.stats import entropy
from spacy.lang.en import English
import joblib
from functools import partial
import numpy as np

logger = logging.getLogger(__name__)


def compute_similarities_and_entropy(
    taxonomy: lancedb,
    documents: lancedb,
    batch_size: int = 1000,
    top_n: int = 10,
    number_returns: int = 1000,
) -> pd.DataFrame:
    """
    Compute similarity scores and Shannon entropy for a set of documents against a
    taxonomy of labels.

    Args:
        taxonomy: LanceDB table for taxonomy embeddings.
        documents: LanceDB table for document embeddings.
        batch_size: Number of documents to process in each batch.
        top_n: Number of top matches to retain for final output.
        number_returns: Number of matches to use for Shannon entropy calculation.

    Returns:
        pd.DataFrame: DataFrame with columns "documents_id", "taxonomy_label_id",
        "similarity_score", and "shannon_entropy".
    """
    # convert LanceDB table to a list of dicts for batch processing
    documents_dict = documents.to_pandas().to_dict(orient="records")

    # Divide documents into batches
    document_batches = [
        documents_dict[i : i + batch_size]
        for i in range(0, len(documents_dict), batch_size)
    ]

    # Process each batch sequentially
    results = []
    for i, batch in enumerate(document_batches):
        logger.info("Processing batch %d / %d", i + 1, len(document_batches))
        batch_results = _search_batch(
            batch, taxonomy, top_n=top_n, number_returns=number_returns
        )
        results.append(batch_results)

    # Concatenate results into a single DataFrame
    logger.info("Flattening results")
    return pd.concat(results, ignore_index=True)


def document_preprocessing(documents: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame of documents.

    Args:
        documents: DataFrame containing the GtR documents.

    Returns:
        pd.DataFrame: DataFrame containing the preprocessed text column.
    """
    nlp = English()
    nlp.add_pipe("sentencizer")

    # create a unique column combining all text columns
    documents["text"] = documents.apply(
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

    # split documents into sentences
    documents["text"] = documents["text"].apply(_split_sentences, nlp=nlp)

    # explode sentences into separate rows
    documents = documents.explode("text")

    # add uuids
    documents["uuid"] = documents["text"].apply(
        lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, x))
    )

    # [HACK] drop duplicate rows to avoid non-informational matches
    # This of course has the downside that it may remove some valid matches.
    documents = documents.drop_duplicates(subset=["text"], keep=False)

    return documents[["project_id", "uuid", "text"]]


def aggregate_sentence_matches(
    sentences: pd.DataFrame,
    sentence_matches: pd.DataFrame,
    min_score_quantile: float,
    relative_score_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Aggregate sentence matches with length-aware scoring.

    Args:
        sentences: DataFrame with sentence metadata
        sentence_matches: Raw similarity matches
        min_score_quantile: Global minimum score quantile threshold
        relative_score_threshold: Keep scores within this fraction of project's max score

    Returns:
        DataFrame with aggregated and filtered similarity scores
    """
    logger.info("Aggregating document matches to project level")

    # Merge documents with matches
    merged_sentences = pd.merge(
        sentences[["project_id", "uuid"]],
        sentence_matches,
        left_on="uuid",
        right_on="document_id",
        how="right",
    )

    # Compute project-level aggregations
    project_scores = (
        merged_sentences.groupby(["project_id", "taxonomy_label_id"])
        .agg(
            {
                "similarity_score": [
                    "sum",  # Raw sum for high-frequency signals
                    "mean",  # Average strength
                    "max",  # Strongest single match
                    "count",  # Number of matching sentences
                ]
            }
        )
        .reset_index()
    )

    # Flatten column names
    project_scores.columns = [
        "project_id",
        "taxonomy_label_id",
        "score_sum",
        "score_mean",
        "score_max",
        "match_count",
    ]

    # Get sentence counts per project for normalisation
    project_lengths = merged_sentences.groupby("project_id")["uuid"].nunique()

    # Compute normalised scores
    project_scores = project_scores.merge(
        project_lengths.reset_index(name="n_sentences"), on="project_id"
    )

    # Compute initial combined scores
    project_scores["initial_score"] = (
        # normalised frequency component
        (project_scores["match_count"] / project_scores["n_sentences"])
        *
        # Average strength component
        project_scores["score_mean"]
        *
        # Boost factor for very strong individual matches
        (1 + np.log1p(project_scores["score_max"]))
    )

    # Apply global quantile threshold
    score_threshold = project_scores["initial_score"].quantile(min_score_quantile)
    filtered_scores = project_scores[project_scores["initial_score"] >= score_threshold]

    # Apply relative threshold within each project
    max_scores = filtered_scores.groupby("project_id")["initial_score"].transform("max")
    final_scores = filtered_scores[
        filtered_scores["initial_score"] >= relative_score_threshold * max_scores
    ]

    # Log distribution of topics per project
    topics_per_project = final_scores.groupby("project_id").size()
    logger.info(
        "Score and topic distribution:\n"
        "Global score threshold: %0.3f\n"
        "Topics per project:\n%s",
        score_threshold,
        topics_per_project.describe().to_string(),
    )

    final_scores = final_scores.rename(columns={"initial_score": "similarity_score"})

    return final_scores


def combine_sentence_and_keyword_scores(
    sentence_scores: pd.DataFrame,
    keyword_scores: pd.DataFrame,
    keyword_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine document-based and keyword-based scores with configurable weights.
    """
    logger.info("Combining document and keyword scores")

    # Map keywords to projects
    project_keywords = (
        keyword_data[["project_ids", "uuid"]]
        .explode("project_ids")
        .rename(columns={"project_ids": "project_id", "uuid": "keyword_id"})
    ).merge(
        keyword_scores[
            ["document_id", "taxonomy_label_id", "similarity_score", "shannon_entropy"]
        ].rename(columns={"document_id": "keyword_id"}),
        on="keyword_id",
        how="left",
    )

    # Merge keyword and sentence scores
    combined_scores = pd.merge(
        project_keywords,
        sentence_scores,
        on=["project_id", "taxonomy_label_id"],
        how="inner",
        suffixes=("_key", "_sent"),
    )

    return combined_scores


def add_metadata(
    combined_scores: pd.DataFrame, keyword_data: pd.DataFrame, taxonomy: pd.DataFrame
) -> pd.DataFrame:
    """
    Add metadata to the combined scores.
    """
    logger.info("Adding metadata to combined scores")

    # Add metadata
    final_scores = combined_scores.merge(
        keyword_data[["uuid", "keyword"]],
        left_on="keyword_id",
        right_on="uuid",
        how="left",
    ).merge(
        taxonomy[["uuid", "label"]],
        left_on="taxonomy_label_id",
        right_on="uuid",
        how="left",
    )

    logger.info(
        "Score distributions:\n"
        "Sentence scores: %s\n"
        "Keyword scores: %s\n"
        "Combined scores: %s",
        combined_scores["similarity_score_sent"].describe(),
        combined_scores["similarity_score_key"].describe(),
        combined_scores["relevance_score"].describe(),
    )

    return final_scores[
        [
            "project_id",
            "keyword_id",
            "taxonomy_label_id",
            "keyword",
            "label",
            "similarity_score_sent",
            "similarity_score_key",
            "relevance_score",
            "shannon_entropy",
        ]
    ]


def aggregate_scores_to_labels(
    scores: pd.DataFrame,
    sentence_weight: float,
    keyword_weight: float,
    similarity_quantile_threshold: float,
    global_q2_threshold: float = 0.50,
    global_q3_threshold: float = 0.75,
    local_q2_threshold: float = 0.50,
    local_q3_threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Aggregate detailed scores to final project-label level summaries.

    Args:
        scores: Raw scores DataFrame
        sentence_weight: Weight for sentence-based scores
        keyword_weight: Weight for keyword-based scores
        similarity_quantile_threshold: Threshold for filtering low scores
        global_q2_threshold: Global threshold for medium confidence
        global_q3_threshold: Global threshold for high confidence
        local_q2_threshold: Project-level threshold for medium confidence
        local_q3_threshold: Project-level threshold for high confidence
    """
    logger.info(
        "Aggregating scores with parameters:\n"
        "Weights - Sentence: %0.2f, Keyword: %0.2f\n"
        "Score threshold: %0.2f\n"
        "Global quantiles - Q2: %0.2f, Q3: %0.2f\n"
        "Local quantiles - Q2: %0.2f, Q3: %0.2f",
        sentence_weight,
        keyword_weight,
        similarity_quantile_threshold,
        global_q2_threshold,
        global_q3_threshold,
        local_q2_threshold,
        local_q3_threshold,
    )

    # Apply weights
    logger.info(
        "Applying weights - Document: %0.1f, Keyword: %0.1f",
        sentence_weight,
        keyword_weight,
    )
    scores["relevance_score"] = (
        (sentence_weight * scores["similarity_score_sent"])
        * (keyword_weight * scores["similarity_score_key"])
    ) ** 2

    # Filter low keyword scores
    scores = scores[
        scores["relevance_score"]
        >= scores["relevance_score"].quantile(similarity_quantile_threshold)
    ]

    logger.info("Aggregating final scores to project-label level")

    aggregated = (
        scores.groupby(["project_id", "taxonomy_label_id", "label"], as_index=False)
        .agg(
            {
                "relevance_score": "sum",
                "similarity_score_sent": "mean",
                "similarity_score_key": "mean",
                "shannon_entropy": "mean",
                "keyword_id": "nunique",
            }
        )
        .rename(columns={"keyword_id": "num_keywords"})
    )

    logger.info(
        "Aggregation summary:\n"
        "Total projects: %d\n"
        "Average labels per project: %0.1f",
        len(scores["project_id"].unique()),
        len(aggregated) / len(aggregated["project_id"].unique()),
    )

    # Add confidence bins with custom thresholds
    aggregated = _assign_confidence_bins(
        aggregated,
        global_q2_threshold,
        global_q3_threshold,
        local_q2_threshold,
        local_q3_threshold,
    )

    return aggregated


def _normalise_within_projects(group: pd.DataFrame) -> pd.DataFrame:
    """Normalise scores within each project using min-max scaling."""
    if len(group) == 1:
        group["similarity_score"] = 1.0
    else:
        max_score = group["similarity_score"].max()
        min_score = group["similarity_score"].min()
        if max_score == min_score:
            group["similarity_score"] = 1.0
        else:
            group["similarity_score"] = (group["similarity_score"] - min_score) / (
                max_score - min_score
            )
    return group


def _search_batch(
    document_batch: List[Dict[str, str]],
    taxonomy_table: lancedb,
    top_n=10,
    number_returns=5000,
) -> pd.DataFrame:
    """
    Perform similarity search for a batch of strings, computing entropy over a larger number
    of matches (number_returns) but only retaining the top N matches for output. It also
    computes the Shannon entropy over the similarity scores of the expanded matches.

    Args:
        document_batch (List[Dict[str, str]]): List of document embeddings.
        taxonomy_table (lancedb): LanceDB table for taxonomy embeddings.
        top_n (int): Number of top matches to retain for final output.
        number_returns (int): Number of matches to use for Shannon entropy calculation.
    """
    results = []

    for document in document_batch:
        embedding = document["vector"]
        document_id = document["id"]

        # perform similarity search, retrieving a larger number of matches for entropy
        expanded_labels = (
            taxonomy_table.search(embedding)
            .metric("cosine")
            .limit(number_returns)
            .to_pandas()
        )

        # normalise similarity as 1 - 1/2*_distance.
        # See https://lancedb.github.io/lancedb/python/python/#lancedb.index.IvfPq
        expanded_labels["similarity_score"] = (2 - expanded_labels["_distance"]) / 2

        # compute Shannon entropy over expanded matches
        entropy_value = _compute_shannon_entropy(
            expanded_labels["similarity_score"].values
        )

        # reduce to top N matches for final output
        top_matches = expanded_labels.nlargest(top_n, "similarity_score")

        # Append results as a DataFrame
        results.append(
            pd.DataFrame(
                {
                    "document_id": document_id,
                    "taxonomy_label_id": top_matches["id"].values,
                    "similarity_score": top_matches["similarity_score"].values,
                    "shannon_entropy": entropy_value,
                }
            )
        )

    # concatenate all DataFrames into a single DataFrame
    return pd.concat(results, ignore_index=True)


def _compute_shannon_entropy(similarity_scores):
    """Compute the Shannon entropy for a given list of similarity scores."""
    return entropy(similarity_scores, base=2)


def _split_sentences(document: str, nlp: English) -> List[str]:
    """Split a document into sentences."""
    doc = nlp(document)
    return [sent.text for sent in doc.sents]


def _assign_confidence_bins(
    df: pd.DataFrame,
    global_q2_threshold: float,
    global_q3_threshold: float,
    local_q2_threshold: float,
    local_q3_threshold: float,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """
    Assign confidence bins to taxonomy labels based on global and local binning strategies.
    Uses parallel processing for faster computation.

    Args:
        df: Input DataFrame with relevance scores
        global_q2_threshold: Global threshold for medium confidence
        global_q3_threshold: Global threshold for high confidence
        local_q2_threshold: Project-level threshold for medium confidence
        local_q3_threshold: Project-level threshold for high confidence
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    logger.info("Assigning confidence bins to taxonomy labels")

    # 1. Global binning (using provided thresholds)
    q2 = df["relevance_score"].quantile(global_q2_threshold)
    q3 = df["relevance_score"].quantile(global_q3_threshold)

    def _assign_global_bin(score):
        if score > q3:
            return "high"
        elif score > q2:
            return "medium"
        return "low"

    df["global_bin"] = df["relevance_score"].apply(_assign_global_bin)

    # 2. Local binning - prepare data for parallel processing
    project_groups = [group for _, group in df.groupby("project_id")]

    # Create partial function with fixed parameters
    _assign_local_bins_partial = partial(
        _process_project_group,
        local_q2_threshold=local_q2_threshold,
        local_q3_threshold=local_q3_threshold,
    )

    # Process groups in parallel
    logger.info("Processing %d projects in parallel", len(project_groups))
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_assign_local_bins_partial)(group) for group in project_groups
    )

    # Combine results
    local_bins = pd.concat(results)

    df["local_bin"] = local_bins

    # 3. Final bin (minimum of global and local)
    def _get_min_bin(row):
        bin_order = {"high": 3, "medium": 2, "low": 1}
        min_val = min(bin_order[row["global_bin"]], bin_order[row["local_bin"]])
        return {3: "high", 2: "medium", 1: "low"}[min_val]

    df["final_bin"] = df.apply(_get_min_bin, axis=1)

    # Log summary statistics
    logger.info(
        "Binning summary:\n"
        "Global thresholds - Q2: %0.3f, Q3: %0.3f\n"
        "Global distribution - high: %0.1f%%, medium: %0.1f%%, low: %0.1f%%\n"
        "Local distribution - high: %0.1f%%, medium: %0.1f%%, low: %0.1f%%\n"
        "Final distribution - high: %0.1f%%, medium: %0.1f%%, low: %0.1f%%",
        q2,
        q3,
        100 * (df["global_bin"] == "high").mean(),
        100 * (df["global_bin"] == "medium").mean(),
        100 * (df["global_bin"] == "low").mean(),
        100 * (df["local_bin"] == "high").mean(),
        100 * (df["local_bin"] == "medium").mean(),
        100 * (df["local_bin"] == "low").mean(),
        100 * (df["final_bin"] == "high").mean(),
        100 * (df["final_bin"] == "medium").mean(),
        100 * (df["final_bin"] == "low").mean(),
    )

    return df


def _process_project_group(
    group: pd.DataFrame,
    local_q2_threshold: float,
    local_q3_threshold: float,
) -> pd.Series:
    """Process a single project group for local binning."""
    n_labels = len(group)

    # Handle edge cases for small groups
    if n_labels == 1:
        return pd.Series(["high"], index=group.index)
    elif n_labels == 2:
        sorted_idx = group["relevance_score"].sort_values(ascending=False).index
        return pd.Series(["high", "medium"], index=sorted_idx)

    # Sort scores in descending order
    sorted_scores = group["relevance_score"].sort_values(ascending=False)

    # Compute project-specific quantiles
    q2_local = sorted_scores.quantile(local_q2_threshold)
    q3_local = sorted_scores.quantile(local_q3_threshold)

    def _assign_quantile_bin(score):
        if score > q3_local:
            return "high"
        elif score > q2_local:
            return "medium"
        return "low"

    quantile_bins = pd.Series(
        [_assign_quantile_bin(score) for score in sorted_scores],
        index=sorted_scores.index,
    )

    # Compute relative gaps
    relative_gaps = []
    for i in range(1, len(sorted_scores)):
        prev_score = sorted_scores.iloc[i - 1]
        curr_score = sorted_scores.iloc[i]
        relative_gap = (prev_score - curr_score) / prev_score if prev_score > 0 else 0
        relative_gaps.append((i - 1, relative_gap))

    # Find two largest relative gaps
    relative_gaps.sort(key=lambda x: x[1], reverse=True)

    if len(relative_gaps) >= 2:
        gap_positions = sorted([x[0] for x in relative_gaps[:2]])
        i_star, j_star = gap_positions[0], gap_positions[1]

        dropoff_bins = pd.Series("low", index=sorted_scores.index, dtype="string")
        dropoff_bins.iloc[: i_star + 1] = "high"
        dropoff_bins.iloc[i_star + 1 : j_star + 1] = "medium"
    else:
        dropoff_bins = quantile_bins

    # Take minimum of quantile and dropoff bins
    bin_order = {"high": 3, "medium": 2, "low": 1}
    final_local_bins = pd.Series(index=sorted_scores.index, dtype="string")

    for idx in sorted_scores.index:
        quantile_val = bin_order[quantile_bins[idx]]
        dropoff_val = bin_order[dropoff_bins[idx]]
        min_val = min(quantile_val, dropoff_val)
        final_local_bins[idx] = {3: "high", 2: "medium", 1: "low"}[min_val]

    return final_local_bins
