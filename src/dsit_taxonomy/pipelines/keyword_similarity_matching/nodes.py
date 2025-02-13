import logging
import uuid
from typing import List, Dict
import lancedb
import pandas as pd
from scipy.stats import entropy
from spacy.lang.en import English
from joblib import Parallel, delayed
from functools import partial
import numpy as np

logger = logging.getLogger(__name__)


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
    documents["sentence_text"] = documents["text"].apply(_split_sentences, nlp=nlp)

    # explode sentences into separate rows
    sentences = documents.explode("sentence_text")

    # add uuids
    sentences["uuid"] = sentences["sentence_text"].apply(
        lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, x))
    )

    # [HACK] drop duplicate rows to avoid non-informational matches
    # This of course has the downside that it may remove some valid matches.
    sentences = sentences.drop_duplicates(subset=["sentence_text"], keep=False)

    return (
        documents[["project_id", "text"]],
        sentences[["project_id", "uuid", "sentence_text"]],
    )


def compute_similarities(
    taxonomy: pd.DataFrame,
    documents: pd.DataFrame,
    batch_size: int = 1000,
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute similarity scores using parallel processing."""
    # Convert to numpy arrays
    taxonomy_embeddings = np.vstack(taxonomy["vector"].values)
    taxonomy_ids = taxonomy["id"].values
    documents_dict = documents.to_dict(orient="records")

    # Divide documents into batches
    document_batches = [
        documents_dict[i : i + batch_size]
        for i in range(0, len(documents_dict), batch_size)
    ]

    # Process batches in parallel
    logger.info("Processing %d batches in parallel", len(document_batches))
    results = Parallel(n_jobs=8, verbose=10)(
        delayed(_search_batch)(
            batch, taxonomy_embeddings, taxonomy_ids, top_n
        ) for batch in document_batches
    )

    logger.info("Flattening results")
    return pd.concat(results, ignore_index=True)


def aggregate_sentence_matches(
    sentences: pd.DataFrame,
    sentence_matches: pd.DataFrame,
    min_score_quantile: float,
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
        # Sublinear scaling of match frequency
        (1 + np.log1p(project_scores["match_count"]))
        / (1 + np.log1p(project_scores["n_sentences"]))
        *
        # Average strength component
        project_scores["score_mean"]
        *
        # Boost factor for very strong individual matches
        (1 + np.log1p(project_scores["score_max"]))
    )

    project_scores = project_scores.rename(
        columns={"initial_score": "similarity_score"}
    )

    # Apply local quantile threshold per project
    project_thresholds = project_scores.groupby("project_id")[
        "similarity_score"
    ].transform(lambda x: x.quantile(min_score_quantile))
    filtered_scores = project_scores[
        project_scores["similarity_score"] >= project_thresholds
    ]

    # Log distribution of topics per project
    topics_per_project = filtered_scores.groupby("project_id").size()
    logger.info(
        "Score and topic distribution:\n" "Topics per project:\n%s",
        topics_per_project.describe().to_string(),
    )

    # Scale scores relative to maximum within each project
    filtered_scores["similarity_score"] = filtered_scores.groupby("project_id")[
        "similarity_score"
    ].transform(lambda x: x / x.max())

    logger.info(
        "Scaled score distribution:\n%s",
        filtered_scores["similarity_score"].describe().to_string(),
    )

    return filtered_scores


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

    return final_scores[
        [
            "project_id",
            "keyword_id",
            "taxonomy_label_id",
            "keyword",
            "label",
            "similarity_score_sent",
            "similarity_score_key",
            "shannon_entropy",
        ]
    ]


def aggregate_scores_to_labels(
    scores: pd.DataFrame,
    sentence_weight: float,
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
        similarity_quantile_threshold: Threshold for filtering low scores
        global_q2_threshold: Global threshold for medium confidence
        global_q3_threshold: Global threshold for high confidence
        local_q2_threshold: Project-level threshold for medium confidence
        local_q3_threshold: Project-level threshold for high confidence
    """
    logger.info(
        "Aggregating scores with parameters:\n"
        "Weights - Sentence: %0.2f\n"
        "Score threshold: %0.2f\n"
        "Global quantiles - Q2: %0.2f, Q3: %0.2f\n"
        "Local quantiles - Q2: %0.2f, Q3: %0.2f",
        sentence_weight,
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
        (1 - sentence_weight),
    )
    scores["relevance_score"] = (
        (sentence_weight * scores["similarity_score_sent"])
        + ((1 - sentence_weight) * scores["similarity_score_key"])
    ) ** 2

    # Filter low keyword scores
    scores = scores[
        scores["similarity_score_key"]
        >= scores["similarity_score_key"].quantile(similarity_quantile_threshold)
    ]

    logger.info("Aggregating final scores to project-label level")

    aggregated = (
        scores.groupby(["project_id", "taxonomy_label_id", "label"], as_index=False)
        .agg(
            {
                "relevance_score": "max",
                "similarity_score_sent": "max",
                "similarity_score_key": "max",
                "shannon_entropy": "mean",
                "keyword_id": "nunique",
            }
        )
        .rename(columns={"keyword_id": "num_keywords"})
    )

    # Assign confidence bins
    return _assign_confidence_bins(
        aggregated,
        global_q2_threshold,
        global_q3_threshold,
        local_q2_threshold,
        local_q3_threshold,
    )


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
    taxonomy_embeddings: np.ndarray,
    taxonomy_ids: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Perform vectorized similarity search for a batch of documents.

    Args:
        document_batch: List of document embeddings
        taxonomy_embeddings: Array of taxonomy label embeddings (n_labels, embedding_dim)
        taxonomy_ids: Array of taxonomy label IDs
        top_n: Number of top matches to retain

    Returns:
        DataFrame with similarity matches
    """
    # Stack document embeddings
    doc_embeddings = np.vstack([doc["vector"] for doc in document_batch])
    doc_ids = [doc["id"] for doc in document_batch]
    
    # Compute cosine similarity matrix
    # (num_docs, embedding_dim) @ (embedding_dim, num_labels) = (num_docs, num_labels)
    similarities = doc_embeddings @ taxonomy_embeddings.T
    
    # Normalize for cosine similarity
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    tax_norms = np.linalg.norm(taxonomy_embeddings, axis=1, keepdims=True).T
    similarities = similarities / (doc_norms @ tax_norms)
    
    # Get top N indices and scores for each document
    top_indices = np.argpartition(-similarities, top_n, axis=1)[:, :top_n]
    
    results = []
    for i, doc_id in enumerate(doc_ids):
        top_idx = top_indices[i]
        scores = similarities[i, top_idx]
        
        # Sort by score
        sort_idx = np.argsort(-scores)
        top_idx = top_idx[sort_idx]
        scores = scores[sort_idx]
        
        results.append(
            pd.DataFrame({
                "document_id": doc_id,
                "taxonomy_label_id": taxonomy_ids[top_idx],
                "similarity_score": scores,
            })
        )
    
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
    results = Parallel(n_jobs=n_jobs)(
        delayed(_assign_local_bins_partial)(group) for group in project_groups
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


def prune_global_matches(
    matches: pd.DataFrame,
    global_embedding_threshold: float,
) -> pd.DataFrame:
    """
    Prune low-scoring matches based on a global quantile threshold.

    Args:
        matches: DataFrame with raw similarity matches
        global_embedding_threshold: Quantile threshold below which matches are dropped

    Returns:
        DataFrame with pruned matches
    """
    logger.info(
        "Pruning global matches with threshold: %0.2f\n" "Initial matches: %d",
        global_embedding_threshold,
        len(matches),
    )

    # Compute threshold
    score_threshold = matches["similarity_score"].quantile(global_embedding_threshold)

    # Filter matches
    pruned_matches = matches[matches["similarity_score"] >= score_threshold]

    logger.info(
        "Score threshold: %0.3f\n" "Remaining matches: %d (%0.1f%%)",
        score_threshold,
        len(pruned_matches),
        100 * len(pruned_matches) / len(matches),
    )

    return pruned_matches
