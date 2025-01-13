import logging
import lancedb
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def compute_similarity(taxonomy, keywords):
    logger.info("Hook test: %s", taxonomy)
    return keywords


def _search_batch(keyword_batch, taxonomy_table, top_n=10, number_returns=5000):
    """
    Perform similarity search for a batch of keywords, computing entropy over a larger number
    of matches (entropy_limit) but only retaining the top N matches for output.
    Args:
        keyword_batch: List of keyword embeddings and IDs.
        taxonomy_table: LanceDB table for taxonomy embeddings.
        top_n: Number of top matches to retain for final output.
        number_returns: Number of matches to use for Shannon entropy calculation.
    Returns:
        List of results for the batch.
    """
    results = []

    for keyword in keyword_batch:
        embedding = keyword["vector"]
        keyword_id = keyword["id"]

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
                    "keyword_id": keyword_id,
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


def compute_similarities_and_entropy(
    taxonomy: lancedb,
    keywords: lancedb,
    batch_size: int = 1000,
    top_n: int = 10,
    number_returns: int = 1000,
):
    # convert LanceDB table to a list of dicts for batch processing
    keywords_dict = keywords.to_pandas().to_dict(orient="records")

    # Divide keywords into batches
    keyword_batches = [
        keywords_dict[i : i + batch_size]
        for i in range(0, len(keywords_dict), batch_size)
    ]

    # Process each batch sequentially
    results = []
    for i, batch in enumerate(keyword_batches):
        logger.info("Processing batch %d / %d", i + 1, len(keyword_batches))
        batch_results = _search_batch(
            batch, taxonomy, top_n=top_n, number_returns=number_returns
        )
        results.extend(batch_results)

    logger.info("Flattening results")
    flat_results = [
        {
            "keyword_id": result["keyword_id"],
            "taxonomy_label_id": match.id,
            "similarity_score": match.similarity_score,
            "shannon_entropy": result["shannon_entropy"],
        }
        for result in results
        for match in result["top_matches"].itertuples()
    ]

    return pd.DataFrame(flat_results)
