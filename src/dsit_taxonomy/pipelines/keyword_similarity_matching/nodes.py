import logging
import lancedb
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def compute_similarity(taxonomy, keywords):
    logger.info("Hook test: %s", taxonomy)
    return keywords


def _search_batch(keyword_batch, taxonomy_table):
    """Perform similarity search for a batch of keywords."""
    results = []
    for keyword in keyword_batch:
        embedding = keyword["embedding"]
        keyword_id = keyword["id"]

        # Perform similarity search
        similar_labels = taxonomy_table.search(embedding).limit(150).to_pandas()
        results.append(
            {"keyword_id": keyword_id, "taxonomy_similarities": similar_labels}
        )

    return results


def _compute_shannon_entropy(similarity_scores):
    """Compute the Shannon entropy for a given list of similarity scores."""
    return entropy(similarity_scores, base=2)


def compute_similarities_and_entropy(
    taxonomy: lancedb, keywords: lancedb, batch_size: int = 1000
):
    # convert LanceDB table to a list of dicts for batch processing
    keywords_dict = keywords.to_pandas().to_dict(orient="records")

    # divide keywords into batches
    keyword_batches = [
        keywords_dict[i : i + batch_size]
        for i in range(0, len(keywords_dict), batch_size)
    ]

    # process each batch sequentially
    results = []
    for i, batch in enumerate(keyword_batches):
        logger.info("Processing batch %d / %d", i + 1, len(keyword_batches))
        batch_results = _search_batch(batch, taxonomy)
        results.extend(batch_results)

    # flatten results into a DataFrame
    flat_results = [
        {
            "keyword_id": result["keyword_id"],
            "taxonomy_label_id": similarity.id,
            "similarity_score": similarity.similarity_score,
        }
        for result in results
        for similarity in result["taxonomy_similarities"].itertuples()
    ]

    df = pd.DataFrame(flat_results)

    # compute Shannon entropy for each keyword
    entropy_results = (
        df.groupby("keyword_id")["similarity_score"]
        .apply(_compute_shannon_entropy)
        .reset_index(name="shannon_entropy")
    )

    # merge entropy results with the flattened results
    final_results = df.merge(entropy_results, on="keyword_id")

    return final_results
