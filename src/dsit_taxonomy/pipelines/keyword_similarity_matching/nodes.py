import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import lancedb
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def compute_similarity(taxonomy, keywords):
    logger.info("Hook test: %s", taxonomy)
    return keywords


def _search_batch(keyword_batch, taxonomy_table):
    """ """
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
    # Convert LanceDB table to a list of dicts for batch processing
    keywords_dict = keywords.to_pandas().to_dict(orient="records")

    # Divide keywords into batches
    keyword_batches = [
        keywords_dict[i : i + batch_size]
        for i in range(0, len(keywords_dict), batch_size)
    ]

    # Use ProcessPoolExecutor for parallel processing
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_search_batch, batch, taxonomy) for batch in keyword_batches
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing batches"
        ):
            results.extend(future.result())

    # Flatten results into a DataFrame
    flat_results = []
    for result in results:
        for similarity in result["taxonomy_similarities"].itertuples():
            flat_results.append(
                {
                    "keyword_id": result["keyword_id"],
                    "taxonomy_label_id": similarity.id,
                    "similarity_score": similarity.similarity_score,
                }
            )

    df = pd.DataFrame(flat_results)

    # Compute Shannon entropy for each keyword
    entropy_results = (
        df.groupby("keyword_id")["similarity_score"]
        .apply(_compute_shannon_entropy)
        .reset_index()
    )
    entropy_results.columns = ["keyword_id", "shannon_entropy"]

    # Merge entropy results with the flattened results
    final_results = df.merge(entropy_results, on="keyword_id")

    return final_results
