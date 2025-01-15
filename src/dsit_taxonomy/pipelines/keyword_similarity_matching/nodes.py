import logging
import uuid
from typing import List, Dict
import lancedb
import pandas as pd
from scipy.stats import entropy
from spacy.lang.en import English

logger = logging.getLogger(__name__)


def _search_batch(
    document_batch: List[Dict[str, str]],
    taxonomy_table: lancedb,
    top_n=10,
    number_returns=5000,
) -> pd.DataFrame:
    """
    Perform similarity search for a batch of strings, computing entropy over a larger number
    of matches (entropy_limit) but only retaining the top N matches for output. It also
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


def compute_document_similarity_and_weights(
    documents: pd.DataFrame, document_matches: pd.DataFrame
) -> pd.DataFrame:
    
    document_dataframe = pd.merge(documents[["project_id", "uuid"]], document_matches, left_on="uuid", right_on="document_id", how="right")

    # groupby project_id, taxonomy_id and extract highest similarity score for all unique labels in the project
    document_matches = (
        document_dataframe.groupby(["project_id", "taxonomy_label_id"])
        .agg({"similarity_score": "max"})
        .reset_index()
    )

    return document_matches


def _split_sentences(document: str, nlp: English) -> List[str]:
    """Split a document into sentences."""
    doc = nlp(document)
    return [sent.text for sent in doc.sents]
