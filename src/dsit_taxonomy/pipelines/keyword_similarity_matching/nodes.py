import logging
import uuid
from typing import List, Dict
import lancedb
import pandas as pd
from scipy.stats import entropy
from spacy.lang.en import English

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


def compute_document_scores(
    documents: pd.DataFrame, document_matches: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the similarity scores for the top 3 matches for each document and sum them
    for each project and taxonomy label.

    Args:
        documents (pd.DataFrame): DataFrame containing the GtR documents.
        document_matches (pd.DataFrame): DataFrame containing the document matches.

    Returns:
        pd.DataFrame: DataFrame containing the project_id, taxonomy_label_id, and
        similarity_score.
    """
    # merge documents with their matches
    merged_documents = pd.merge(
        documents[["project_id", "uuid"]],
        document_matches,
        left_on="uuid",
        right_on="document_id",
        how="right",
    )

    # get top 3 similarity scores for each document
    top_3_scores = (
        merged_documents.sort_values(
            by=["project_id", "uuid", "similarity_score"], ascending=[True, True, False]
        )
        .groupby(["project_id", "uuid"], as_index=False)
        .head(3)
    )

    # [HACK] Limit the number of non-informational matches
    # by filtering out scores in the bottom quantile
    top_3_scores = top_3_scores[
        top_3_scores["similarity_score"]
        >= top_3_scores["similarity_score"].quantile(0.25)
    ]

    # sum the top 3 scores for each project and taxonomy label
    output_scores = top_3_scores.groupby(
        ["project_id", "taxonomy_label_id"], as_index=False
    ).agg(similarity_score=("similarity_score", "sum"))

    logger.info("Normalising scores")
    normalised_output_scores = output_scores.groupby(
        "project_id", group_keys=False
    ).apply(_normalise_project_scores)

    return normalised_output_scores


def create_project_score_data(
    document_scores: pd.DataFrame,
    keyword_scores: pd.DataFrame,
    keyword_data: pd.DataFrame,
    taxonomy: lancedb,
) -> pd.DataFrame:
    """
    Merge the document scores with the keyword scores and compute the relevance scores.

    Args:
        document_scores (pd.DataFrame): DataFrame containing the document scores.
        keyword_scores (pd.DataFrame): DataFrame containing the keyword scores.
        keyword_data (pd.DataFrame): DataFrame containing the keyword data.

    Returns:
        pd.DataFrame: DataFrame containing the project_id, keyword_id, taxonomy_label_id,
    """

    # merge the document scores with the keyword scores
    project_keyword_map = (
        keyword_data[["project_ids", "uuid"]]
        .explode("project_ids")
        .rename(columns={"project_ids": "project_id", "uuid": "keyword_id"})
    )

    # merge the keyword scores with the project ids
    project_keywords = pd.merge(
        project_keyword_map,
        keyword_scores[["keyword_id", "taxonomy_label_id"]],
        on="keyword_id",
        how="left",
    )

    # merge the keyword_ids to the documents
    documents = pd.merge(
        document_scores,
        project_keywords,
        on=["project_id", "taxonomy_label_id"],
        how="inner",
    )

    # rename documents' similarity_score to "weight"
    documents.rename(columns={"similarity_score": "weight"}, inplace=True)

    # merge back the keyword similarity and entropy
    project_data = pd.merge(
        documents,
        keyword_scores,
        on=["keyword_id", "taxonomy_label_id"],
        how="left",
    )

    # create relevance score
    project_data["relevance_score"] = (
        project_data["weight"] * project_data["similarity_score"]
    )

    # merge with keywords and taxonomy labels
    project_data = project_data.merge(
        keyword_data[["uuid", "keyword"]],
        left_on="keyword_id",
        right_on="uuid",
        how="left",
    )
    project_data = project_data.merge(
        taxonomy[["uuid", "label"]],
        left_on="taxonomy_label_id",
        right_on="uuid",
        how="left",
    )
    project_data.drop(columns=["uuid_x", "uuid_y"], inplace=True)

    # groupby project_id, taxonomy_label_id to sum the relevance scores

    return project_data[
        [
            "project_id",
            "keyword_id",
            "taxonomy_label_id",
            "keyword",
            "label",
            "weight",
            "similarity_score",
            "shannon_entropy",
            "relevance_score",
        ]
    ]


def aggregate_scores_to_labels(
    keyword_scores: pd.DataFrame,
):
    """Aggregate keyword scores to the project label level."""
    return (
        keyword_scores.groupby(["project_id", "taxonomy_label_id"], as_index=False)
        .agg(
            weight=("weight", "first"),
            relevance_score=("relevance_score", "sum"),
        )
        .reset_index(drop=True)
    )


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


def _normalise_project_scores(group):  #
    """Normalise similarity scores for a project."""
    if len(group) == 1:
        # single label case: assign normalised score of 1.0
        group["similarity_score"] = 1.0
    else:
        max_score = group["similarity_score"].max()
        min_score = group["similarity_score"].min()
        if max_score == min_score:
            # assign equal normalised scores if max == min
            group["similarity_score"] = 1.0
        else:
            # normalise scores normally
            group["similarity_score"] = (group["similarity_score"] - min_score) / (
                max_score - min_score
            )
    return group


def _split_sentences(document: str, nlp: English) -> List[str]:
    """Split a document into sentences."""
    doc = nlp(document)
    return [sent.text for sent in doc.sents]
