"""Utility functions for the keyword similarity matching pipeline."""

import numpy as np
import pandas as pd
from typing import List, Dict
from spacy.lang.en import English


def search_batch(
    document_batch: List[Dict[str, str]],
    taxonomy_embeddings: np.ndarray,
    taxonomy_ids: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Perform vectorised similarity search for a batch of documents.

    Args:
        document_batch: List of dictionaries containing document embeddings and IDs
        taxonomy_embeddings: Array of taxonomy label embeddings (n_labels, embedding_dim)
        taxonomy_ids: Array of taxonomy label IDs corresponding to embeddings
        top_n: Number of top matches to retain per document

    Returns:
        DataFrame with columns:
            - document_id: ID of the document
            - taxonomy_label_id: ID of the matched taxonomy label
            - similarity_score: Cosine similarity score between document and label
    """
    # Stack document embeddings
    doc_embeddings = np.vstack([doc["vector"] for doc in document_batch])
    doc_ids = [doc["id"] for doc in document_batch]

    # Compute cosine similarity matrix
    similarities = doc_embeddings @ taxonomy_embeddings.T

    # Normalise for cosine similarity
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
            pd.DataFrame(
                {
                    "document_id": doc_id,
                    "taxonomy_label_id": taxonomy_ids[top_idx],
                    "similarity_score": scores,
                }
            )
        )

    return pd.concat(results, ignore_index=True)


def split_sentences(document: str, nlp: English) -> List[str]:
    """
    Split a document into sentences using spaCy.

    Args:
        document: Text document to split into sentences
        nlp: Initialised spaCy English language model with sentencizer

    Returns:
        List of sentences extracted from the document
    """
    doc = nlp(document)
    return [sent.text for sent in doc.sents] 