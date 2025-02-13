"""
A Kedro hook to process documents into embeddings using SentenceTransformers
before a specified node runs.
"""

import logging
from typing import Any
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingsHook:
    """
    A Kedro hook to process documents into embeddings using SentenceTransformers
    before a specified node runs.
    """

    def __init__(self, target_node_name: str):
        """
        Start the hook.

        Args:
            target_node_name (str): The name of the node before which the hook will execute.
        """
        self.target_node_name = target_node_name

    @hook_impl
    def before_node_run(self, node, catalog: DataCatalog, inputs: dict) -> None:
        """
        Hook implementation to process documents into embeddings.

        Args:
            node: The node that is about to run.
            catalog (DataCatalog): Kedro's DataCatalog.
            inputs (dict): The inputs to the node.
        """
        if node.name != self.target_node_name:
            return

        # Load parameters to find the embeddings model
        parameters = catalog.load("parameters")
        model = SentenceTransformer(parameters["embeddings_model_name"])

        # Process all input datasets containing ".db"
        embedded_tables = {}
        for input_name, input_object in inputs.items():
            if input_name.endswith(".db"):
                logger.info("Processing table '%s' for embeddings.", input_name)

                assert isinstance(input_object, pd.DataFrame), (
                    f"Expected input '{input_name}' to be a DataFrame, "
                    f"but got {type(input_object).__name__}."
                )

                # Extract texts
                texts = self._extract_texts(input_object)

                # Convert texts to embeddings
                embeddings = model.encode(texts, show_progress_bar=True)

                # Create DataFrame with embeddings
                if "project_id" in input_object.columns:
                    if "uuid" in input_object.columns:
                        df = pd.DataFrame(
                            {
                                "project_id": input_object["project_id"],
                                "id": input_object["uuid"],
                                "text": texts,
                                "vector": list(embeddings),  # Store as numpy arrays
                            }
                        )
                    else:
                        df = pd.DataFrame(
                            {
                                "id": input_object["project_id"],
                                "text": texts,
                                "vector": list(embeddings),
                            }
                        )
                else:
                    df = pd.DataFrame(
                        {
                            "id": input_object["uuid"],
                            "text": texts,
                            "vector": list(embeddings),
                        }
                    )

                embedded_tables[input_name] = df

        return embedded_tables

    def _extract_texts(self, documents: Any) -> list:
        """Extract text strings from the provided dataset."""
        if isinstance(documents, list):
            return documents
        elif isinstance(documents, pd.DataFrame):
            for column in ["label", "text", "keyword", "sentence_text"]:
                if column in documents.columns:
                    return documents[column].dropna().tolist()
            raise ValueError("No suitable text column found in the DataFrame.")
        else:
            raise ValueError("Unsupported document format. Expected list or DataFrame.")
