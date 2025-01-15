"""
A Kedro hook to process documents into embeddings using SentenceTransformers
and store them in LanceDB before a specified node runs.
"""

import logging
from typing import Any
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LanceDBHook:
    """
    A Kedro hook to process documents into embeddings using SentenceTransformers
    and store them in LanceDB before a specified node runs.
    """

    def __init__(
        self,
        target_node_name: str,
        lancedb_path: str,
    ):
        """
        Start the hook.

        Args:
            target_node_name (str): The name of the node before which the hook will execute.
            lancedb_path (str): Path to the LanceDB database.
        """
        self.target_node_name = target_node_name
        self.lancedb_path = lancedb_path

    @hook_impl
    def before_node_run(self, node, catalog: DataCatalog, inputs: dict) -> None:
        """
        Hook implementation to process documents into embeddings and store them in LanceDB.

        Args:
            node: The node that is about to run.
            catalog (DataCatalog): Kedro's DataCatalog to load datasets and parameters.
            inputs (dict): The inputs to the node.
        """
        if node.name != self.target_node_name:
            return

        # initialise LanceDB connection
        db = lancedb.connect(self.lancedb_path)

        # load parameters to find the embeddings model
        parameters = catalog.load("parameters")
        model = SentenceTransformer(parameters["embeddings_model_name"])

        # process all input datasets containing ".db"
        db_tables = {}
        for input_name, input_object in inputs.items():
            if input_name.endswith(".db"):
                logger.info("Processing table '%s' as a vector table.", input_name)

                assert isinstance(input_object, pd.DataFrame), (
                    f"Expected input '{input_name}' to be a DataFrame, "
                    f"but got {type(input_object).__name__}."
                )

                # extract texts
                texts = self._extract_texts(input_object)

                # convert texts to embeddings
                embeddings = model.encode(texts, show_progress_bar=True)

                # extract the column uuid (only accept 1)
                uuids = input_object["uuid"].tolist()

                if "project_id" in input_object.columns:
                    project_ids = input_object["project_id"].tolist()
                    data_to_insert = [
                        {
                            "project_id": project_id,
                            "id": uuid,
                            "text": text,
                            "vector": embedding.tolist(),
                        }
                        for project_id, uuid, text, embedding in zip(
                            project_ids, uuids, texts, embeddings
                        )
                    ]
                else:
                    data_to_insert = [
                        {"id": uuid, "text": text, "vector": embedding.tolist()}
                        for uuid, text, embedding in zip(uuids, texts, embeddings)
                    ]

                # create or overwrite table in LanceDB
                self._store_embeddings_in_lancedb(db, input_name, data_to_insert)
                db_tables[input_name] = db[input_name]

        return db_tables

    def _extract_texts(self, documents: Any) -> list:
        """
        Extract a list of text strings from the provided dataset.

        Args:
            documents (Any): The dataset to extract texts from (e.g., list or DataFrame).

        Returns:
            list: A list of text strings.
        """
        if isinstance(documents, list):
            return documents
        elif isinstance(documents, pd.DataFrame):
            for column in ["label", "text", "keyword"]:
                if column in documents.columns:
                    return documents[column].dropna().tolist()
            raise ValueError("No suitable text column found in the DataFrame.")
        else:
            raise ValueError("Unsupported document format. Expected list or DataFrame.")

    def _store_embeddings_in_lancedb(
        self, db: lancedb, table_name: str, data: list
    ) -> None:
        """
        Store embeddings in LanceDB.

        Args:
            db (lancedb.LanceDB): The LanceDB instance.
            table_name (str): Name of the table to create or overwrite.
            data (list): List of dictionaries containing 'text' and 'embedding'.
        """
        if table_name in db.table_names():
            db.drop_table(table_name)
        db.create_table(table_name, data=data)
        logger.info("Embeddings successfully stored in LanceDB table '%s'.", table_name)
