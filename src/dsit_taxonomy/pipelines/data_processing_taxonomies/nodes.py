import logging
import uuid
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger(__name__)


def preprocess_cwts_topics(cwts_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataframe with tree-like concatenation of labels and IDs across levels.

    Args:
        cwts_dataframe (pd.DataFrame): The input dataframe with taxonomy data.

    Returns:
        pd.DataFrame: A dataframe with two columns:
            - 'label': The tree-like concatenation of all labels across levels.
            - 'id_path': The tree-like concatenation of all IDs across levels.
    """
    labels, id_paths, levels = [], [], []

    for _, row in cwts_dataframe.iterrows():
        paths = [
            ("domain_name", "domain_id"),
            ("field_name", "field_id"),
            ("subfield_name", "subfield_id"),
            ("topic_name", "topic_id"),
        ]

        label_path = []
        id_path = []

        for level, (label_col, id_col) in enumerate(paths):
            if row[label_col]:
                label_path.append(row[label_col])
                id_path.append(str(row[id_col]))
                labels.append(" > ".join(label_path))
                id_paths.append(" > ".join(id_path))
                levels.append(level)

    result_df = pd.DataFrame({"label": labels, "id_path": id_paths, "level": levels})
    result_df.drop_duplicates(subset=["label", "id_path"], inplace=True)
    taxonomy, bottom_level = _before_return(result_df)

    return taxonomy, bottom_level


def preprocess_goscience_taxonomy(goscience_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataframe with tree-like concatenation of labels and IDs across levels.

    Args:
        goscience_dataframe (pd.DataFrame): The input dataframe with taxonomy data.

    Returns:
        pd.DataFrame: A dataframe with two columns:
            - 'label': The tree-like concatenation of all labels across levels.
            - 'id_path': The tree-like concatenation of all IDs across levels.

    NB. The mapping of nodes is non-unique and not strictly monotonically increasing
        ie. Quantum timing has path L0 > L1 > L1.
    This is assumed to be a design choice and is preserved in the output.
    """
    labels, level_paths, levels = [], [], []

    def _build_path(concept_name, current_path, current_level_path, current_level):
        row = goscience_dataframe[
            goscience_dataframe["concept_name"] == concept_name
        ].iloc[0]
        current_path.append(row["concept_name"])
        current_level_path.append(row["taxonomy_level"])
        current_level.append(row["taxonomy_level"])

        labels.append(" > ".join(current_path))
        level_paths.append(" > ".join(current_level_path))
        levels.append(current_level[-1])

        children = goscience_dataframe[
            goscience_dataframe["parent_name"] == concept_name
        ]
        for _, child in children.iterrows():
            _build_path(
                child["concept_name"],
                current_path.copy(),
                current_level_path.copy(),
                current_level.copy(),
            )

    # start with root nodes (nodes with no parent)
    root_nodes = goscience_dataframe[pd.isnull(goscience_dataframe["parent_name"])]
    for _, root in root_nodes.iterrows():
        _build_path(root["concept_name"], [], [], [])

    result_df = pd.DataFrame(
        {"label": labels, "level_path": level_paths, "level": levels}
    )

    result_df.drop_duplicates(subset=["label", "level_path"], inplace=True)
    taxonomy, bottom_level = _before_return(result_df)

    return taxonomy, bottom_level


def preprocess_oa_concepts(concepts_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the OpenAlex concepts data, derived from the MAG taxonomy.

    Args:
        concepts_dataframe (pd.DataFrame): The input dataframe with OA concepts data.

    Returns:
        pd.DataFrame: A dataframe with two columns:
            - 'label': The tree-like concatenation of all display names across levels.
            - 'id_path': The tree-like concatenation of all IDs across levels.
    """
    # preprocess the data
    concepts_dataframe = _preprocess_concepts(concepts_dataframe)

    # identify root nodes
    root_nodes = concepts_dataframe[
        pd.isnull(concepts_dataframe["parent_ids"]) & (concepts_dataframe["level"] == 0)
    ]

    labels, id_paths, levels = [], [], []

    def _build_path(concept_name, current_path, current_id_path, current_level):
        row = concepts_dataframe[
            concepts_dataframe["display_name"] == concept_name
        ].iloc[0]
        current_path.append(row["display_name"])
        current_id_path.append(row["openalex_id"])
        current_level.append(str(row["level"]))

        labels.append(" > ".join(current_path))
        id_paths.append(" > ".join(current_id_path))
        levels.append(current_level[-1])

        for _, child in concepts_dataframe[
            concepts_dataframe["parent_display_names"] == concept_name
        ].iterrows():
            _build_path(
                child["display_name"],
                current_path.copy(),
                current_id_path.copy(),
                current_level.copy(),
            )

    # start with root nodes (nodes with no parent)
    for _, root in root_nodes.iterrows():
        logger.info(
            "Building path for root node: %s. Number of roots: %d",
            root["display_name"],
            len(root_nodes),
        )
        _build_path(root["display_name"], [], [], [])

    result_df = pd.DataFrame({"label": labels, "id_path": id_paths, "level": levels})

    # drop duplicates
    result_df.drop_duplicates(subset=["label", "id_path"], inplace=True)
    taxonomy, bottom_level = _before_return(result_df)

    return taxonomy, bottom_level


def _preprocess_concepts(concepts_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the OpenAlex concepts data, derived from the MAG taxonomy."""
    # preprocess parent columns and clean up the data
    return (
        concepts_dataframe.copy()
        .assign(
            parent_display_names=lambda df: df["parent_display_names"]
            .str.split(", ")
            .apply(lambda x: x if isinstance(x, list) else []),
            parent_ids=lambda df: df["parent_ids"]
            .str.split(", ")
            .apply(lambda x: x if isinstance(x, list) else [np.nan, np.nan]),
        )
        .assign(
            parent_tuples=lambda df: df.apply(
                lambda x: list(zip(x["parent_display_names"], x["parent_ids"])), axis=1
            )
        )
        .explode("parent_tuples", ignore_index=True)
        .assign(
            parent_display_names=lambda df: df["parent_tuples"].apply(
                lambda x: x[0] if pd.notna(x) else ""
            ),
            parent_ids=lambda df: df["parent_tuples"].apply(
                lambda x: x[1] if pd.notna(x) else ""
            ),
        )
        .drop(columns=["parent_tuples"])
        .replace({"parent_display_names": {"": np.nan}, "parent_ids": {"": np.nan}})
        .assign(
            openalex_id=lambda df: df["openalex_id"]
            .str.replace("https://openalex.org/", "")
            .str.lower(),
            parent_ids=lambda df: df["parent_ids"]
            .str.replace("https://openalex.org/", "")
            .str.lower(),
        )
    )


def _before_return(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the keywords data."""
    # add uuids
    dataframe["uuid"] = dataframe["label"].apply(
        lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, x))
    )

    # output also the bottom level of the hierarchy
    bottom_level = dataframe[dataframe["level"] == dataframe["level"].max()]

    return dataframe, bottom_level
