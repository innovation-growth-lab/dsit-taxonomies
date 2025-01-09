import logging
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger(__name__)


def preprocess_cwts(cwts_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataframe with tree-like concatenation of labels and IDs across levels.

    Args:
        cwts_dataframe (pd.DataFrame): The input dataframe with taxonomy data.

    Returns:
        pd.DataFrame: A dataframe with two columns:
            - 'label': The tree-like concatenation of all labels across levels.
            - 'id_path': The tree-like concatenation of all IDs across levels.
    """
    labels = []
    id_paths = []

    for _, row in cwts_dataframe.iterrows():
        paths = [
            ('domain_name', 'domain_id'),
            ('field_name', 'field_id'),
            ('subfield_name', 'subfield_id'),
            ('topic_name', 'topic_id')
        ]

        label_path = []
        id_path = []

        for label_col, id_col in paths:
            if row[label_col]:
                label_path.append(row[label_col])
                id_path.append(str(row[id_col]))
                labels.append(' > '.join(label_path))
                id_paths.append(' > '.join(id_path))

    result_df = pd.DataFrame({
        'label': labels,
        'id_path': id_paths
    })

    return result_df


# def _preprocess_keywords(keywords: pd.Series) -> pd.Series:
#     """Preprocess the keywords by lowercasing, and removing trailing spaces"""
#     return keywords.str.lower().str.strip()


# def generate_embeddings(keyword_dataframe: pd.DataFrame) -> pd.DataFrame:
#     """
#     Generate embeddings for each keyword in the dataframe.

#     Args:
#         keyword_dataframe: The dataframe containing the keywords.

#     Returns:
#         pd.DataFrame: A dataframe with three columns:
#             - 'label': The unique keyword.
#             - 'num_annotators': The number of distinct annotator classes the keyword appears in.
#             - 'embedding': The embedding for the keyword.
#     """
#     embeddings = model.encode(
#         keyword_dataframe["label"].tolist(),
#         show_progress_bar=True,
#         convert_to_tensor=False,
#     )
#     logger.info("Generated embeddings for %s keywords", len(embeddings))
#     keyword_dataframe["embedding"] = embeddings
#     return keyword_dataframe
