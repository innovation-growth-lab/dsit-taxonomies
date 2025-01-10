import logging
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger(__name__)


def aggregate_keyword_annotators(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the number of distinct annotator classes each keyword appears in.

    Args:
        dataframes (pd.DataFrame): The dataframes to process.

    Returns:
        pd.DataFrame: A dataframe with two columns:
            - 'label': The unique keyword.
            - 'num_annotators': The number of distinct annotator classes the keyword appears in.
    """
    keyword_to_annotators = {}

    for df in dataframes:
        logger.info("Processing dataframe with columns: %s", df.columns)
        keyword_column = next(col for col in df.columns if "_keywords" in col)
        annotator_class = keyword_column.split("_")[0]

        # explode the keywords and drop NaN values
        exploded = df.explode(keyword_column).dropna(subset=[keyword_column])

        # preprocess the keywords
        exploded[keyword_column] = _preprocess_keywords(exploded[keyword_column])

        # aggregate the keywords with annotator classes
        for keyword in exploded[keyword_column].unique():
            keyword_to_annotators.setdefault(keyword, set()).add(annotator_class)

    # prepare the output dataframe
    output_df = pd.DataFrame(
        {
            "label": keyword_to_annotators.keys(),
            "num_annotators": [
                len(annotators) for annotators in keyword_to_annotators.values()
            ],
        }
    )

    # sort the dataframe by num_annotators in descending order, then by label alphabetically
    return output_df.sort_values(
        by=["num_annotators", "label"], ascending=[False, True]
    )


def _preprocess_keywords(keywords: pd.Series) -> pd.Series:
    """Preprocess the keywords by lowercasing, and removing trailing spaces"""
    return keywords.str.lower().str.strip()


def generate_embeddings(keyword_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate embeddings for each keyword in the dataframe.

    Args:
        keyword_dataframe: The dataframe containing the keywords.

    Returns:
        pd.DataFrame: A dataframe with three columns:
            - 'label': The unique keyword.
            - 'num_annotators': The number of distinct annotator classes the keyword appears in.
            - 'embedding': The embedding for the keyword.
    """
    keyword_dataframe = keyword_dataframe[
        keyword_dataframe["num_annotators"] > 1
    ].copy()
    embeddings = model.encode(
        keyword_dataframe["label"].tolist(),
        show_progress_bar=True,
        convert_to_tensor=False,
    )
    
    logger.info("Generated embeddings for %s keywords", len(embeddings))
    keyword_dataframe["embedding"] = embeddings.tolist()

    # convert the dataframe to a pyarrow table
    keyword_dataframe = pa.Table.from_pandas(keyword_dataframe, schema=pa.schema([
        ("label", pa.string()),
        ("embedding", pa.list_(pa.float32()))
    ]))

    return keyword_dataframe
