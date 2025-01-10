import logging
import string
from typing import Dict, List
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger(__name__)


def aggregate_keyword_annotators(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the number of distinct annotator classes each keyword appears in and the list of project IDs.

    Args:
        dataframes (pd.DataFrame): The dataframes to process.

    Returns:
        pd.DataFrame: A dataframe with three columns:
            - 'label': The unique keyword.
            - 'num_annotators': The number of distinct annotator classes the keyword appears in.
            - 'project_ids': The list of project IDs in which the keyword appears.
    """
    keyword_to_info = {}

    for df in dataframes:
        logger.info("Processing dataframe with columns: %s", df.columns)
        keyword_column = next(col for col in df.columns if "_keywords" in col)
        annotator_class = keyword_column.split("_")[0]

        # explode the keywords and drop NaN values
        exploded = df.explode(keyword_column).dropna(subset=[keyword_column])

        # preprocess the keywords
        exploded[keyword_column] = _preprocess_keywords(exploded[keyword_column])

        # aggregate the keywords with annotator classes and project IDs
        for keyword, project_id in zip(
            exploded[keyword_column], exploded["project_id"]
        ):
            if keyword not in keyword_to_info:
                keyword_to_info[keyword] = {"annotators": set(), "project_ids": set()}
            keyword_to_info[keyword]["annotators"].add(annotator_class)
            keyword_to_info[keyword]["project_ids"].add(project_id)

    # prepare the output dataframe
    output_df = pd.DataFrame(
        {
            "label": keyword_to_info.keys(),
            "num_annotators": [
                len(info["annotators"]) for info in keyword_to_info.values()
            ],
            "project_ids": [
                list(info["project_ids"]) for info in keyword_to_info.values()
            ],
        }
    )

    # sort the dataframe by num_annotators in descending order, then by label alphabetically
    return output_df.sort_values(
        by=["num_annotators", "label"], ascending=[False, True]
    )


def _preprocess_keywords(keywords: pd.Series) -> pd.Series:
    """
    Preprocess the keywords by lowercasing, removing trailing spaces,
    punctuation, stopwords, and stemming
    """
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def preprocess(text):
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return " ".join(words)

    return keywords.apply(preprocess)


def generate_embeddings(keyword_dataframe: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Generate embeddings for each keyword in the dataframe.

    Args:
        keyword_dataframe: The dataframe containing the keywords.

    Returns:
        Dict[str, List[float]]: A dictionary with keywords as keys and their embeddings as values.
    """
    # keyword_dataframe = keyword_dataframe[keyword_dataframe["num_annotators"] > 1].copy()
    embeddings = model.encode(
        keyword_dataframe["label"].tolist(),
        show_progress_bar=True,
        convert_to_tensor=False,
    )

    logger.info("Generated embeddings for %s keywords", len(embeddings))
    keyword_dataframe["embedding"] = embeddings.tolist()

    # create a dictionary with keywords as keys and embeddings as values
    keyword_embeddings = {
        keyword: embedding
        for keyword, embedding in zip(
            keyword_dataframe["label"], keyword_dataframe["embedding"]
        )
    }

    return keyword_embeddings
