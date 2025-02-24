"""
Placeholder.
"""

import requests
from requests.adapters import HTTPAdapter, Retry
import spacy
from rake_nltk import Rake
import yake

# TMP: disable SSL warnings due to DBpedia API expired certificate
requests.packages.urllib3.disable_warnings()  # pylint: disable=no-member

nlp = spacy.load("en_core_web_md")


def preprocess_text(text: str) -> str:
    """
    Preprocesses the given text by removing stop words, punctuation,
        and keeping only nouns, proper nouns, and adjectives.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.

    """
    doc = nlp(text)

    # extract tokens and remove stop words, punctuation, and non-nouns
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.pos_ in {"NOUN", "PROPN", "ADJ"}
    ]

    # prune tokens that are GPE (location), PERSON, LOC
    bad_ents = [str(ent).lower() for ent in doc.ents if ent in {"GPE", "LOC", "PERSON"}]
    tokens = [token for token in tokens if token not in bad_ents]

    return " ".join(tokens)


def get_dbp_annotation(text: str, confidence: float = 0.5, support: int = 200) -> list:
    """
    Retrieves DBpedia annotations for the given text.

    Args:
        text (str): The input text to be annotated.
        confidence (float, optional): The confidence score threshold for
            DBpedia annotations. Defaults to 0.25.
        support (int, optional): The support threshold for DBpedia annotations.
            Defaults to 200.

    Returns:
        list: A list of unique annotations extracted from the DBpedia resources.

    """
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))

    base_url = "https://api.dbpedia-spotlight.org/en/annotate"
    headers = {"Accept": "application/json"}
    data = {
        "text": text,
        "confidence": confidence,
        "support": support,
        "policy": "whitelist",
    }

    response = session.post(base_url, data=data, headers=headers, verify=False)
    response_json = response.json()
    resources = response_json.get("Resources", [])
    annotations = [
        resource.get("@URI")
        .replace("http://dbpedia.org/resource/", "")
        .replace("_", " ")
        for resource in resources
    ]

    return list(set(annotations))

def get_rake_keywords(text: str) -> list:
    """
    Extracts keywords from the given text using RAKE.

    Args:
        text (str): The input text to extract keywords from.

    Returns:
        list: A list of keywords extracted from the text.

    """
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def get_yake_keywords(text: str) -> list:
    """
    Extracts keywords from the given text using YAKE.

    Args:
        text (str): The input text to extract keywords from.

    Returns:
        list: A list of keywords extracted from the text.

    """
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=20)
    keywords = kw_extractor.extract_keywords(text)
    return [keyword for keyword, _ in keywords]


def get_keybert_keywords(text: str, extractor: object) -> list:
    """
    Extracts keywords from the given text using KeyBERT.

    Args:
        text (str): The input text to extract keywords from.

    Returns:
        list: A list of keywords extracted from the text.

    """
    keywords = extractor.extract_keywords(text, keyphrase_ngram_range=(1, 3))
    return [keyword for keyword, _ in keywords]