"""
This pipeline performs semantic similarity matching between research projects
and taxonomy labels at multiple levels.

The pipeline consists of three main steps:
1. Document Preprocessing
   - Combines text fields from research projects
   - Splits documents into sentences
   - Generates unique IDs for tracking

2. Similarity Computation
   - Computes similarity scores between texts and taxonomy labels
   - Processes at three levels:
     * Global project-level matching
     * Sentence-level matching for granular evidence
     * Keyword-level matching for term overlap

3. Score Combination
   - Combines scores from different matching approaches
   - Applies weights to balance different evidence types
   - Produces granular scores for each (project, label) pair

Dependencies:
    - pandas
    - numpy
    - spacy
    - joblib
    - torch

Example:
    Run the complete matching pipeline:
    ```
    kedro run --pipeline project_similarity_matching
    ```
    Or run specific steps:
    ```
    kedro run --pipeline project_similarity_matching --nodes document_preprocessing
    ```
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    document_preprocessing,
    compute_similarities,
    combine_scores,
    add_metadata,
    prune_raw_matches,
)


def create_pipeline(**kwargs) -> Pipeline:  # pylint: disable=C0116,W0613
    doc_preprocess_pipeline = pipeline(
        [
            node(
                func=document_preprocessing,
                inputs="gtr.projects.documents",
                outputs=["projects.gtr_data.db", "sentences.gtr_data.db"],
                name="document_preprocessing",
                tags=["matching"],
            )
        ]
    )

    def compute_raw_matches_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                node(
                    func=compute_similarities,
                    inputs={
                        "taxonomy": f"taxonomy.{taxonomy_name}.full.db",
                        "documents": "projects.gtr_data.db",
                        "batch_size": "params:similarity_matching.projects.batch_size",
                        "top_n": "params:similarity_matching.projects.top_n",
                        "n_jobs": "params:similarity_matching.n_jobs",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_matches.raw",
                    name=f"compute_global_matches_{taxonomy_name}",
                ),
                node(
                    func=compute_similarities,
                    inputs={
                        "taxonomy": f"taxonomy.{taxonomy_name}.full.db",
                        "documents": "sentences.gtr_data.db",
                        "batch_size": "params:similarity_matching.sentences.batch_size",
                        "top_n": "params:similarity_matching.sentences.top_n",
                        "n_jobs": "params:similarity_matching.n_jobs",
                    },
                    outputs=f"sentences.gtr_data.{taxonomy_name}_matches.raw",
                    name=f"compute_sentence_matches_{taxonomy_name}",
                ),
                node(
                    func=compute_similarities,
                    inputs={
                        "taxonomy": f"taxonomy.{taxonomy_name}.full.db",
                        "documents": "keywords.gtr_data.db",
                        "batch_size": "params:similarity_matching.keywords.batch_size",
                        "top_n": "params:similarity_matching.keywords.top_n",
                        "n_jobs": "params:similarity_matching.n_jobs",
                    },
                    outputs=f"keywords.gtr_data.{taxonomy_name}_matches.raw",
                    name=f"compute_keyword_matches_{taxonomy_name}",
                ),
            ],
            tags=[taxonomy_name, "matching"],
        )

    def scoring_pipeline(taxonomy_name: str) -> Pipeline:
        return pipeline(
            [
                node(
                    func=add_metadata,
                    inputs={
                        "sentence_scores": f"sentences.gtr_data.{taxonomy_name}_matches.raw",
                        "keyword_scores": f"keywords.gtr_data.{taxonomy_name}_matches.raw",
                        "sentence_db": "sentences.gtr_data.db",
                        "keyword_db": "keywords.gtr_data.db",
                        "taxonomy": f"taxonomy.{taxonomy_name}.full.db",
                    },
                    outputs=[
                        f"sentences.gtr_data.{taxonomy_name}_matches.intermediate",
                        f"keywords.gtr_data.{taxonomy_name}_matches.intermediate",
                    ],
                    name=f"add_metadata_{taxonomy_name}",
                ),
                node(
                    func=prune_raw_matches,
                    inputs={
                        "sentence_matches": f"sentences.gtr_data.{taxonomy_name}_matches.intermediate",
                        "global_matches": f"projects.gtr_data.{taxonomy_name}_matches.raw",
                        "keyword_matches": f"keywords.gtr_data.{taxonomy_name}_matches.intermediate",
                        "sentence_threshold": "params:similarity_matching.pruning.sentence_threshold",
                        "global_threshold": "params:similarity_matching.pruning.global_threshold",
                        "keyword_threshold": "params:similarity_matching.pruning.keyword_threshold",
                        "use_quantile": "params:similarity_matching.pruning.use_quantile",
                    },
                    outputs=[
                        f"sentences.gtr_data.{taxonomy_name}_matches.pruned",
                        f"projects.gtr_data.{taxonomy_name}_matches.pruned",
                        f"keywords.gtr_data.{taxonomy_name}_matches.pruned",
                    ],
                    name=f"prune_raw_matches_{taxonomy_name}",
                ),
                node(
                    func=combine_scores,
                    inputs={
                        "sentence_scores": f"sentences.gtr_data.{taxonomy_name}_matches.pruned",
                        "keyword_scores": f"keywords.gtr_data.{taxonomy_name}_matches.pruned",
                        "global_scores": f"projects.gtr_data.{taxonomy_name}_matches.pruned",
                        "sentence_weight": "params:similarity_matching.score_weights.sentence_weight",
                        "global_weight": "params:similarity_matching.score_weights.global_weight",
                    },
                    outputs=f"projects.gtr_data.{taxonomy_name}_scores.granular",
                    name=f"combine_scores_{taxonomy_name}",
                ),
            ],
            tags=[taxonomy_name, "matching"],
        )

    return (
        doc_preprocess_pipeline
        + sum(
            compute_raw_matches_pipeline(tax)
            for tax in ["cwts", "goscience", "oa_concepts"]
        )
        + sum(scoring_pipeline(tax) for tax in ["cwts", "goscience", "oa_concepts"])
    )
