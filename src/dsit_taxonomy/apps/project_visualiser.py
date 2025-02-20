"""
Streamlit app for visualising project taxonomy assignments with sentence-level highlighting.
"""

import os
from typing import Dict, List, Tuple
import colorsys
import streamlit as st
import numpy as np
import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Set wide layout
st.set_page_config(layout="wide")

# Disable Streamlit's file watcher to avoid PyTorch conflict (not working)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


@st.cache_data
def load_data():
    """Load required datasets using Kedro catalog."""
    bootstrap_project(project_path=".")
    with KedroSession.create() as session:
        catalog = session.load_context().catalog

        # Load final scores for all taxonomies
        cwts = catalog.load("projects.gtr_data.cwts_scores.final")
        goscience = catalog.load("projects.gtr_data.goscience_scores.final")
        oa_concepts = catalog.load("projects.gtr_data.oa_concepts_scores.final")

        # Load sentences and their scores
        sentences = catalog.load("sentences.gtr_data.db")
        sentences["sentence_id"] = sentences["uuid"]
        sentences_cwts = catalog.load("sentences.gtr_data.cwts_matches.pruned")
        sentences_goscience = catalog.load(
            "sentences.gtr_data.goscience_matches.pruned"
        )
        sentences_oa = catalog.load("sentences.gtr_data.oa_concepts_matches.pruned")

        # Load project metadata
        projects = catalog.load("gtr.projects.documents")

        return {
            "cwts": cwts,
            "goscience": goscience,
            "oa_concepts": oa_concepts,
            "sentences": sentences,
            "sentences_cwts": sentences_cwts,
            "sentences_goscience": sentences_goscience,
            "sentences_oa": sentences_oa,
            "projects": projects,
        }


@st.cache_data
def get_distinct_colors(n: int) -> List[str]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + np.random.random() * 0.3
        value = 0.8 + np.random.random() * 0.2
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})")
    return colors


@st.cache_data
def process_project_sentences(
    project_id: str,
    sentences_df: pd.DataFrame,
    sentences_scores: pd.DataFrame,
    taxonomy_scores: pd.DataFrame,
    projects_df: pd.DataFrame,
) -> Tuple[str, Dict[str, Dict], Dict[str, str], pd.DataFrame]:
    """
    Process sentences for a project and prepare for visualisation.
    """
    # Get project title from projects dataset
    project_title = projects_df[projects_df["project_id"] == project_id]["title"].iloc[
        0
    ]

    # Get project sentences
    project_sentences = sentences_df[sentences_df["project_id"] == project_id]

    # Get top label for each sentence
    top_scores = sentences_scores[
        sentences_scores["sentence_id"].isin(project_sentences["sentence_id"])
    ]
    top_scores = top_scores.loc[
        top_scores.groupby("sentence_id")["similarity_score"].idxmax()
    ]

    # Merge sentences with scores
    sentences_with_scores = pd.merge(
        project_sentences,
        top_scores[["sentence_id", "taxonomy_label_id", "similarity_score"]],
        on="sentence_id",
        how="left",
    )

    # Get unique labels that appear in sentences
    used_label_ids = sentences_with_scores["taxonomy_label_id"].dropna().unique()

    # Filter project labels to only those used in sentences
    project_labels = taxonomy_scores[
        (taxonomy_scores["project_id"] == project_id)
        & (taxonomy_scores["taxonomy_label_id"].isin(used_label_ids))
    ].sort_values("relevance_score", ascending=False)

    # Add global match indicator to project labels
    project_labels["is_global_match"] = project_labels[
        "similarity_score_global"
    ].notna()
    project_labels = project_labels.sort_values(
        ["is_global_match", "relevance_score"], ascending=[False, False]
    )

    # Generate colors for labels
    colors = get_distinct_colors(len(project_labels))
    label_colours = dict(zip(project_labels["taxonomy_label_id"], colors))

    # Create sentence to label mapping (simplified from previous version)
    sentence_scores = {}
    for _, row in sentences_with_scores.iterrows():
        sentence_scores[row["sentence_text"]] = {
            "label_id": row["taxonomy_label_id"] if pd.notna(row["taxonomy_label_id"]) else None
        }

    return project_title, sentence_scores, label_colours, project_labels


def render_colored_text(
    sentence_scores: Dict[str, Dict],
    label_colours: Dict[str, str],
    project_labels: pd.DataFrame,
) -> str:
    """Generate HTML for colored sentence visualisation with tooltips."""
    html = []

    # Create label lookup dictionary with all label info
    label_lookup = project_labels.set_index("taxonomy_label_id").to_dict("index")

    for sentence, info in sentence_scores.items():
        if info["label_id"] and info["label_id"] in label_colours:
            color = label_colours[info["label_id"]]
            rgba_color = color.replace("rgb", "rgba").replace(")", ", 0.15)")

            # Get label info for tooltip
            label_info = label_lookup.get(info["label_id"], {})
            label_text = label_info.get("taxonomy_label", "Unknown")

            # Get all scores
            sentence_score = label_info.get("similarity_score_sent", 0.0)
            global_score = label_info.get("similarity_score_global", "N/A")
            key_score = label_info.get("similarity_score_key", "N/A")
            
            tooltip = (
                f"{label_text}\n"
                f"Sentence match: {sentence_score:.3f}\n"
                f"Global match: {global_score if global_score == 'N/A' else f'{global_score:.3f}'}\n"
                f"Key match: {key_score if key_score == 'N/A' else f'{key_score:.3f}'}\n"
            )

            html.append(
                f'<span title="{tooltip}" style="background-color: {rgba_color};">{sentence}</span>'
            )
        else:
            html.append(sentence)
    return " ".join(html)


def confidence_level_to_rank(level: str) -> int:
    """Convert confidence level to numeric rank for sorting."""
    ranks = {"very high": 5, "high": 4, "medium": 3, "low": 2, "very low": 1}
    return ranks.get(level.lower(), 0)


def main():
    st.title("Project Taxonomy Visualiser")

    # Load data
    with st.spinner("Loading data..."):
        data = load_data()

    # Sidebar controls
    with st.sidebar:
        st.header("Search Controls")

        # Taxonomy selection
        taxonomy = st.selectbox(
            "Select Taxonomy",
            ["CWTS", "GOScience", "OpenAlex Concepts"],
            help="Choose which taxonomy to visualise",
        )

        # Map selection to data
        taxonomy_map = {
            "CWTS": ("cwts", data["cwts"], data["sentences_cwts"]),
            "GOScience": ("goscience", data["goscience"], data["sentences_goscience"]),
            "OpenAlex Concepts": (
                "oa_concepts",
                data["oa_concepts"],
                data["sentences_oa"],
            ),
        }

        _, tax_scores, tax_sentences = taxonomy_map[taxonomy]

        # Get all project titles
        project_titles = pd.merge(
            data["projects"][["project_id", "title"]],
            tax_scores[["project_id"]].drop_duplicates(),
            on="project_id",
        )

        st.markdown("---")

        # Project search
        st.subheader("Project Search")
        search_term = st.text_input(
            "Search by title",
            key="project_search",
            placeholder="Type to filter projects...",
        )

        filtered_titles = (
            project_titles[
                project_titles["title"].str.contains(search_term, case=False)
            ]
            if search_term
            else project_titles
        )

        # Show number of matches
        if search_term:
            st.caption(f"Found {len(filtered_titles)} matching projects")

        selected_title = st.selectbox(
            "Select a project",
            filtered_titles["title"].tolist(),
            key="project_select",
        )

        st.markdown("---")
        st.image("src/dsit_taxonomy/apps/igl_logo.png")

    # Main content area
    if selected_title:
        project_id = project_titles[project_titles["title"] == selected_title][
            "project_id"
        ].iloc[0]

        # Process project data
        title, sentence_scores, label_colours, project_labels = (
            process_project_sentences(
                project_id,
                data["sentences"],
                tax_sentences,
                tax_scores,
                data["projects"],
            )
        )

        # Split into text and tabs
        col1, col2 = st.columns([3, 2])

        with col1:
            st.header("Project Text")
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Project ID:** {project_id}")

            st.markdown("**Description:**")
            st.markdown(
                render_colored_text(sentence_scores, label_colours, project_labels),
                unsafe_allow_html=True,
            )

        with col2:
            # Create tabs for all information with tooltips
            matched_tab, basic_tab, confidence_tab = st.tabs(
                ["Matched Labels", "Basic Assignments", "Composite Assignments"]
            )

            with matched_tab:
                if label_colours:
                    legend_html = []
                    for _, row in project_labels.iterrows():
                        color = label_colours.get(row["taxonomy_label_id"], "gray")
                        rgba_color = color.replace("rgb", "rgba").replace(
                            ")", ", 0.15)"
                        )

                        # Format scores, showing N/A for missing values
                        global_score = row.get("similarity_score_global", "N/A")
                        key_score = row.get("similarity_score_key", "N/A")
                        relevance_score = row.get("relevance_score", "N/A")
                        global_text = (
                            f"{global_score:.2f}" if global_score != "N/A" else "N/A"
                        )
                        key_text = f"{key_score:.2f}" if key_score != "N/A" else "N/A"
                        relevance_text = (
                            f"{relevance_score:.2f}" if relevance_score != "N/A" else "N/A"
                        )

                        legend_html.append(
                            f'<div style="margin-bottom: 5px;">'
                            f'<span style="background-color: {rgba_color}; padding: 2px 5px; margin-right: 10px;">&nbsp;&nbsp;&nbsp;</span>'
                            f'{row["taxonomy_label"]}<br>'
                            f'<span style="font-size: 0.9em; margin-left: 25px;">'
                            f'<br>Sentence match: {row["similarity_score_sent"]:.2f}<br>'
                            f"Global match: {global_text}<br>"
                            f"Key match: {key_text}<br>"
                            f"<b>Relevance score: {relevance_text}</b>"
                            f"</span>"
                            f"</div><br>"
                        )
                    st.markdown("\n".join(legend_html), unsafe_allow_html=True)

            with basic_tab:
                st.markdown("""
                    **Basic assignments** show the initial confidence levels derived from:
                    - **Sentence-based**: Confidence based on matching specific sentences in the project text
                    - **Zero-shot**: Confidence from analysing the entire project text using zero-shot classification
                """)
                
                sent_tab, zero_tab = st.tabs(["Sentence-based", "Zero-shot"])

                with sent_tab:
                    st.markdown("""
                        *Confidence levels based on how well specific sentences match taxonomy labels. 
                        Uses sentence embeddings to detect semantic similarity.*
                    """)
                    sentence_assignments = project_labels[
                        project_labels["sentence_bin"].notna()
                    ].copy()
                    sentence_assignments["sort_rank"] = sentence_assignments[
                        "sentence_bin"
                    ].apply(confidence_level_to_rank)
                    sentence_assignments = sentence_assignments.sort_values(
                        "sort_rank", ascending=False
                    )

                    if not sentence_assignments.empty:
                        for _, row in sentence_assignments.iterrows():
                            st.markdown(
                                f"- **{row['taxonomy_label']}** ({row['sentence_bin']})"
                            )
                    else:
                        st.info("No sentence-based assignments")

                with zero_tab:
                    st.markdown("""
                        *Confidence levels from zero-shot classification of the entire project text.
                        Validates whether the project actually discusses each label.*
                    """)
                    zeroshot_assignments = project_labels[
                        project_labels["zeroshot_bin"].notna()
                    ].copy()
                    zeroshot_assignments["sort_rank"] = zeroshot_assignments[
                        "zeroshot_bin"
                    ].apply(confidence_level_to_rank)
                    zeroshot_assignments = zeroshot_assignments.sort_values(
                        "sort_rank", ascending=False
                    )

                    if not zeroshot_assignments.empty:
                        for _, row in zeroshot_assignments.iterrows():
                            st.markdown(
                                f"- **{row['taxonomy_label']}** ({row['zeroshot_bin']})"
                            )
                    else:
                        st.info("No zero-shot assignments")

            with confidence_tab:
                st.markdown("""
                    **Composite assignments** combine sentence-based and zero-shot scores in different ways:
                    - **Max Confidence**: Takes the highest confidence between sentence and zero-shot scores
                    - **Zero-shot favoured**: Favours zero-shot scores when there's significant disagreement
                    - **Sentence favoured**: Favours sentence-based scores when there's significant disagreement
                """)
                
                max_tab, cons_tab, sent_tab = st.tabs(
                    ["Max Confidence", "Zeroshot-favouring", "Sentence-favouring"]
                )

                with max_tab:
                    st.markdown("""
                        *Shows the highest confidence level between sentence-based and zero-shot scores.
                        Useful when you want to capture all potential matches.*
                    """)
                    conf_assignments = project_labels[
                        project_labels["max_confidence"].notna()
                    ].copy()
                    conf_assignments["sort_rank"] = conf_assignments[
                        "max_confidence"
                    ].apply(confidence_level_to_rank)
                    conf_assignments = conf_assignments.sort_values(
                        "sort_rank", ascending=False
                    )

                    if not conf_assignments.empty:
                        for _, row in conf_assignments.iterrows():
                            st.markdown(
                                f"- **{row['taxonomy_label']}** ({row['max_confidence']})"
                            )
                    else:
                        st.info("No max confidence assignments")

                with cons_tab:
                    st.markdown("""
                        *Uses zero-shot scores to validate sentence matches.
                        When scores disagree significantly, favours the zero-shot confidence.
                        Most reliable for avoiding false positives.*
                    """)
                    conf_assignments = project_labels[
                        project_labels["zeroshot_favouring_confidence"].notna()
                    ].copy()
                    conf_assignments["sort_rank"] = conf_assignments[
                        "zeroshot_favouring_confidence"
                    ].apply(confidence_level_to_rank)
                    conf_assignments = conf_assignments.sort_values(
                        "sort_rank", ascending=False
                    )

                    if not conf_assignments.empty:
                        for _, row in conf_assignments.iterrows():
                            st.markdown(
                                f"- **{row['taxonomy_label']}** ({row['zeroshot_favouring_confidence']})"
                            )
                    else:
                        st.info("No zero-shot favoured confidence assignments")

                with sent_tab:
                    st.markdown("""
                        *Prioritises sentence-level matches over sero-shot scores.
                        When scores disagree significantly, keeps the sentence-based confidence.
                        Best for capturing specific mentions even if the overall text is less relevant.*
                    """)
                    conf_assignments = project_labels[
                        project_labels["sentence_favouring_confidence"].notna()
                    ].copy()
                    conf_assignments["sort_rank"] = conf_assignments[
                        "sentence_favouring_confidence"
                    ].apply(confidence_level_to_rank)
                    conf_assignments = conf_assignments.sort_values(
                        "sort_rank", ascending=False
                    )

                    if not conf_assignments.empty:
                        for _, row in conf_assignments.iterrows():
                            st.markdown(
                                f"- **{row['taxonomy_label']}** ({row['sentence_favouring_confidence']})"
                            )
                    else:
                        st.info("No sentence-favouring assignments")
    else:
        st.info("👈 Select a project from the sidebar to begin")


if __name__ == "__main__":
    main()
