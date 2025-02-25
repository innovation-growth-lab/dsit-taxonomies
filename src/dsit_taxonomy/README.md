# Project Source Structure

This directory contains the core source code for the taxonomy classification project. The code is structured as a Kedro project, which provides a standardised way to build modular data pipelines.

The project processes research project data through multiple stages:
1. Collecting data from the Gateway to Research (GtR) API
2. Processing and analysing project descriptions
3. Matching projects against multiple taxonomies
4. Scoring and implementing decision rules for final label assignment

## Directories

- **apps/** - Streamlit applications for visualising and exploring the data
  - Contains an app for interactive exploration of taxonomy assignments

- **datasets/** - [Custom dataset](https://docs.kedro.org/en/stable/data/how_to_create_a_custom_dataset.html) definitions for Kedro
  - Includes data loaders and savers for data with a defaultable option to an empty dataset

- **pipelines/** - Data processing and analysis [pipelines](https://docs.kedro.org/en/stable/tutorial/create_a_pipeline.html)
  - Contains modular pipelines for each taxonomy
  - Includes data preparation, matching, and scoring logic

## Core Kedro Files

- **hooks.py** - Project-specific [Kedro hooks](https://docs.kedro.org/en/stable/hooks/introduction.html)
  - Computes vector embeddings after data loading and before node run 

- **settings.py** - Project [settings](https://docs.kedro.org/en/stable/kedro_project_setup/settings.html) and configuration
  - Defines project-wide settings, and declares use of hooks
  - Contains environment-specific configurations for the catalog

- **pipeline_registry.py** - [Pipeline registration](https://docs.kedro.org/en/stable/nodes_and_pipelines/pipeline_registry.html)
  - Registers all project pipelines
  - Defines pipeline dependencies and execution order

# Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Kedro commands:
```bash
# Run all pipelines
kedro run

# Run a specific pipeline
kedro run --pipeline data_collection_gtr

# Run a specific node
kedro run --node fetch_gtr_data

# Run nodes with a specific tag
kedro run --tag gtr
```

3. Launch the visualiser:
```bash
streamlit run src/dsit_taxonomy/apps/project_visualiser.py
```

For detailed information about specific components, see the README files in each directory. 