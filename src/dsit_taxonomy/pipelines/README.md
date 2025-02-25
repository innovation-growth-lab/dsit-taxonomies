# Project Pipelines

This directory contains the data processing pipelines that form the core taxonomy classification workflow.

## Pipeline Structure

### Data Collection
- **data_collection_gtr/** - Fetches project data from Gateway to Research (GtR)
  - Collects project metadata, descriptions, and abstracts
  - Handles API pagination and rate limiting
  - Stores raw data in standardised format

### Data Processing
- **data_processing_taxonomies/** - Processes taxonomy data
  - Loads and standardises taxonomy hierarchies
  - Prepares taxonomy labels for matching
  - Generates embeddings for taxonomy terms

### Keyword Processing
- **keyword_processing_gtr/** - Extracts and processes keywords
  - Identifies key terms from project descriptions
  - Processes research topics and themes
  - Generates keyword embeddings

### Project Similarity
- **project_similarity_matching/** - Core matching pipeline
  - Computes similarity between projects and taxonomy labels
  - Generates sentence-level matches
  - Produces initial confidence scores

- **project_similarity_refinement/** - Refines initial matches
  - Applies zero-shot classification
  - Combines different matching approaches
  - Implements confidence thresholds

- **project_similarity_tune_and_validate/** - Tunes and validates results
  - Optimises matching parameters
  - Validates against manual classifications
  - Generates performance metrics

## Pipeline Dependencies

The pipelines are designed to run in sequence:
1. Data collection must complete before processing
2. Taxonomy and keyword processing can run in parallel
3. Matching requires processed data and keywords
4. Refinement and validation depend on initial matches

## Running Pipelines

Run specific pipelines using:
```bash
# Data collection
kedro run --pipeline data_collection_gtr

# Processing
kedro run --pipeline data_processing_taxonomies
kedro run --pipeline keyword_processing_gtr

# Matching and refinement
kedro run --pipeline project_similarity_matching
kedro run --pipeline project_similarity_refinement
kedro run --pipeline project_similarity_tune_and_validate
```

For detailed information about each pipeline, see the README files in their respective directories.

---