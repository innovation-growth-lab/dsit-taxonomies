# Keyword extraction and processing GtR pipeline

This pipeline extracts and processes keywords from Gateway to Research (GtR) project descriptions in two stages:

1. **Data annotation**: Extracts keywords using multiple methods
2. **Keyword processing**: Combines and processes extracted keywords for downstream matching

## Features

### Annotation features
- Multiple extraction methods:
  - **DBpedia Spotlight**: Links text to knowledge base concepts
  - **RAKE**: Statistical keyword extraction using word co-occurrence
  - **YAKE**: Unsupervised keyword extraction with text features
  - **KeyBERT**: Transformer-based semantic keyword extraction
- Incremental processing with oracle tracking
- Parallel processing for performance
- Batched processing for memory efficiency

### Processing features
- Keyword aggregation across extractors
- Consensus-based filtering
- Semantic embedding generation
- Project-keyword mapping preservation

## Pipeline components

### Annotation nodes
1. **`dbp_keywords`**
   - Links text to DBpedia concepts
   - Uses confidence and support thresholds
   - Handles API rate limiting and retries

2. **`rake_keywords`**
   - Statistical keyword extraction
   - Focuses on multi-word phrases
   - Fast processing for large datasets

3. **`yake_keywords`**
   - Feature-based keyword extraction
   - Handles domain-specific terminology
   - Configurable n-gram size

4. **`keybert_keywords`**
   - Semantic keyword extraction
   - Processes in memory-efficient batches
   - Includes timestamp-based partitioning

5. **`concatenate_partitions`**
   - Combines KeyBERT results
   - Handles parallel loading
   - Deduplicates results

### Processing nodes

1. **`aggregate_keyword_annotators`**
   - Combines keywords from all extractors
   - Counts extractor agreement for each keyword
   - Filters out single-extractor keywords
   - Maps keywords to source projects
   - Generates unique identifiers

2. **`generate_keyword_embeddings`**
   - Creates dense vector representations
   - Uses Sentence Transformers model
   - Enables semantic similarity matching
   - Optimises for memory efficiency

## Incremental processing

### Oracle catalogs and Kedro's pipeline resolution
The annotation pipeline uses oracle catalogs to enable incremental processing, working around Kedro's default pipeline resolution behaviour. Here's how:

Kedro normally resolves datasets in a forward-only manner - a node can't know about the state of its outputs before running. This makes incremental processing challenging, as you can't easily skip already-processed items.

The oracle catalog solves this by:
1. Providing a "peek" at the output state before processing
2. Using a custom dataset type (`DefaultableParquetDataset`) that returns an empty DataFrame if the file doesn't exist
3. Allowing nodes to compare input data against previously processed items

Example oracle catalog: