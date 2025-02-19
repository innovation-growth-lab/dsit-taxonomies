# Data annotation GtR pipeline

The **Data annotation GtR pipeline** extracts keywords and concepts from Gateway to Research (GtR) project descriptions using multiple annotation methods. This pipeline is designed for incremental processing, using oracle catalogs to track and skip previously processed projects.

## Features
- Multiple annotation methods:
  - **DBpedia Spotlight**: Links text to knowledge base concepts
  - **RAKE**: Statistical keyword extraction using word co-occurrence
  - **YAKE**: Unsupervised keyword extraction with text features
  - **KeyBERT**: Transformer-based semantic keyword extraction
- Incremental processing with oracle tracking
- Parallel processing for performance
- Batched processing for memory efficiency

## Pipeline components

### Nodes
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

## Incremental processing

### Oracle catalogs and Kedro's pipeline resolution
The pipeline uses oracle catalogs to enable incremental processing, working around Kedro's default pipeline resolution behavior. Here's how:

Kedro normally resolves datasets in a forward-only manner - a node can't know about the state of its outputs before running. This makes incremental processing challenging, as you can't easily skip already-processed items.

The oracle catalog solves this by:
1. Providing a "peek" at the output state before processing
2. Using a custom dataset type (`DefaultableParquetDataset`) that returns an empty DataFrame if the file doesn't exist
3. Allowing nodes to compare input data against previously processed items

Example oracle catalog:
```yaml
dbp.gtr_data.annotated.oracle:
  type: DefaultableParquetDataset
  filepath: .../annotations/dbp.parquet

rake.gtr_data.annotated.oracle:
  type: DefaultableParquetDataset
  filepath: .../annotations/rake.parquet
```

Each node uses its oracle catalog to:
1. Identify previously processed projects
2. Skip already annotated content
3. Process only new projects
4. Merge new results with existing annotations

### Forcing full reprocessing
To reprocess all projects (ignoring oracle catalogs), remove or rename the oracle files:
   ```bash
   aws s3 rm s3://igl-dsit-impact/.../annotations/dbp.parquet
   ```

## Usage

### Running the full pipeline
```bash
kedro run --pipeline data_annotation_gtr
```

### Running individual annotators
```bash
kedro run --pipeline data_annotation_gtr --nodes dbp_keywords
kedro run --pipeline data_annotation_gtr --nodes rake_keywords
kedro run --pipeline data_annotation_gtr --nodes yake_keywords
kedro run --pipeline data_annotation_gtr --nodes "keybert_keywords,concatenate_partitions"
```

## Dependencies
- **Core libraries**:
  - `keybert`
  - `rake-nltk`
  - `yake`
  - `spacy` (with `en_core_web_md` model)
  - `requests` (for DBpedia API)
- **API access**: DBpedia Spotlight API

## Data flow

### Inputs
- Project descriptions from:
  ```
  gtr.data_collection.projects.intermediate
  ```
- Oracle catalogs for each method:
  ```
  {method}.gtr_data.annotated.oracle
  ```

### Outputs
- Individual annotation results:
  ```
  dbp.gtr_data.annotated
  rake.gtr_data.annotated
  yake.gtr_data.annotated
  keybert.gtr_data.annotated
  ```
- Partitioned KeyBERT results:
  ```
  keybert.gtr_data.annotated.ptd
  ```

## Performance considerations
- DBpedia annotations are rate-limited
- KeyBERT processes in batches of 100 projects
- RAKE and YAKE run in parallel
- Oracle catalogs prevent redundant processing 