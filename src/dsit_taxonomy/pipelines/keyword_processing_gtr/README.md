# Keyword processing GtR pipeline

The **Keyword processing GtR pipeline** combines and processes keywords extracted by multiple methods into a unified set of project descriptors. It aggregates keywords based on extractor agreement and generates semantic embeddings for downstream matching.

## Features
- Keyword aggregation across multiple extractors:
  - DBpedia Spotlight
  - RAKE
  - YAKE
  - KeyBERT
- Consensus-based filtering
- Semantic embedding generation
- Project-keyword mapping preservation

## Pipeline components

### Nodes

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

### Processing steps

1. **Keyword aggregation**
   - Processes each extractor's output
   - Standardises keyword format
   - Tracks extractor agreement
   - Maintains project linkages
   - Filters based on consensus (>1 extractor)

2. **Embedding generation**
   - Converts keywords to vectors
   - Uses all-MiniLM-L6-v2 model
   - Generates 384-dimensional embeddings
   - Stores as float32 arrays

## Usage

### Running the full pipeline
```bash
kedro run --pipeline keyword_processing_gtr
```

### Running individual nodes
```bash
kedro run --pipeline keyword_processing_gtr --nodes aggregate_keyword_annotators
kedro run --pipeline keyword_processing_gtr --nodes generate_keyword_embeddings
```

## Data flow

### Inputs
```yaml
# Keyword extraction results
dbp.gtr_data.annotated:
  type: pandas.ParquetDataset
  filepath: .../annotations/dbp.parquet

rake.gtr_data.annotated:
  type: pandas.ParquetDataset
  filepath: .../annotations/rake.parquet

yake.gtr_data.annotated:
  type: pandas.ParquetDataset
  filepath: .../annotations/yake.parquet

keybert.gtr_data.annotated:
  type: pandas.ParquetDataset
  filepath: .../annotations/keybert.parquet
```

### Outputs
```yaml
# Aggregated keywords
keywords.gtr_data.db:
  type: pandas.ParquetDataset
  filepath: .../keywords/preprocessed.parquet

# Keyword embeddings
keywords.gtr_data.embeddings:
  type: pandas.ParquetDataset
  filepath: .../keywords/embeddings.parquet
```

## Output structure

### Aggregated keywords
```python
{
    'keyword': 'machine learning',
    'num_annotators': 3,
    'project_ids': ['PRJ123', 'PRJ456'],
    'uuid': '550e8400-e29b-41d4-a716-446655440000'
}
```

### Keyword embeddings
```python
{
    'keyword': 'machine learning',
    'embedding': [0.123, -0.456, ...] # 384-dimensional vector
}
```

## Implementation details

### Keyword aggregation
- Case-insensitive matching
- Whitespace normalisation
- Minimum 2-extractor agreement
- UUID generation for tracking
- Project linkage preservation

### Embedding generation
- Batched processing
- Progress tracking
- Memory-efficient storage
- float32 precision

## Dependencies
- **Core libraries**:
  - `pandas`
  - `numpy`
  - `sentence-transformers`
  - `uuid`
- **Models**: all-MiniLM-L6-v2 