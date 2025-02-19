# Data processing taxonomies pipeline

The **Data processing taxonomies pipeline** standardises multiple research taxonomies into a consistent format for downstream matching. Each taxonomy has its own unique structure and requires custom preprocessing, but all are transformed into a unified hierarchical format.

## Features
- Custom preprocessing for each taxonomy source:
  - **CWTS**: Research topics from Leiden Ranking
  - **GO-SCIENCE**: UK government research areas
  - **OpenAlex**: Research concept ontology
- Standardised output format across all taxonomies
- Preservation of hierarchical relationships
- Generation of both full and dynamic bottom-level taxonomies

## Pipeline components

### Input taxonomies

1. **CWTS Topics**
   - Hierarchical structure with fixed levels:
     - Domain
     - Field
     - Subfield
     - Topic
   - Each level has name and ID columns

2. **GO-SCIENCE Areas**
   - Variable-depth hierarchy
   - Parent-child relationships
   - Rich metadata including definitions and examples
   - Non-monotonic level structure (e.g., L0 > L1 > L1)

3. **OpenAlex Concepts**
   - Complex network structure
   - Multiple parent relationships possible
   - Includes concept relationships and cross-references

### Standardised output format
All taxonomies are transformed into a consistent structure:

```python
{
    'label': 'Computer Science > Machine Learning > Deep Learning',
    'id_path': 'CS001 > ML042 > DL123',
    'level': 2,
    'uuid': '550e8400-e29b-41d4-a716-446655440000'
}
```

Two versions are generated for each taxonomy:
1. **Full hierarchy** (`taxonomy.{source}.full.db`)
   - Contains all levels of the hierarchy
   - Preserves complete paths
   - Includes intermediate nodes
   - Used in current implementation

2. **Dynamic bottom level** (`taxonomy.{source}.bottom.db`)
   - Contains only terminal nodes (leaves)
   - Accounts for varying depth across branches
   - Currently preserved but not used
   - Available for granular matching if needed
   - Note: Bottom-level terms may sometimes capture content less effectively than their parents

## Usage

### Running the full pipeline
```bash
kedro run --pipeline data_processing_taxonomies
```

### Processing individual taxonomies
```bash
kedro run --pipeline data_processing_taxonomies --nodes preprocess_cwts_topics
kedro run --pipeline data_processing_taxonomies --nodes preprocess_goscience_taxonomy
kedro run --pipeline data_processing_taxonomies --nodes preprocess_oa_concepts
```

## Data flow

### Inputs
```yaml
taxonomy.cwts.raw:
  type: pandas.ExcelDataset
  filepath: .../cwts_oa_topics.xlsx

taxonomy.goscience.raw:
  type: pandas.ExcelDataset
  filepath: .../go_science_topics.xlsx

taxonomy.oa_concepts.raw:
  type: pandas.ExcelDataset
  filepath: .../oa_concepts.xlsx
```

### Outputs
```yaml
# Full hierarchies (currently used)
taxonomy.cwts.full.db:
  type: pandas.ParquetDataset
  filepath: .../cwts_oa_topics.parquet

taxonomy.goscience.full.db:
  type: pandas.ParquetDataset
  filepath: .../go_science_topics.parquet

taxonomy.oa_concepts.full.db:
  type: pandas.ParquetDataset
  filepath: .../oa_concepts.parquet

# Dynamic bottom level (preserved for future use)
taxonomy.{source}.bottom.db:
  type: pandas.ParquetDataset
  filepath: .../{source}_bottom_level.parquet
```

## Implementation details

### Common processing steps
1. Extract hierarchical relationships
2. Build concatenated label paths
3. Generate unique IDs where needed
4. Create both full and dynamic bottom-level versions

### Taxonomy-specific handling

1. **CWTS Topics**
   - Fixed column names for each level
   - Direct path construction
   - Simple level counting
   - Dynamic terminal node identification

2. **GO-SCIENCE Areas**
   - Recursive path building
   - Parent-child relationship tracking
   - Non-monotonic level preservation
   - Variable-depth terminal nodes

3. **OpenAlex Concepts**
   - Multiple parent handling
   - Path explosion for network structure
   - Dynamic terminal node detection
   - Branch-specific depth handling

## Dependencies
- **Core libraries**:
  - `pandas`
  - `numpy`
  - `uuid`
- **File formats**: Excel, Parquet 