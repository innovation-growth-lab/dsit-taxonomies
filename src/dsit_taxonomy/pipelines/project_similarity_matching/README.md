# Project similarity matching pipeline

The **Project similarity matching pipeline** performs semantic similarity matching between research projects and taxonomy labels using a multi-level approach. It combines evidence from project-level, sentence-level, and keyword-level matches to produce granular similarity scores.

## Features
- Multi-level similarity matching:
  - Global project-level matching
  - Sentence-level granular matching
  - Keyword-level term matching
- Weighted score combination
- Pruning of low-confidence matches
- Metadata enrichment for traceability

## Pipeline components

### Nodes

1. **`document_preprocessing`**
   - Combines project text fields
   - Splits into sentences
   - Generates unique identifiers
   - Removes duplicate sentences

2. **`compute_similarities`**
   - Vectorised batch processing
   - Cosine similarity computation
   - Top-N match selection
   - Configurable batch sizes

3. **`add_metadata`**
   - Enriches matches with metadata
   - Links to source projects
   - Adds taxonomy information
   - Preserves match provenance

4. **`prune_raw_matches`**
   - Filters low-scoring matches
   - Supports quantile thresholding
   - Handles multiple match types
   - Reports pruning statistics

5. **`combine_scores`**
   - Weighted score combination
   - Global match boosting
   - Keyword match boosting
   - Granular score computation

## Implementation details

### Similarity computation
1. **Document preparation**
   - Combine title, abstracts, impact text
   - Split into sentences using spaCy
   - Generate UUIDs for tracking
   - Remove duplicate sentences

2. **Batch processing**
   - Process documents in configurable batches
   - Parallel similarity computation
   - Memory-efficient processing
   - Progress tracking

3. **Score computation**
   ```python
   similarity = (doc_embeddings @ taxonomy_embeddings.T) / (doc_norms @ tax_norms)
   ```

### Score combination
1. **Base sentence scores**
   - Raw similarity scores from sentence matching
   - Project-level aggregation
   - Sentence-level granularity

2. **Global boost**
   - Project-level match scores
   - Weighted by configurable alpha
   - Reinforces consistent matches

3. **Keyword boost**
   - Maximum similarity across project keywords
   - Weighted by (1-alpha-beta)
   - Supports term-level evidence

4. **Final score**
   ```python
   final_score = (
       sentence_weight * sentence_score +
       global_weight * global_score +
       (1 - sentence_weight - global_weight) * keyword_score
   )
   ```

## Matching process

### 1. Input preparation
For each project $p$, we process:
- Full project text as a single document
- Individual sentences $S_p = \{s_1, \dots, s_n\}$
- Project keywords $K_p = \{k_1, \dots, k_m\}$

Each taxonomy provides labels $L = \{l_1, \dots, l_t\}$ with their embeddings $E(l_k)$.

### 2. Multi-level matching
1. **Global project matching**
   - For each project $p$, find top-50 taxonomy labels:
     $$C_g(p) = \{\,l_1, \dots, l_{50}\}$$
   - Each match has similarity score $\mathrm{sim}_g(p, l_k)$

2. **Sentence-level matching**
   - For each sentence $s_i$, find top-5 labels:
     $$C_s(s_i) = \{\,l_1, \dots, l_5\}$$
   - Compute cosine similarity:
     $$\mathrm{sim}_s(s_i, l_k) = \frac{E(s_i) \cdot E(l_k)}{\|E(s_i)\| \|E(l_k)\|}$$

3. **Keyword matching**
   - For each keyword $k_j$, find top-10 labels:
     $$C_k(k_j) = \{\,l_1, \dots, l_{10}\}$$
   - With similarity scores $\mathrm{sim}_k(k_j, l_k)$

### 3. Score pruning
For each match type (global, sentence, keyword):
1. **Quantile-based pruning**
   - Compute score threshold $t_q$ at quantile $q$:
     $$t_q = Q_q(\{\mathrm{sim}(x, l_k)\})$$
   - Keep only matches above threshold:
     $$M_{pruned} = \{(x, l_k) : \mathrm{sim}(x, l_k) > t_q\}$$

2. **Direct threshold pruning**
   - Use fixed threshold $t$:
     $$M_{pruned} = \{(x, l_k) : \mathrm{sim}(x, l_k) > t\}$$

### 4. Score combination
For each project-label pair $(p, l_k)$:

1. **Base sentence score**
   $$\mathrm{score}_s(p, l_k) = \max_{s_i \in S_p} \mathrm{sim}_s(s_i, l_k)$$

2. **Global boost**
   $$\mathrm{boost}_g(p, l_k) = \alpha \cdot \mathrm{sim}_g(p, l_k)$$
   where $\alpha$ is the global weight (default 0.3)

3. **Keyword boost**
   $$\mathrm{boost}_k(p, l_k) = \beta \cdot \max_{k_j \in K_p} \mathrm{sim}_k(k_j, l_k)$$
   where $\beta = 1 - \alpha - \gamma$ and $\gamma$ is the sentence weight (default 0.6)

4. **Final score**
   $$\mathrm{score}_{final}(p, l_k) = \gamma \cdot \mathrm{score}_s(p, l_k) + \mathrm{boost}_g(p, l_k) + \mathrm{boost}_k(p, l_k)$$

This approach:
- Preserves granular sentence-level evidence
- Rewards consistency across levels
- Balances different types of matches
- Allows fine-tuning via weights

## Usage

### Running the full pipeline
```bash
kedro run --pipeline project_similarity_matching
```

### Running for specific taxonomies
```bash
kedro run --pipeline project_similarity_matching --tags cwts
kedro run --pipeline project_similarity_matching --tags goscience
kedro run --pipeline project_similarity_matching --tags oa_concepts
```

## Data flow

### Inputs
```yaml
# Project documents
gtr.projects.documents:
  type: pandas.ParquetDataset
  filepath: .../projects/documents.parquet

# Taxonomy data
taxonomy.{taxonomy_name}.full.db:
  type: pandas.ParquetDataset
  filepath: .../taxonomies/{taxonomy_name}.parquet
```

### Outputs
```yaml
# Raw matches
{group}.gtr_data.{taxonomy_name}_matches.raw:
  type: pandas.ParquetDataset
  filepath: .../matches/{group}_{taxonomy_name}_raw.parquet

# Intermediate matches
{group}.gtr_data.{taxonomy_name}_matches.intermediate:
  type: pandas.ParquetDataset
  filepath: .../matches/{group}_{taxonomy_name}_intermediate.parquet

# Pruned matches
{group}.gtr_data.{taxonomy_name}_matches.pruned:
  type: pandas.ParquetDataset
  filepath: .../matches/{group}_{taxonomy_name}_pruned.parquet

# Final scores
projects.gtr_data.{taxonomy_name}_scores.granular:
  type: pandas.ParquetDataset
  filepath: .../scores/{taxonomy_name}_granular.parquet
```

## Configuration

### Batch processing
```yaml
similarity_matching:
  sentences:
    batch_size: 1000
    top_n: 5
  keywords:
    batch_size: 1000
    top_n: 10
  projects:
    batch_size: 1000
    top_n: 50
```

### Score weights
```yaml
score_weights:
  sentence_weight: 0.6  # alpha
  global_weight: 0.3    # beta
  # keyword_weight: 0.1 (implicit 1-alpha-beta)
```

### Pruning thresholds
```yaml
pruning:
  use_quantile: true
  sentence_threshold: 0.6
  global_threshold: 0.6
  keyword_threshold: 0.6
```

## Dependencies
- **Core libraries**:
  - `pandas`
  - `numpy`
  - `spacy`
  - `joblib`
- **Models**: English language model with sentencizer