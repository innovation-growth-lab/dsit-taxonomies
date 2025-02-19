# Project similarity refinement pipeline

The **Project similarity refinement pipeline** processes granular similarity scores into final taxonomy assignments with confidence levels. It combines evidence from multiple matching approaches and validates assignments using zero-shot classification.

## Features
- Multi-level score aggregation:
  - Sentence-level evidence
  - Global project matches
  - Keyword-based signals
- Dual-threshold confidence binning:
  - Global score distribution
  - Project-specific patterns
- Zero-shot validation
- Memory-efficient processing

## Pipeline components

### Nodes

1. **`aggregate_scores_to_labels`**
   - Combines granular matches into project-label pairs
   - Computes relevance scores with optional normalisation
   - Assigns initial confidence bins
   - Tracks matching evidence statistics

2. **`enhance_with_zeroshot`**
   - Validates assignments using NLI models
   - Processes in memory-efficient batches
   - Provides independent confidence scores
   - Enables cross-validation of matches

## Implementation details

### Score aggregation
1. **Base aggregation**
   - Maximum sentence-level score per label
   - Count of matching sentences
   - Project-level statistics

2. **Score normalisation (optional, currently disabled)**
   ```python
   normalised_score = raw_score * (matching_sentences / total_sentences)
   ```

3. **Confidence binning**
   - **Global thresholds**:
     $$
     bin(s) = \begin{cases}
       \text{high} & \text{if } s > Q_3 \\
       \text{medium} & \text{if } Q_2 < s \leq Q_3 \\
       \text{low} & \text{if } s \leq Q_2
     \end{cases}
     $$
     where $Q_2$, $Q_3$ are configurable quantiles (default 0.5, 0.75)

   - **Local thresholds**:
     - Project-specific quantiles
     - Relative score gap analysis
     - Conservative bin assignment

### Zero-shot validation
The pipeline uses natural language inference (NLI) to independently validate taxonomy assignments:

1. **Classification setup**
   - Converts each taxonomy label into a natural language hypothesis
   - Template: "This research project is about {label}"
   - Example: "This research project is about machine learning algorithms"
   - Processes projects in batches for efficiency

2. **NLI Model**
   - Uses ModernBERT-large-nli for textual entailment
   - Input: (project_text, hypothesis) pairs
   - Output: Probability of entailment
   - Batch size: 32 for memory efficiency

3. **Score computation**
   ```python
   confidence_bins = {
       "very high": score >= 0.9,  # Strong entailment
       "high": score >= 0.7,       # Clear entailment
       "medium": score >= 0.5,     # Possible entailment
       "low": score >= 0.25,       # Weak connection
       "very low": score < 0.25    # No clear connection
   }
   ```

4. **Validation process**
   - Independent from similarity scores
   - Considers full project context
   - Handles hierarchical label relationships
   - Provides complementary evidence

5. **Combined confidence**
   - Integrates similarity and NLI evidence
   - Uses maximum score approach
   - Enables cross-validation of matches
   - Helps filter spurious correlations

This approach provides several advantages:
- Independent validation mechanism
- Natural language understanding
- Explicit label interpretation
- Robust to vocabulary mismatches
- Handles complex relationships

## Validation Results

The pipeline has been validated against expert-labelled samples for two taxonomies (OpenAlex's CWTS topics and the GOScience technology taxonomy), comparing three approaches:
1. **Similarity**: Core embedding-based matching using sentence and project-level similarities
2. **Zero-shot**: Natural language inference using explicit label descriptions
3. **Combined**: Integration of both approaches

### Performance Metrics

#### CWTS Taxonomy

| Approach   | Threshold | Precision | Recall | F1    |
|------------|-----------|-----------|--------|-------|
| Similarity | Strict    | 0.74      | 0.35   | 0.48  |
|            | Relaxed   | 0.61      | 0.60   | 0.61  |
| Zero-shot  | Strict    | 0.85      | 0.55   | 0.66  |
|            | Relaxed   | 0.78      | 0.66   | 0.72  |
| Combined   | Strict    | 0.78      | 0.61   | 0.68  |
|            | Relaxed   | 0.75      | 0.69   | 0.72  |

#### GOScience Taxonomy

| Approach   | Threshold | Precision | Recall | F1    |
|------------|-----------|-----------|--------|-------|
| Similarity | Strict    | 0.42      | 0.44   | 0.43  |
|            | Relaxed   | 0.31      | 0.68   | 0.42  |
| Zero-shot  | Strict    | 0.80      | 0.56   | 0.66  |
|            | Relaxed   | 0.65      | 0.66   | 0.66  |
| Combined   | Strict    | 0.53      | 0.64   | 0.58  |
|            | Relaxed   | 0.50      | 0.70   | 0.58  |

**Note on validation methodology**: These metrics are derived from expert labels collected through language-model-assisted annotation, considering only high-confidence expert assignments as ground truth. The validation process:
- Uses expert-assessed true/false positives for precision
- Considers expert-suggested high-confidence labels for recall
- Applies both strict (high/very high) and relaxed (including medium) thresholds

### Analysis

The results demonstrate distinct patterns across approaches and taxonomies:

1. **Zero-shot performance**:
   - Consistently achieves the highest precision (80-85%)
   - Maintains stable performance across taxonomies
   - Provides the best overall F1 scores

2. **Similarity matching**:
   - Shows variable performance between taxonomies
   - Achieves good recall with relaxed thresholds
   - Struggles with precision, especially for GOScience

3. **Combined approach**:
   - Successfully improves recall over zero-shot
   - Trades some precision for better coverage
   - Benefits vary by taxonomy:
     * CWTS: Maintains F1 score while improving recall
     * GOScience: Recall gains don't offset precision loss

**Recommendation**: While zero-shot classification provides the most reliable performance, combining approaches may be beneficial in scenarios where recall is prioritized and some precision loss is acceptable. The choice between approaches should consider the specific taxonomy and use case requirements.

## Usage

### Running the full pipeline
```bash
kedro run --pipeline project_similarity_refinement
```

### Running for specific taxonomies
```bash
kedro run --pipeline project_similarity_refinement --tags cwts
kedro run --pipeline project_similarity_refinement --tags goscience
kedro run --pipeline project_similarity_refinement --tags oa_concepts
```

## Configuration

### Binning parameters
```yaml
binning:
  global_q2: 0.5   # Medium confidence threshold
  global_q3: 0.75  # High confidence threshold
  local_q2: 0.5    # Project-level medium threshold
  local_q3: 0.75   # Project-level high threshold
```

### Score normalisation
```yaml
normalise_by_matches: true  # Weight by matching sentence ratio
```

### Zero-shot settings
```yaml
zeroshot:
  batch_size: 32
  model_name: "tasksource/ModernBERT-large-nli"
```

## Dependencies
- **Core libraries**:
  - `pandas`
  - `numpy`
  - `transformers`
  - `torch`
- **Models**: ModernBERT-large-nli 