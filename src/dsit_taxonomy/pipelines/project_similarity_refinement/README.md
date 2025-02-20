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

## Confidence Scoring Approaches

The pipeline now implements five different confidence scoring methods:

1. **Sentence-based (sentence_bin)**:
   - Original confidence from sentence-level matching
   - Based on score distributions and project-specific patterns
   - Good for detailed text matches

2. **Zero-shot (zeroshot_bin)**:
   - Pure NLI-based confidence
   - Independent from similarity scores
   - Strong at catching semantic mismatches

3. **Maximum confidence (max_confidence)**:
   - Takes highest confidence between sentence and zero-shot
   - Optimistic approach favouring any strong signal
   - Best for maximising recall

4. **Zero-shot favouring (zeroshot_favouring_confidence)**:
   - Favours zero-shot when large disagreement exists
   - Falls back to maximum when approaches agree
   - Prioritises precision over recall
   - Best for high-confidence assignments

5. **Sentence-favouring (sentence_favouring_confidence)**:
   - Trusts sentence-level scores when large disagreement exists
   - Uses maximum confidence when approaches agree
   - Useful when detailed text matching is critical

### Updated Validation Results

#### CWTS Taxonomy

| Approach          | Threshold | Precision | Recall | F1    |
|------------------|-----------|-----------|--------|-------|
| Sentence         | Strict    | 0.739     | 0.350  | 0.475 |
|                  | Relaxed   | 0.609     | 0.601  | 0.605 |
| Zero-shot        | Strict    | 0.847     | 0.545  | 0.663 |
|                  | Relaxed   | 0.784     | 0.661  | 0.717 |
| Maximum          | Strict    | 0.781     | 0.608  | 0.684 |
|                  | Relaxed   | 0.645     | 0.747  | 0.692 |
| Zero-shot favouring| Strict  | 0.843     | 0.569  | 0.679 |
|                  | Relaxed   | 0.765     | 0.691  | 0.726 |
| Sentence-favouring| Strict   | 0.765     | 0.430  | 0.550 |
|                  | Relaxed   | 0.608     | 0.661  | 0.634 |

#### GOScience Taxonomy

| Approach          | Threshold | Precision | Recall | F1    |
|------------------|-----------|-----------|--------|-------|
| Sentence         | Strict    | 0.416     | 0.435  | 0.425 |
|                  | Relaxed   | 0.308     | 0.680  | 0.424 |
| Zero-shot        | Strict    | 0.798     | 0.562  | 0.660 |
|                  | Relaxed   | 0.646     | 0.664  | 0.655 |
| Maximum          | Strict    | 0.533     | 0.640  | 0.582 |
|                  | Relaxed   | 0.350     | 0.760  | 0.479 |
| Zero-shot favouring| Strict  | 0.767     | 0.602  | 0.674 |
|                  | Relaxed   | 0.580     | 0.678  | 0.625 |
| Sentence-favouring| Strict   | 0.456     | 0.510  | 0.482 |
|                  | Relaxed   | 0.319     | 0.713  | 0.441 |

### Key Findings

1. **Best Overall Performance**:
   - CWTS: Zero-shot favouring approach (relaxed) achieves F1=0.726
   - GOScience: Zero-shot approach (relaxed) achieves F1=0.655

2. **Precision vs Recall Trade-offs**:
   - Zero-shot favouring approach maintains high precision while improving recall
   - Maximum confidence maximises recall but with precision cost
   - Zero-shot consistently provides best precision

3. **Taxonomy-Specific Patterns**:
   - CWTS benefits from combined approaches
   - GOScience performs best with pure zero-shot
   - Sentence-based methods struggle with GOScience

### Recommendations

1. **For CWTS Taxonomy**:
   - Use zero-shot favouring approach with relaxed threshold
   - Provides optimal balance (P=0.765, R=0.691, F1=0.726)
   - Maintains high precision while improving coverage

2. **For GOScience Taxonomy**:
   - Prefer zero-shot classification
   - Most reliable performance (P=0.646, R=0.664, F1=0.655)
   - More consistent across thresholds

## Usage

### Running the full pipeline```bash
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
