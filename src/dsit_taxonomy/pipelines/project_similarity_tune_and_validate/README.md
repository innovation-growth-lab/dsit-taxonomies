# Taxonomy matching tuning and validation pipeline

This pipeline fine-tunes the taxonomy matching algorithm and validates its performance using expert LLM evaluations. The key innovation is using two independent evaluation approaches to establish both algorithm-agnostic "true" labels and assess algorithm-proposed labels.

## Methodology

### 1. Expert Label Generation
Uses GPT-4 with retrieval-augmented generation to independently identify relevant taxonomy labels:

- **Input**: Project descriptions + retrieved taxonomy context
- **Process**:
  - RAG setup retrieves top-20 candidate labels
  - LLM evaluates each candidate's relevance
  - Assigns confidence levels (high/medium/low)
- **Output**: JSON array of `{label, likelihood}` pairs

Example prompt structure:
```yaml
system: |
  You are an expert research taxonomy analyst...
  CONFIDENCE LEVELS:
  - HIGH: Perfect match with project's core focus
  - MEDIUM: Related or partially overlapping field
  - LOW: Tangential connection
```

### 2. Algorithm Result Assessment
Evaluates labels proposed by our matching algorithm:

- **Input**: Project + algorithm's high-confidence labels
- **Process**: 
  - LLM evaluates each proposed label
  - Provides binary classification (true/false positive)
  - Includes explanation for decision
- **Output**: JSON array of `{label_id, positive, explanation}`

### 3. Parameter Tuning
Optimizes key parameters using grid search over:

1. **Similarity thresholds**:
   ```yaml
   sentence_threshold: [0.5, 0.55, 0.6]  # Best: 0.55
   global_threshold: [0.5, 0.55, 0.6]    # Best: 0.55
   keyword_threshold: [0.5, 0.55, 0.6]   # Best: 0.55
   ```

2. **Score weights**:
   ```yaml
   sentence_weight: [0.6, 0.7]  # Best: 0.6
   global_weight: [0.2, 0.3]    # Best: 0.3
   # keyword_weight is implicit: 1 - sentence - global
   ```

3. **Confidence binning**:
   ```yaml
   global_q2_threshold: [0.25]  # Medium confidence
   global_q3_threshold: [0.5]   # High confidence
   ```

4. **Other parameters**:
   ```yaml
   use_quantile: [False, True]           # Best: True
   normalise_by_matches: [False, True]   # Best: False
   ```

## Pipeline Components

### Nodes

1. `select_sample_projects`
   - Creates stratified sample (n=300)
   - Ensures representation across research areas
   - Uses fixed random seed for reproducibility

2. `get_expert_labels`
   - Generates algorithm-agnostic expert labels
   - Uses RAG-enhanced LLM evaluation
   - Returns confidence-scored assignments

3. `get_expert_assessment`
   - Evaluates algorithm-proposed labels
   - Provides binary classification
   - Includes explanatory feedback

4. `prepare_tuning_data`
   - Processes expert labels and assessments
   - Handles label mapping and cleaning
   - Prepares data for parameter tuning

5. `tune_matching_parameters`
   - Performs grid search
   - Computes performance metrics
   - Returns optimal parameters

6. `evaluate_scoring_quality`
   - Validates final performance
   - Computes precision, recall, F1
   - Analyzes different confidence thresholds

### Key Datasets

1. **Input Data**:
   - `gtr.projects.documents`: Project descriptions
   - `taxonomy.{name}.full.db`: Complete taxonomies

2. **Expert Labels**:
   - `gtr.projects.sample.expert_labels.{taxonomy}`: Independent expert labels
   - `gtr.projects.sample.expert_assessment.{taxonomy}`: Assessment of algorithm labels

3. **Results**:
   - `tuning.{taxonomy}.parameter_tuning_results`: Parameter optimization results
   - `validate.{taxonomy}.zeroshot_quality_metrics`: Final validation metrics

## Usage

### Running full pipeline
```bash
kedro run --pipeline project_similarity_tune_and_validate
```

### Running specific taxonomies
```bash
kedro run --pipeline project_similarity_tune_and_validate --tags cwts
kedro run --pipeline project_similarity_tune_and_validate --tags goscience
```

### Configuration
Parameters are defined in:
```yaml
conf/base/parameters_project_similarity_tune_and_validate.yml
```

## Dependencies
- **Core libraries**: pandas, numpy, transformers
- **LLM**: GPT-4 via OpenAI API
- **Embeddings**: text-embedding-3-small