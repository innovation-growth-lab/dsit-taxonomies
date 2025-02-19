# Taxonomy matching parameter tuning pipeline

This pipeline fine-tunes the parameters of our taxonomy matching algorithm using expert LLM evaluations as ground truth. The key innovation is using two independent LLM evaluations to establish both algorithm-agnostic "true" labels and evaluate algorithm-proposed labels, enabling robust measurement of matching performance.

## Features

### Two-Stage Expert Evaluation
1. **Algorithm-Agnostic Expert Labels**
   - LLM evaluates projects independently of algorithm results
   - Uses RAG to access full taxonomy context
   - Assigns confidence levels (high/medium/low) to chosen labels
   - Provides baseline "true" labels unbiased by algorithm choices

2. **Algorithm Result Evaluation**
   - LLM evaluates labels proposed by matching algorithm
   - Binary classification (true/false positive) with explanations
   - Allows identification of algorithm errors and biases
   - Provides direct feedback on algorithm performance

### Performance Measurement
By combining both evaluations, we can identify:
- **True Positives**: Algorithm proposes high-confidence label that expert agrees with
- **False Positives**: Algorithm proposes high-confidence label that expert rejects
- **False Negatives**: Expert identifies high-confidence label that algorithm missed

This helps estimate precision, recall, and F1 scores for parameter tuning.

### Parameter Space
Key algorithmic parameters being tuned:
- **Weighting balance**: Relative importance of sentence-level vs keyword-level matching
- **Similarity thresholds**: Minimum required similarity scores
- **Confidence thresholds**: 
  - Global: Dataset-wide thresholds for similarity scores
  - Local: Project-specific thresholds accounting for relative label strengths

## Nodes Overview

1. `select_sample_projects`  
  Creates a stratified sample of projects for expert evaluation, ensuring representation across different research areas.
   
2. `get_expert_labels`  
  Generates algorithm-agnostic expert labels using RAG-enhanced LLM evaluation. Returns confidence-scored taxonomy assignments for each project.

3. `evaluate_algorithmic_assignments`  
  Evaluates algorithm-proposed labels using LLM, providing binary classification and explanations for each label.

4. `prepare_tuning_data`  
  Processes and aligns expert labels with algorithm evaluations, handling label mapping and data cleaning.

5. `tune_matching_parameters`  
  Performs grid search over parameter space, computing performance metrics for each combination. Returns both aggregate and per-project results.

## Key Datasets

### Raw Inputs:
- `gtr.projects.documents`: Project metadata including abstracts and descriptions
- `taxonomy.{name}.bottom.db`: Taxonomy labels and hierarchical structure

### Intermediate Outputs:
- `gtr.projects.sample`: Stratified sample of projects for evaluation
- `gtr.projects.sample.expert_labels.{taxonomy}`: Expert-assigned labels with confidence scores
- `gtr.projects.sample.expert_validation.{taxonomy}`: LLM evaluation of algorithm assignments

### Final Outputs:
- `validation.{taxonomy}.expert_labels.processed`: Processed expert labels
- `validation.{taxonomy}.scores.processed`: Processed algorithm evaluations
- `validation.{taxonomy}.parameter_tuning_results`: Parameter tuning results and metrics

## Configuration

The pipeline is configured through several YAML files:

### Sample Selection
- `sample.size`: Number of projects to evaluate
- `sample.random_state`: Random seed for reproducibility

### LLM Settings
- `llm.model`: GPT-4 model specification
- `llm.embedding_model`: Embedding model for RAG
- `llm.max_retries`: API retry attempts

### Expert Labeling
- `expert_labeling.retriever_k`: Number of labels to retrieve for RAG
- `expert_labeling.system_prompt`: System prompt for label generation
- `expert_labeling.question_prompt`: Project evaluation prompt

### Parameter Tuning
- `tuning.parameter_grid`: Grid of parameters to evaluate
  - Weighting parameters
  - Similarity thresholds
  - Confidence thresholds