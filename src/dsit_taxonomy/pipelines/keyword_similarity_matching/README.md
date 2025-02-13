# Taxonomy matching pipeline

This pipeline matches research projects to taxonomy labels using a hybrid approach combining sentence-level and keyword-level similarity scores. It uses vector similarity search with confidence binning to produce high/medium/low confidence matches.

## Features

### 1. Document Processing
- Project text is split into sentences and keywords
- Each component is embedded using a domain-adapted language model
- For a project $p$, we have:
  - Sentences $S_p = \{s_1, ..., s_n\}$ with embeddings $E(s_i)$
  - Keywords $K_p = \{k_1, ..., k_m\}$ with embeddings $E(k_j)$
  - Taxonomy labels $L = \{l_1, ..., l_t\}$ with embeddings $E(l_k)$

### 2. Similarity Computation
1. **Sentence-Level Similarity**
   - For each sentence $s_i$ and label $l_k$:
     $$sim(s_i, l_k) = \frac{E(s_i) \cdot E(l_k)}{\|E(s_i)\| \|E(l_k)\|}$$

2. **Keyword-Level Similarity**
   - Similar computations for each keyword $k_j$:
     $$sim(k_j, l_k) = \frac{E(k_j) \cdot E(l_k)}{\|E(k_j)\| \|E(l_k)\|}$$

### 3. Score Aggregation
1. **Sentence Scores**
   - For each project-label pair $(p, l_k)$, compute:
     - Mean similarity: $mean(p, l_k) = \frac{1}{n}\sum_{i=1}^n sim(s_i, l_k)$
     - Maximum similarity: $max(p, l_k) = \max_i sim(s_i, l_k)$
     - Match count: $count(p, l_k)$ = number of sentences matching label

   Combined score balancing frequency and strength:
   $$score(p, l_k) = \frac{1 + \log(1 + count(p, l_k))}{1 + \log(1 + n\_sentences)} \cdot mean(p, l_k) \cdot (1 + \log(1 + max(p, l_k)))$$
   where $n\_sentences$ is the total number of sentences in project $p$

   Filtering and normalisation:
   1. Project-level quantile filtering:
      $$S_{filtered}(p) = \{s : s > Q_q(S_{score}(p))\}$$
      where $Q_q(S_{score}(p))$ is the $q$-th quantile of scores within project $p$
   2. Per-project score normalisation:
      $$S_{final}(p, l_k) = \frac{S_{filtered}(p, l_k)}{\max_{l_j} S_{filtered}(p, l_j)}$$
      This ensures each project's top score is 1.0 while preserving relative strengths

2. **Keyword Scores**
   - For each keyword $k_j$, we have raw similarity scores with labels:
     $$sim(k_j, l_k) \text{ for } l_k \in L$$
   - Valid labels are restricted to those that passed sentence filtering:
     $$L_{valid}(p) = \{l_k : l_k \in S_{final}(p)\}$$
   - Keyword scores are then defined only on this subset:
     $$K_{score}(p, l_k) = \begin{cases}
       sim(k_j, l_k) & \text{if } l_k \in L_{valid}(p) \\
       \text{undefined} & \text{otherwise}
       \end{cases}$$
   This ensures keyword matches only reinforce labels that were relevant in sentence matching.

3. **Combined Score**
   - Weighted combination with configurable weights $\alpha$ and $\beta$:
     $$score(p, l_k) = ((\alpha \cdot S_{score}(p, l_k)) \cdot (\beta \cdot K_{score}(k_j, l_k)))^2$$
   - Filter by keyword similarity quantile threshold
   - Aggregate to project-label level:
     - Take maximum of relevance scores
     - Count unique matching keywords
     - Average entropy across matches

### 4. Confidence Binning
1. **Global Thresholding**
   - Compute quantiles $Q_2$ and $Q_3$ across all scores
   - Assign initial bins:
     ```
     if score > Q_3:      high
     if Q_2 < score ≤ Q_3: medium
     if score ≤ Q_2:      low
     ```

2. **Local Thresholding**
   - For each project $p$:
     - Sort scores in descending order: $score(p, l_{(1)}) \geq ... \geq score(p, l_{(t)})$
     - Compute relative gaps:
       $$gap_i = \frac{score(p, l_{(i)}) - score(p, l_{(i+1)})}{score(p, l_{(i)})}$$
     - Find two largest gaps at positions $i^*$ and $j^*$
     - Assign local bins:
       ```
       if i ≤ i*:         high
       if i* < i ≤ j*:    medium
       if i > j*:         low
       ```

3. **Final Assignment**
   - Take minimum confidence between global and local bins:
     ```
     final_bin = min(global_bin, local_bin)
     ```
   - This ensures conservative confidence assignment

### 5. Key Features
- Dual-level matching captures both context and specifics
- Local binning accounts for project-specific score distributions
- Conservative binning reduces false positives

## Nodes Overview

1. `document_preprocessing`
   Splits project descriptions into sentences and prepares them for similarity matching.

2. `compute_similarities_and_entropy`
   Computes cosine similarities between input vectors (sentences or keywords) and taxonomy labels.

3. `aggregate_sentence_matches`
   Filters and combines sentence-level matches using global and project-relative thresholds.

4. `combine_sentence_and_keyword_scores`
   Combines sentence and keyword similarity scores using configurable weights.

5. `add_metadata`
   Adds taxonomy hierarchy information and other metadata to matched labels.

6. `aggregate_scores_to_labels`
   Aggregates keyword-level scores to project-label level with boosting and assigns confidence levels.

## Configuration

### Score Aggregation
- `min_score_quantile`: Project-level minimum score quantile threshold
- `score_weights`: Sentence vs keyword importance
- `similarity_quantile_threshold`: Minimum similarity threshold

### Confidence Binning
- `global_quantile_threshold`: Global similarity quantile threshold
- `local_quantile_threshold`: Local similarity quantile threshold
- `global_bin_threshold`: Global bin threshold
- `local_bin_threshold`: Local bin threshold