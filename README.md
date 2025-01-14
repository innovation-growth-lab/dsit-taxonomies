# A Taxonomy of Research Projects

## Enhancing the Mapping of Funded Projects onto Taxonomies

### Overview
This project aims to systematically classify all UKRI-funded research projects listed in the Gateway to Research (GtR) database. The primary objective is to map these projects onto two distinct, multi-level taxonomies: one focused on research topics and the other on conceptual frameworks. Both taxonomies feature hierarchical structures, allowing for both granular and broad categorisations of research projects. Our approach combines semantic analysis, machine learning, and validation techniques to ensure precise, reproducible, and comprehensive taxonomy assignments.

### Taxonomies in Focus
The project utilises two taxonomies:

1. **CWTS Leiden Taxonomy**: Developed in collaboration with OpenAlex, this taxonomy features 4,516 topics across 252 subfields, 26 fields, and 4 domains. It provides a comprehensive overview of the research landscape, derived from the abstracts on OpenAlex articles.

2. **OpenAlex Concepts Taxonomy**: This is an updated and modified version of the Microsoft Academic Graph taxonomy, containing approximately 65,000 abstract concepts across six hierarchical layers. It provides a granular view of research ideas and has been selected for its depth, open-source access, and utility for future data generation.

### Approach and Methodology

#### 1. Data Preparation and Semantic Embeddings
- **Entity Extraction**: Project abstracts from the GtR database will be processed to extract relevant concepts using DBpedia Spotlight, an entity annotator API that identifies Wikipedia entities related to project descriptions. This aligns closely with OpenAlex concepts.
- **Numeric Representations**: Extracted entities are transformed into numeric representations using SPECTER, a large language model optimised for academic content.

#### 2. Taxonomy Mapping via Semantic Distance Measurement
- **Semantic Distance Calculation**: The semantic distance between project descriptors and elements of the taxonomies will be measured using Euclidean space. This involves comparing the semantic embeddings of projects to the CWTS Leiden taxonomy and OpenAlex concepts.
- **Cosine Similarity**: Vectorised operations will be used to calculate cosine similarity values for project entities and clusters of DBpedia concepts. Retrieval-augmented generation (RAG) via OpenAI’s API and Langchain pipelines will generate expert associations for training cut-off values for similarity results.

#### 3. Supplementary Labelling and Machine Learning Approach
- **Rules-Based Approach**: Initial mapping will use a rules-based approach but may face uneven results due to assumptions and decision flaws in defining association rules.
- **Machine Learning Classification Models**: To address potential inconsistencies, a second approach will use data from publications associated with UKRI-funded projects. For older projects, data will be collected from OpenAlex to create a labeled dataset, which will be used to train multi-class, multi-label classification models to infer CWTS topics and OpenAlex concepts.
- **Validation and Fine-Tuning**: A hold-out set will be used to validate and fine-tune the models, avoiding overfitting.

#### 4. Integration and Validation
- **Dataset Integration**: Outputs from both methodologies will be integrated to assign accurate labels across multiple levels of the two taxonomies for each UKRI-funded project.
- **Consensus Ruling and Tiebreaking**: The RAG pipeline will be used to validate the combined approaches, enhance consensus ruling, and address tiebreaking.

# Requisites
nltk.download('punkt_tab')
python -m spacy download en_core_web_sm

# Todo list
- [] Control for continuation of other projcets, ie. REF (ID).
- [] Consider teachnical abstracts or impact summaries when these are substantially larger than abstracts.


### **Revised pipeline for assigning and validating taxonomy labels**

---

#### **Step 1: Input data preparation**
1. **Data inputs:**
   - Project abstracts or descriptions.
   - Extracted keywords (from DBPedia, RAKE, YAKE, KeyBERT, etc.).
   - Taxonomy labels, including hierarchical structure.

2. **Hierarchical concatenation of taxonomy labels:**
   - Transform the taxonomy into **hierarchically concatenated labels** for similarity scoring:
     - For each bottom-level node $ l_j $, concatenate its parent labels to create a hierarchical path:
       $$
       l_j = \text{"Parent > Child > Bottom Level"}
       $$
     - Example: `Physics > Quantum Mechanics > Quantum Optics`.

3. **Embeddings:**
   - Compute or retrieve precomputed embeddings for:
     - Keywords (from project abstracts).
     - Hierarchical taxonomy labels $ l_j $ (concatenated paths).
   - Use a pretrained model like OpenAI embeddings, SPECTER, or Sentence-BERT.

---

#### **Step 2a: Keyword-Taxonomy similarity**
1. **Keyword-Taxonomy similarity:**
   - For each keyword $ k_i $, compute its similarity to all concatenated hierarchical labels $ l_j $:
     $$
     S(k_i, l_j) = \text{cosine\_similarity}(\text{embedding}(k_i), \text{embedding}(l_j))
     $$
   - Store the similarity matrix for all projects.

---

#### **Step 2b: Document-Taxonomy similarity**
1. **Document embedding:**
   - Use pretrained models (start basic with all-MiniLM-L6-v2) to compute an embedding for the full project abstract $ d $.

2. **Document similarity:**
   - Compute similarity between the project document embedding $ d $ and each hierarchical label $ l_j $:
     $$
     S_{\text{document}}(d, l_j) = \text{cosine\_similarity}(\text{embedding}(d), \text{embedding}(l_j))
     $$

---

#### **Step 3: Entropy and weight calculation**
1. **Keyword entropy:**
   - For each keyword $ k_i $, calculate Shannon entropy $ H(k_i) $ across the concatenated hierarchical labels:
     $$
     H(k_i) = -\sum_{j=1}^{L} p_{ij} \log p_{ij}, \quad p_{ij} = \frac{\exp(S(k_i, l_j))}{\sum_{j=1}^{L} \exp(S(k_i, l_j))}
     $$

2. **Keyword weighting:**
   - Assign a weight $ w(k_i) $ to each keyword based on its entropy:
     $$
     w(k_i) = 1 - \frac{H(k_i)}{H_\text{max}}
     $$
   - Low-entropy keywords get higher weights, reflecting their stronger alignment to specific hierarchical labels.

---

#### **Step 4: Aggregate keyword scores to hierarchical labels**
1. **Label relevance scores:**
   - For each hierarchical taxonomy label $ l_j $, compute its relevance score $ R(l_j) $ as the weighted sum of similarity scores for all associated keywords:
     $$
     R(l_j) = \sum_{i=1}^{N} w(k_i) \cdot S(k_i, l_j)
     $$

2. **Weighted combination:**
   - Combine scores from the keyword-based and document-based methods to calculate a final relevance score for each label:
     $$
     R_{\text{final}}(l_j) = \alpha \cdot R_{\text{keywords}}(l_j) + (1 - \alpha) \cdot S_{\text{document}}(d, l_j)
     $$
     - $ \alpha $: Weight parameter to balance the influence of keywords vs. document embeddings.
     - Example: Start with $ \alpha = 0.5 $ and tune based on validation.

2. **Normalise combined scores:**
   - Normalise $ R_{\text{final}}(l_j) $ across all labels:
     $$
     R_{\text{normalised}}(l_j) = \frac{R_{\text{final}}(l_j)}{\max(R_{\text{final}}(l))}
     $$

---

#### **Step 5: Final label selection using relevance drop-Off**
1. **Sort Labels:**
   - Sort concatenated hierarchical labels $ l_j $ by $ R_{\text{normalised}}(l_j) $ in descending order.

2. **Find the drop-off point (Elbow):**
   - Compute differences between consecutive sorted scores:
     $$
     \Delta R_j = R_{\text{normalised}}(l_j) - R_{\text{normalised}}(l_{j+1})
     $$
   - Identify the largest $ \Delta R_j $, which indicates the "elbow" or sharp drop in relevance.

3. **Select labels:**
   - Include all hierarchical labels up to the elbow point in the sorted list.

---

#### **Step 6: OpenAI API validation**
1. **Define validation Level:**
   - Select a **sufficiently high level** of the taxonomy for validation (e.g., parent categories such as "Physics," "Engineering," "Biology").
   - Extract all parent-level labels from the taxonomy.

2. **Construct OpenAI API prompt:**
   - Provide the project abstract, top-ranked hierarchical labels from the previous step, and the list of high-level taxonomy labels.
   - Example prompt:
     ```
     Abstract: {Project Abstract}
     
     Based on the project abstract, select the most appropriate high-level taxonomy labels from the following list:
     - Physics
     - Biology
     - Engineering
     - Computer Science
     - Social Sciences
     - Medicine
     
     Provide a ranked list of the most relevant labels, and explain why they are appropriate.
     ```

3. **Query OpenAI API:**
   - Use the OpenAI API to generate the validation results:
     - High-level taxonomy labels with explanations.
     - Comparison of selected labels to ensure consistency with bottom-level assignments.

4. **Adjust and validate final labels:**
   - Use the OpenAI-generated high-level labels to validate the bottom-level selections:
     - If high-level labels conflict with bottom-level labels, re-examine similarity scores and rerun Step 5.

---

#### **Step 7: Validation using existing tags for outcomes associated with projects**

#### **Step 8: Output**
   - For each project, output:
     - Project ID.
     - Selected bottom-level taxonomy labels (with hierarchical paths).
     - Validated high-level taxonomy labels from OpenAI.
     - Explanations for label selection.