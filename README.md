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