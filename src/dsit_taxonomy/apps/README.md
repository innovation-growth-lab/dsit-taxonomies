# Project Taxonomy Visualiser

A Streamlit application for visualising and exploring project taxonomy assignments with sentence-level highlighting.

## Overview

The Project Taxonomy Visualiser helps users explore how research projects are classified according to different taxonomies. It provides an interactive interface to:

- Search through projects by title
- View project descriptions with highlighted sentences showing taxonomy matches
- Explore different confidence levels and assignment methods
- Compare sentence-based and zero-shot classification results

## Features

### Project Search
- Filter projects by title using the search bar
- Select from matching projects to view detailed analysis

### Text Visualisation
- Project descriptions with color-coded sentence highlighting
- Interactive tooltips showing match scores and confidence levels
- Clear visual distinction between different taxonomy labels

### Label & Score Tabs

1. **Matched Labels**
   - Shows all taxonomy labels matched in the text
   - Displays detailed scores for each match:
     - Sentence match scores
     - Global match scores
     - Key match scores
     - Overall relevance scores

2. **Basic Assignments**
   - Sentence-based confidence levels
   - Zero-shot classification confidence levels

3. **Composite Assignments**
   - Max Confidence (highest confidence between methods)
   - Zero-shot favoured assignments (select highest when disagreement is minor, otherwise select zero-shot)
   - Sentence-favoured assignments (select highest when disagreement is minor, otherwise select sentence)

## Usage

1. Launch the application:
```bash
streamlit run src/dsit_taxonomy/apps/project_visualiser.py
```

2. Select a taxonomy from the dropdown menu
3. Search for a project using the search bar
4. Explore the visualisations and analysis in the main panel

Upon request, the application can also be made temporarily available at [a dedicated URL](igl-dsit.dap-tools.uk).

## Dependencies

- Streamlit
- Pandas
- NumPy
- Kedro
- Python 3.7+

## Data Requirements

The application expects the following data to be available through the Kedro catalog:

- Project metadata
- Taxonomy scores (CWTS, GOScience, OpenAlex Concepts)
- Sentence-level matches
- Project descriptions

## Notes

- The application uses caching to optimise performance
- Color schemes are automatically generated for visual distinction
- Paragraph structure is preserved in the visualisation