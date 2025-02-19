# Data Collection GtR Pipeline

The **Data Collection GtR Pipeline** fetches and preprocesses data from the Gateway to Research (GtR) API. While the pipeline supports multiple data types, it is primarily designed and recommended to be used for collecting project data.

<img src="https://eosc.eu/wp-content/uploads/2024/11/UKRI-logo.png" alt="GtR" style="width:100%;"/>

## Features
- Robust API data fetching with retry mechanisms and rate limiting
- Comprehensive preprocessing focused on project data:
  - Projects: Handles research topics, subjects, and linked publications
  - Rich extraction of project metadata including abstracts and impact statements
- Support for additional endpoints (publications, organisations, funds) if needed
- Parallel processing for efficient data concatenation
- Test mode support for development and debugging

## Pipeline Components

### Nodes
1. **`fetch_gtr_data`**  
   - Handles paginated API requests with configurable retry logic
   - Implements web etiquette with randomised delays
   - Preprocesses data using type-specific methods
   - Yields data in a structured JSON format

2. **`concatenate_endpoint`**  
   - Parallelises data loading for improved performance
   - Combines multiple data chunks into a unified DataFrame
   - Handles both dictionary and DataFrame inputs

### Key Classes
- **`GtRDataPreprocessor`**: Centralises data preprocessing logic with specialised methods for each data type

## Configuration

The pipeline is configured through `parameters_data_collection_gtr.yml`:

```yaml
gtr_config:
  base_url: https://gtr.ukri.org/gtr/api/
  headers:
    Accept: application/vnd.rcuk.gtr.json-v7
  page_size: 100
max_retries: 5
backoff_factor: 0.3
```

## Usage

### Recommended Usage - Projects Only
```bash
kedro run --pipeline data_collection_gtr --tags projects
```

This is the recommended way to run the pipeline, as it focuses on collecting project data which contains the relevant information for this project.

### Full Pipeline (Not Necessary)
While possible, running the full pipeline is usually unnecessary:
```bash
kedro run --pipeline data_collection_gtr
```

### Other Endpoints (Optional)
These endpoints are supported but not needed for downstream purposes:
```bash
kedro run --pipeline data_collection_gtr --tags publications
kedro run --pipeline data_collection_gtr --tags organisations
kedro run --pipeline data_collection_gtr --tags funds
```

### Test Mode
Enable test mode in the parameters file to limit data collection for testing:
```yaml
test_mode: true
```

## Dependencies
- **Core Libraries**: 
  - `kedro`
  - `pandas`
  - `requests`
  - `numpy`
  - `joblib`
- **Python Version**: 3.8+
- **API Access**: Gateway to Research API

## Data Outputs

### Primary Output
The pipeline's main output is the projects dataset with the following structure:

- **Projects**:
  - Research topics and subjects with percentage allocations
  - Abstract text and technical details
  - Potential impact statements
  - Grant categories and funding information
  - Linked publications
  
### Additional Outputs (If Required)
The pipeline can also process:
  
- **Publications**:
  - Bibliographic information (DOI, ISBN, ISSN)
  - Publication dates and journal details
  - Project linkages
  
- **Organisations**:
  - Main address information
  - Contact details
  
- **Funds**:
  - Standardised currency amounts
  - Funding details

  