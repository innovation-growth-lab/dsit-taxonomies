# Key Elements

## 1. Datasets
The `datasets` folder contains the data catalog configuration files. These define the input and output datasets for various pipelines, including their locations and types.

## 2. Pipelines
The `pipelines` folder contains all data processing and analysis pipelines. Each pipeline is organised into modular components such as `nodes.py`, `pipeline.py`, and utility scripts. Refer to the `README.md` in the `pipelines` folder for a detailed description of all pipelines. See Kedro's [documentation](https://docs.kedro.org/en/stable/) for details on the modular components and how to execute them.

## 3. `pipeline_registry.py`
This file registers all available pipelines for execution. It is the entry point for running pipelines using Kedro.

## 4. `settings.py`
This file contains the project settings, including configuration paths and Kedro-specific options.

---

# Usage

To execute a specific pipeline, use the following Kedro command:

```bash
kedro run --pipeline <pipeline_name>
```

For additional configuration and options, refer to the Kedro documentation or the relevant sections in this project.

---

Feel free to explore the repository and [reach out](david.ampudia@nesta.org.uk) if you have any questions or need assistance!

