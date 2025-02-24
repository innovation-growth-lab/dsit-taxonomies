"""
Test file for pipeline 'data_collection_gtr'. We use pytest, see more at:
https://docs.pytest.org/en/latest/getting-started.html

To run this test file just type in the terminal:
$ pytest tests/pipelines/data_collection_gtr/test_unit.py

To run a specific parameterized test, use the following command:
$ pytest tests/pipelines/data_collection_gtr/test_unit.py::test_nodes[projects]
"""

# pylint: skip-file
import pandas as pd
import pytest

from dsit_taxonomy.pipelines.data_collection_gtr.nodes import fetch_gtr_data


@pytest.fixture
def params(project_context):
    """Get the parameters for the GtR API."""
    return project_context.config_loader["parameters"]["gtr"]["data_collection"]


def test_node(params):
    """
    Test that data is fetched from the GtR API and processed correctly.

    Args:
        params (dict): The parameters for the GtR API.
        endpoint (str): The API endpoint to fetch data from.
        label (str): The label specifying the data preprocessing method.

    Raises:
        AssertionError: If any of the assertions fail.
    """

    # fetch data from the GtR API
    data_generator = fetch_gtr_data(
        parameters=params["projects"]["param_requests"],
        endpoint=params["projects"]["label"],
    )

    data = next(data_generator)

    # assert the returned object is a yield dictionary
    _response_object_is_dict(data)

    # fetch the first item from the response list
    data = list(data.values())[0]

    # assert that the processed data is also a dictionary
    _response_object_is_dict(data)

    # assert that the processed data does not have a "links" column
    _dataframe_has_key_columns(data)


def _response_object_is_dict(data):
    """Assert that each item in the response list is a dictionary."""
    assert isinstance(data, dict)


def _dataframe_is_returned(data):
    """Assert that the processed data is a pandas DataFrame."""
    assert isinstance(data, pd.DataFrame)


def _dataframe_has_key_columns(data):
    """Assert that the processed data does have "project_id", "title", and "abstract_text" columns."""
    assert "project_id" in data.keys()
    assert "title" in data.keys()
    assert "abstract_text" in data.keys()
