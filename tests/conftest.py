"""
Configuration file for pytest. We use pytest, see more at:
https://docs.pytest.org/en/latest/getting-started.html
"""
# pylint: skip-file
import pytest
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog


@pytest.fixture
def config_loader():
    """Create a config loader using a local conf folder."""
    return OmegaConfigLoader(conf_source=str(Path.cwd()))


@pytest.fixture
def project_context(config_loader):
    """Create a project context using a local conf folder."""
    return KedroContext(
        package_name="dsit_impact",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
        env="local",
    )


@pytest.fixture
def seq_runner():
    """Create a sequential runner."""
    return SequentialRunner()


@pytest.fixture
def catalog():
    """Load the data catalog."""
    return DataCatalog()
