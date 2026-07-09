from src.csr.config.configuration import ConfigurationManager
from src.csr.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def test_mlflow_config_uses_environment() -> None:
    config = ConfigurationManager().get_mlflow_config()

    assert config.tracking_uri == MLFLOW_TRACKING_URI
    assert config.experiment_name == MLFLOW_EXPERIMENT_NAME
