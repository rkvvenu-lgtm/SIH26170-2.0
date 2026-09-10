import json
from pathlib import Path


class ConfigLoader:
    """
    Loads pipeline configuration and engineering specifications.
    """

    def __init__(self, config_dir="config"):
        self.config_dir = Path(config_dir)

        self.pipeline_config_path = (
            self.config_dir / "pipeline_config.json"
        )

        self.specifications_path = (
            self.config_dir / "specifications.json"
        )

    def _load_json(self, path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Configuration path is not a file: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    def load_pipeline_config(self):
        return self._load_json(
            self.pipeline_config_path
        )

    def load_specifications(self):
        return self._load_json(
            self.specifications_path
        )