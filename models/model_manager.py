from pathlib import Path
import joblib


class ModelManager:
    """
    Manage trained machine-learning models.

    Responsibilities:
    - Save trained models
    - Load trained models
    - Check model availability
    - Maintain model files in a common directory
    """

    def __init__(self, model_directory="models/saved"):

        self.model_directory = Path(
            model_directory
        )

        self.model_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def _model_path(self, model_name):

        if not model_name:
            raise ValueError(
                "Model name cannot be empty."
            )

        safe_name = str(
            model_name
        ).strip().replace(" ", "_")

        return (
            self.model_directory
            / f"{safe_name}.joblib"
        )

    def save(self, model, model_name):

        if model is None:
            raise ValueError(
                "Cannot save an empty model."
            )

        model_path = self._model_path(
            model_name
        )

        joblib.dump(
            model,
            model_path
        )

        return model_path

    def load(self, model_name):

        model_path = self._model_path(
            model_name
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        return joblib.load(
            model_path
        )

    def exists(self, model_name):

        return self._model_path(
            model_name
        ).exists()

    def list_models(self):

        if not self.model_directory.exists():
            return []

        return sorted([
            file.stem
            for file in self.model_directory.glob(
                "*.joblib"
            )
        ])

    def delete(self, model_name):

        model_path = self._model_path(
            model_name
        )

        if model_path.exists():
            model_path.unlink()
            return True

        return False