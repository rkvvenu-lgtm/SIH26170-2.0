from pathlib import Path
import json

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from models.model_manager import ModelManager


class TrainingPipeline:
    """
    Complete model-training pipeline.

    Workflow:
        Dataset
          ↓
        Feature/Target selection
          ↓
        Train/Test split
          ↓
        Model comparison
          ↓
        Best model selection
          ↓
        Train best model
          ↓
        Save model
          ↓
        Update model registry
    """

    def __init__(
        self,
        model_directory="models/saved",
        registry_path="models/model_registry.json",
        random_state=42
    ):

        self.random_state = random_state

        self.model_manager = ModelManager(
            model_directory=model_directory
        )

        self.registry_path = Path(
            registry_path
        )

    def _create_models(self):

        return {
            "LinearRegression": LinearRegression(),

            "RandomForest": RandomForestRegressor(
                n_estimators=200,
                random_state=self.random_state
            ),

            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state
            )
        }

    def _prepare_data(
        self,
        data,
        feature_columns,
        target_column
    ):

        missing_features = [
            column
            for column in feature_columns
            if column not in data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing feature columns: "
                f"{missing_features}"
            )

        if target_column not in data.columns:
            raise ValueError(
                f"Target column not found: "
                f"{target_column}"
            )

        X = data[feature_columns].copy()

        y = data[target_column].copy()

        for column in feature_columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

        # Remove rows where target is unavailable
        valid_rows = y.notna()

        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

        # Median imputation for numerical features
        X = X.fillna(
            X.median()
        )

        if X.empty:
            raise ValueError(
                "No valid training samples available."
            )

        return X, y

    def compare_models(
        self,
        data,
        feature_columns,
        target_column
    ):

        X, y = self._prepare_data(
            data,
            feature_columns,
            target_column
        )

        if len(X) < 10:
            raise ValueError(
                "At least 10 valid samples are required "
                "for model training."
            )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=self.random_state
            )
        )

        results = []

        for model_name, model in (
            self._create_models().items()
        ):

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                prediction
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    prediction
                )
            )

            r2 = r2_score(
                y_test,
                prediction
            )

            results.append({
                "Parameter": target_column,
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

        comparison = pd.DataFrame(
            results
        )

        # Lowest MAE = selected model
        best_index = comparison[
            "MAE"
        ].idxmin()

        best_model_name = comparison.loc[
            best_index,
            "Model"
        ]

        return comparison, best_model_name

    def train_and_save(
        self,
        data,
        feature_columns,
        target_column
    ):

        comparison, best_model_name = (
            self.compare_models(
                data,
                feature_columns,
                target_column
            )
        )

        X, y = self._prepare_data(
            data,
            feature_columns,
            target_column
        )

        models = self._create_models()

        best_model = models[
            best_model_name
        ]

        # Train selected model using all
        # available valid training data.
        best_model.fit(
            X,
            y
        )

        model_name = (
            f"{target_column}_"
            f"{best_model_name}"
        )

        model_path = self.model_manager.save(
            best_model,
            model_name
        )

        self._update_registry(
            target_column,
            best_model_name,
            model_path,
            comparison
        )

        return {
            "parameter": target_column,
            "selected_model": best_model_name,
            "model_path": str(model_path),
            "comparison": comparison
        }

    def _load_registry(self):

        if not self.registry_path.exists():
            return {
                "models": {}
            }

        with open(
            self.registry_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    def _save_registry(self, registry):

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.registry_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                registry,
                file,
                indent=4
            )

    def _update_registry(
        self,
        parameter,
        model_name,
        model_path,
        comparison
    ):

        registry = self._load_registry()

        registry.setdefault(
            "models",
            {}
        )

        registry["models"][parameter] = {
            "selected_model": model_name,
            "status": "TRAINED",
            "model_path": str(model_path),
            "metrics": comparison.to_dict(
                orient="records"
            )
        }

        self._save_registry(
            registry
        )

    def train_multiple(
        self,
        data,
        configurations
    ):

        results = []

        for configuration in configurations:

            result = self.train_and_save(
                data=data,
                feature_columns=configuration[
                    "features"
                ],
                target_column=configuration[
                    "target"
                ]
            )

            results.append(result)

        return results