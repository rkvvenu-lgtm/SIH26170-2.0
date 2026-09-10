import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class DriftPredictor:
    """
    Predict parameter drift using candidate regression models.

    Candidate models:
    - Linear Regression
    - Random Forest
    - Gradient Boosting

    The model with the lowest MAE on unseen test data
    is selected for each parameter.
    """

    def __init__(self, random_state=42):

        self.random_state = random_state
        self.models = {}
        self.selected_models = {}

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

    def _prepare_features(self, data, feature_columns):

        missing = [
            column
            for column in feature_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}"
            )

        X = data[feature_columns].copy()

        for column in feature_columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        X = X.fillna(X.median())

        return X

    def compare_models(
        self,
        data,
        feature_columns,
        target_column
    ):
        """
        Compare regression models using an unseen test set.
        """

        if target_column not in data.columns:
            raise ValueError(
                f"Target column not found: {target_column}"
            )

        X = self._prepare_features(
            data,
            feature_columns
        )

        y = pd.to_numeric(
            data[target_column],
            errors="coerce"
        )

        valid_rows = y.notna()

        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

        if len(X) < 10:
            raise ValueError(
                f"Not enough valid samples for {target_column}."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=self.random_state
        )

        model_results = []

        for model_name, model in self._create_models().items():

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

            model_results.append({
                "Parameter": target_column,
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

        results = pd.DataFrame(
            model_results
        )

        best_row = results.loc[
            results["MAE"].idxmin()
        ]

        self.selected_models[
            target_column
        ] = best_row["Model"]

        return results

    def train_best_model(
        self,
        data,
        feature_columns,
        target_column
    ):
        """
        Train the selected model using the complete
        available dataset after model selection.
        """

        if target_column not in self.selected_models:

            self.compare_models(
                data,
                feature_columns,
                target_column
            )

        selected_name = self.selected_models[
            target_column
        ]

        models = self._create_models()

        model = models[selected_name]

        X = self._prepare_features(
            data,
            feature_columns
        )

        y = pd.to_numeric(
            data[target_column],
            errors="coerce"
        )

        valid_rows = y.notna()

        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

        model.fit(
            X,
            y
        )

        self.models[
            target_column
        ] = model

        return model

    def predict(
        self,
        data,
        feature_columns,
        target_column
    ):
        """
        Predict the target parameter.
        """

        if target_column not in self.models:

            self.train_best_model(
                data,
                feature_columns,
                target_column
            )

        X = self._prepare_features(
            data,
            feature_columns
        )

        model = self.models[
            target_column
        ]

        return model.predict(X)

    def calculate_drift(
        self,
        current_value,
        predicted_value
    ):
        """
        Calculate relative drift.

        Positive value  -> parameter increased
        Negative value  -> parameter decreased
        """

        current_value = np.asarray(
            current_value,
            dtype=float
        )

        predicted_value = np.asarray(
            predicted_value,
            dtype=float
        )

        denominator = np.where(
            np.abs(current_value) < 1e-12,
            np.nan,
            current_value
        )

        drift = (
            predicted_value - current_value
        ) / denominator

        return np.nan_to_num(
            drift,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

    def generate_drift_risk(
        self,
        data,
        current_column,
        predicted_column,
        warning_threshold=0.20,
        reject_threshold=0.40
    ):
        """
        Convert predicted drift into a risk level.
        """

        result = data.copy()

        drift = self.calculate_drift(
            result[current_column],
            result[predicted_column]
        )

        result["Drift_Rate"] = drift

        absolute_drift = np.abs(
            result["Drift_Rate"]
        )

        result["Drift_Risk"] = np.clip(
            absolute_drift / reject_threshold,
            0.0,
            1.0
        )

        result["Drift_Status"] = np.select(
            [
                absolute_drift >= reject_threshold,
                absolute_drift >= warning_threshold
            ],
            [
                "HIGH_RISK",
                "EARLY_WARNING"
            ],
            default="NORMAL"
        )

        return result

    def run(
        self,
        data,
        feature_columns,
        target_column,
        current_column,
        warning_threshold=0.20,
        reject_threshold=0.40
    ):
        """
        Complete drift prediction pipeline.
        """

        result = data.copy()

        # Compare candidate models
        comparison = self.compare_models(
            result,
            feature_columns,
            target_column
        )

        # Train selected model
        self.train_best_model(
            result,
            feature_columns,
            target_column
        )

        # Predict future parameter
        result[
            "Predicted_" + target_column
        ] = self.predict(
            result,
            feature_columns,
            target_column
        )

        # Calculate drift risk
        result = self.generate_drift_risk(
            result,
            current_column,
            "Predicted_" + target_column,
            warning_threshold,
            reject_threshold
        )

        return result, comparison