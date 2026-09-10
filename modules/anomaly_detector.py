import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Unsupervised anomaly detection using:
    1. Isolation Forest
    2. Statistical Z-score
    3. Combined anomaly risk
    """

    def __init__(
        self,
        contamination=0.15,
        n_estimators=200,
        random_state=42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )

        self.is_fitted = False

    def fit(self, data, feature_columns):

        features = self._prepare_features(
            data,
            feature_columns
        )

        scaled_features = self.scaler.fit_transform(
            features
        )

        self.model.fit(
            scaled_features
        )

        self.is_fitted = True

        return self

    def predict(self, data, feature_columns):

        if not self.is_fitted:
            self.fit(
                data,
                feature_columns
            )

        features = self._prepare_features(
            data,
            feature_columns
        )

        scaled_features = self.scaler.transform(
            features
        )

        prediction = self.model.predict(
            scaled_features
        )

        decision_score = self.model.decision_function(
            scaled_features
        )

        # Isolation Forest:
        # lower decision score = more anomalous
        isolation_score = np.clip(
            0.5 - decision_score,
            0,
            1
        )

        # Statistical anomaly
        z_scores = np.abs(
            (features - features.mean())
            / features.std(ddof=0).replace(0, np.nan)
        )

        z_scores = z_scores.fillna(0)

        max_z_score = z_scores.max(
            axis=1
        )

        statistical_score = np.clip(
            max_z_score / 5.0,
            0,
            1
        )

        anomaly_status = np.where(
            (prediction == -1)
            | (max_z_score >= 3.0),
            "ANOMALY",
            "NORMAL"
        )

        anomaly_risk = (
            0.5 * isolation_score
            + 0.5 * statistical_score
        )

        result = data.copy()

        result["Isolation_Prediction"] = prediction

        result["Isolation_Status"] = np.where(
            prediction == -1,
            "ANOMALY",
            "NORMAL"
        )

        result["Isolation_Score"] = (
            isolation_score
        )

        result["Max_Z_Score"] = (
            max_z_score
        )

        result["Statistical_Status"] = np.where(
            max_z_score >= 3.0,
            "ANOMALY",
            "NORMAL"
        )

        result["Statistical_Score"] = (
            statistical_score
        )

        result["Anomaly_Risk"] = np.clip(
            anomaly_risk,
            0,
            1
        )

        result["Anomaly_Status"] = (
            anomaly_status
        )

        return result

    def _prepare_features(
        self,
        data,
        feature_columns
    ):

        missing_columns = [
            column
            for column in feature_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing anomaly features: "
                + ", ".join(missing_columns)
            )

        features = data[
            feature_columns
        ].copy()

        for column in feature_columns:
            features[column] = pd.to_numeric(
                features[column],
                errors="coerce"
            )

        features = features.replace(
            [np.inf, -np.inf],
            np.nan
        )

        features = features.fillna(
            features.median()
        )

        features = features.fillna(0)

        return features

    def run(
        self,
        data,
        feature_columns
    ):

        return self.predict(
            data,
            feature_columns
        )