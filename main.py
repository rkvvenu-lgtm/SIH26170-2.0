from pathlib import Path

import pandas as pd

from utils.data_loader import DataLoader
from utils.data_preprocessor import DataPreprocessor
from utils.config_loader import ConfigLoader

from modules.data_validator import DataValidator
from modules.specification_engine import SpecificationEngine
from modules.anomaly_detector import AnomalyDetector
from modules.drift_predictor import DriftPredictor
from modules.risk_fusion import RiskFusion


class ScreeningPipeline:
    """
    AI-Assisted Adaptive Component Screening Pipeline

    Complete workflow:

        Data Loading
            ↓
        Preprocessing
            ↓
        Data Validation
            ↓
        Current Engineering Specification Check
            ↓
        AI Anomaly Detection
            ↓
        Multi-Parameter Drift Prediction
            ↓
        Predicted 168h Engineering Specification Check
            ↓
        Risk Fusion
            ↓
        Final Decision
            ↓
        Save Results + Model Metrics

    Final decisions:
        PASS
        INVESTIGATE
        REJECT

    Important:
        Actual 168h measurements are retained for retrospective
        validation/reference.

        Predicted 168h values are used for future-oriented screening.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self, config_dir="config"):

        # -----------------------------------------------------
        # Configuration
        # -----------------------------------------------------

        self.config_loader = ConfigLoader(config_dir)

        self.pipeline_config = (
            self.config_loader.load_pipeline_config()
        )

        self.specifications = (
            self.config_loader.load_specifications()
        )

        # -----------------------------------------------------
        # Core utilities
        # -----------------------------------------------------

        self.data_loader = DataLoader()

        self.preprocessor = DataPreprocessor()

        self.validator = DataValidator()

        # -----------------------------------------------------
        # Specification Engine
        # -----------------------------------------------------

        self.specification_engine = SpecificationEngine(
            self.specifications.get(
                "electronics",
                {}
            )
        )

        # -----------------------------------------------------
        # Anomaly Detector
        # -----------------------------------------------------

        anomaly_config = self.pipeline_config.get(
            "anomaly_detection",
            {}
        )

        self.anomaly_detector = AnomalyDetector(
            contamination=anomaly_config.get(
                "contamination",
                0.15
            ),
            n_estimators=anomaly_config.get(
                "n_estimators",
                200
            ),
            random_state=anomaly_config.get(
                "random_state",
                42
            )
        )

        # -----------------------------------------------------
        # Drift Predictor
        # -----------------------------------------------------

        drift_config = self.pipeline_config.get(
            "drift_prediction",
            {}
        )

        self.drift_predictor = DriftPredictor(
            random_state=drift_config.get(
                "random_state",
                42
            )
        )

        # -----------------------------------------------------
        # Risk Fusion
        # -----------------------------------------------------

        risk_config = self.pipeline_config.get(
            "risk_fusion",
            {}
        )

        self.risk_fusion = RiskFusion(
            anomaly_weight=risk_config.get(
                "anomaly_weight",
                0.5
            ),
            drift_weight=risk_config.get(
                "drift_weight",
                0.5
            ),
            investigate_threshold=risk_config.get(
                "investigate_threshold",
                0.25
            ),
            reject_threshold=risk_config.get(
                "reject_threshold",
                0.70
            )
        )

    # =========================================================
    # 1. CURRENT SPECIFICATION EVALUATION
    # =========================================================

    def evaluate_specifications(self, data):

        """
        Evaluate actual 168h measurements against
        engineering limits.

        This result is retained as a reference/
        retrospective validation result.

        It is NOT directly used as the future screening
        decision.
        """

        result = data.copy()

        result = self.specification_engine.evaluate_dataset(
            result
        )

        return result

    # =========================================================
    # 2. ANOMALY DETECTION
    # =========================================================

    def detect_anomalies(self, data):

        """
        Run AI-based anomaly detection.

        Uses configured anomaly features and
        combines Isolation Forest behaviour with
        statistical abnormality.
        """

        anomaly_config = self.pipeline_config.get(
            "anomaly_detection",
            {}
        )

        features = anomaly_config.get(
            "features",
            []
        )

        available_features = [
            feature
            for feature in features
            if feature in data.columns
        ]

        if not available_features:

            raise ValueError(
                "No anomaly detection features "
                "are available in the dataset."
            )

        result = self.anomaly_detector.run(
            data,
            available_features
        )

        return result

    # =========================================================
    # 3. MULTI-PARAMETER DRIFT PREDICTION
    # =========================================================

    def predict_all_drifts(self, data):

        """
        Predict future 168h behaviour for:

            IDDQ
            Leakage
            Delay

        For every parameter:

            0h + 24h + 96h
                    ↓
              ML Model Selection
                    ↓
              Best Model
                    ↓
              Predicted 168h
                    ↓
              Drift Rate
                    ↓
              Drift Risk
        """

        result = data.copy()

        drift_config = self.pipeline_config.get(
            "drift_prediction",
            {}
        )

        parameters = drift_config.get(
            "parameters",
            {}
        )

        parameter_metrics = []

        drift_risk_columns = []

        # -----------------------------------------------------
        # Process each parameter
        # -----------------------------------------------------

        for parameter, settings in parameters.items():

            print(
                f"   Processing {parameter}..."
            )

            # -------------------------------------------------
            # Configuration
            # -------------------------------------------------

            features = settings.get(
                "features",
                []
            )

            target = settings.get(
                "target"
            )

            current_column = settings.get(
                "current"
            )

            # -------------------------------------------------
            # Check feature availability
            # -------------------------------------------------

            available_features = [
                feature
                for feature in features
                if feature in result.columns
            ]

            if not available_features:

                print(
                    f"   Skipping {parameter}: "
                    f"prediction features missing."
                )

                continue

            if target not in result.columns:

                print(
                    f"   Skipping {parameter}: "
                    f"target column missing."
                )

                continue

            if current_column not in result.columns:

                print(
                    f"   Skipping {parameter}: "
                    f"current value column missing."
                )

                continue

            # -------------------------------------------------
            # Run drift prediction
            # -------------------------------------------------

            parameter_result, comparison = (
                self.drift_predictor.run(
                    data=result,
                    feature_columns=available_features,
                    target_column=target,
                    current_column=current_column,
                    warning_threshold=drift_config.get(
                        "warning_threshold",
                        0.20
                    ),
                    reject_threshold=drift_config.get(
                        "reject_threshold",
                        0.40
                    )
                )
            )

            # -------------------------------------------------
            # IMPORTANT:
            # Create prediction column using target name.
            #
            # Example:
            #
            # target:
            # Iddq_168h_uA
            #
            # prediction:
            # Predicted_Iddq_168h_uA
            # -------------------------------------------------

            predicted_column = (
                f"Predicted_{target}"
            )

            # -------------------------------------------------
            # Drift columns
            # -------------------------------------------------

            drift_rate_column = (
                f"{parameter}_Drift_Rate"
            )

            drift_risk_column = (
                f"{parameter}_Drift_Risk"
            )

            drift_status_column = (
                f"{parameter}_Drift_Status"
            )

            # -------------------------------------------------
            # Copy prediction
            # -------------------------------------------------

            if predicted_column in parameter_result.columns:

                result[predicted_column] = (
                    parameter_result[
                        predicted_column
                    ]
                )

            # -------------------------------------------------
            # Copy drift rate
            # -------------------------------------------------

            if drift_rate_column in parameter_result.columns:

                result[drift_rate_column] = (
                    parameter_result[
                        drift_rate_column
                    ]
                )

            # -------------------------------------------------
            # Copy drift risk
            # -------------------------------------------------

            if drift_risk_column in parameter_result.columns:

                result[drift_risk_column] = (
                    parameter_result[
                        drift_risk_column
                    ]
                )

                drift_risk_columns.append(
                    drift_risk_column
                )

            # -------------------------------------------------
            # Copy drift status
            # -------------------------------------------------

            if drift_status_column in parameter_result.columns:

                result[drift_status_column] = (
                    parameter_result[
                        drift_status_column
                    ]
                )

            # -------------------------------------------------
            # Store model comparison
            # -------------------------------------------------

            comparison = comparison.copy()

            if "Parameter" not in comparison.columns:

                comparison.insert(
                    0,
                    "Parameter",
                    parameter
                )

            parameter_metrics.append(
                comparison
            )

        # =====================================================
        # COMBINED DRIFT RISK
        # =====================================================

        if drift_risk_columns:

            result["Drift_Risk"] = (
                result[
                    drift_risk_columns
                ]
                .max(axis=1)
                .clip(0, 1)
            )

        else:

            result["Drift_Risk"] = 0.0

        # =====================================================
        # COMBINED DRIFT STATUS
        # =====================================================

        result["Drift_Status"] = (
            result[
                "Drift_Risk"
            ].apply(
                self._get_drift_status
            )
        )

        # =====================================================
        # MODEL METRICS
        # =====================================================

        if parameter_metrics:

            metrics = pd.concat(
                parameter_metrics,
                ignore_index=True
            )

        else:

            metrics = pd.DataFrame(
                columns=[
                    "Parameter",
                    "Model",
                    "MAE",
                    "RMSE",
                    "R2"
                ]
            )

        return result, metrics

    # =========================================================
    # COMBINED DRIFT STATUS
    # =========================================================

    def _get_drift_status(self, risk):

        if risk >= 0.70:

            return "HIGH_RISK"

        if risk >= 0.25:

            return "EARLY_WARNING"

        return "NORMAL"

    # =========================================================
    # 4. PREDICTED SPECIFICATION EVALUATION
    # =========================================================

    def evaluate_predicted_specifications(self, data):

        """
        Evaluate predicted 168h values against
        engineering limits.

        This is the main future-oriented engineering
        screening stage.
        """

        result = data.copy()

        result = (
            self.specification_engine
            .evaluate_predicted_dataset(
                result
            )
        )

        return result

    # =========================================================
    # 5. FINAL DECISION
    # =========================================================

    def apply_specification_decision(self, data):

        """
        Combine AI decision with predicted engineering
        specification status.

        Priority:

            Predicted specification REJECT
                    ↓
                 REJECT

            Predicted specification INVESTIGATE
                    ↓
              INVESTIGATE

            Otherwise:
                    ↓
              Keep AI decision
        """

        result = data.copy()

        def apply_override(row):

            # -------------------------------------------------
            # IMPORTANT:
            # Use predicted specification status.
            # -------------------------------------------------

            specification_status = str(
                row.get(
                    "Predicted_Specification_Status",
                    "PASS"
                )
            )

            ai_decision = str(
                row.get(
                    "Decision",
                    "PASS"
                )
            )

            # -------------------------------------------------
            # Predicted engineering limit exceeded
            # -------------------------------------------------

            if specification_status == "REJECT":

                return "REJECT"

            # -------------------------------------------------
            # Predicted engineering warning
            # -------------------------------------------------

            if (
                specification_status
                == "INVESTIGATE"
                and
                ai_decision
                == "PASS"
            ):

                return "INVESTIGATE"

            # -------------------------------------------------
            # Otherwise keep AI decision
            # -------------------------------------------------

            return ai_decision

        result["Decision"] = result.apply(
            apply_override,
            axis=1
        )

        return result

    # =========================================================
    # 6. SAVE RESULTS
    # =========================================================

    def save_results(
        self,
        result,
        metrics,
        output_dir="results"
    ):

        """
        Save final screening results and
        model comparison metrics.
        """

        output_path = Path(
            output_dir
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # -----------------------------------------------------
        # Final result file
        # -----------------------------------------------------

        result_file = (
            output_path
            / "final_screening_results.csv"
        )

        # -----------------------------------------------------
        # Model metrics file
        # -----------------------------------------------------

        metrics_file = (
            output_path
            / "model_metrics.csv"
        )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        result.to_csv(
            result_file,
            index=False
        )

        metrics.to_csv(
            metrics_file,
            index=False
        )

        return (
            result_file,
            metrics_file
        )

    # =========================================================
    # 7. COMPLETE PIPELINE
    # =========================================================

    def run(self, file_path):

        print()
        print("=" * 60)
        print("AI-ASSISTED ADAPTIVE COMPONENT SCREENING")
        print("=" * 60)

        # =====================================================
        # STEP 1
        # =====================================================

        print()
        print("1/7 Loading data...")

        data = self.data_loader.load_data(
            file_path
        )

        print(
            f"   Loaded {len(data)} records."
        )

        # =====================================================
        # STEP 2
        # =====================================================

        print()
        print("2/7 Preprocessing data...")

        data = self.preprocessor.prepare(
            data
        )

        # =====================================================
        # STEP 3
        # =====================================================

        print()
        print("3/7 Validating data...")

        validation = (
            self.validator.validate(
                data
            )
        )

        if not validation["valid"]:

            raise ValueError(
                "Data validation failed: "
                +
                "; ".join(
                    validation["errors"]
                )
            )

        if validation.get("warnings"):

            print()
            print("   Validation warnings:")

            for warning in validation[
                "warnings"
            ]:

                print(
                    f"   - {warning}"
                )

        # =====================================================
        # STEP 4
        # =====================================================

        print()
        print(
            "4/7 Evaluating current engineering specifications..."
        )

        data = (
            self.evaluate_specifications(
                data
            )
        )

        # =====================================================
        # STEP 5
        # =====================================================

        print()
        print(
            "5/7 Running AI anomaly detection..."
        )

        data = (
            self.detect_anomalies(
                data
            )
        )

        # =====================================================
        # STEP 6
        # =====================================================

        print()
        print(
            "6/7 Running multi-parameter drift prediction..."
        )

        data, metrics = (
            self.predict_all_drifts(
                data
            )
        )

        # -----------------------------------------------------
        # Predicted 168h specification check
        # -----------------------------------------------------

        print()
        print(
            "   Evaluating predicted 168h engineering specifications..."
        )

        data = (
            self.evaluate_predicted_specifications(
                data
            )
        )

        # =====================================================
        # STEP 7
        # =====================================================

        print()
        print(
            "7/7 Calculating final risk..."
        )

        # -----------------------------------------------------
        # Risk fusion
        # -----------------------------------------------------

        data = (
            self.risk_fusion.calculate_risk(
                data
            )
        )

        # -----------------------------------------------------
        # Apply predicted specification override
        # -----------------------------------------------------

        data = (
            self.apply_specification_decision(
                data
            )
        )

        # =====================================================
        # SAVE
        # =====================================================

        result_file, metrics_file = (
            self.save_results(
                data,
                metrics
            )
        )

        # =====================================================
        # FINAL SUMMARY
        # =====================================================

        print()
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print()

        print(
            f"Components processed : {len(data)}"
        )

        # -----------------------------------------------------
        # Decision summary
        # -----------------------------------------------------

        print()
        print("Final decisions:")

        print(
            data[
                "Decision"
            ]
            .value_counts()
            .to_string()
        )

        # -----------------------------------------------------
        # Anomaly summary
        # -----------------------------------------------------

        if "Anomaly_Status" in data.columns:

            print()
            print("Anomaly status:")

            print(
                data[
                    "Anomaly_Status"
                ]
                .value_counts()
                .to_string()
            )

        # -----------------------------------------------------
        # Drift summary
        # -----------------------------------------------------

        if "Drift_Status" in data.columns:

            print()
            print("Drift status:")

            print(
                data[
                    "Drift_Status"
                ]
                .value_counts()
                .to_string()
            )

        # -----------------------------------------------------
        # Predicted specification summary
        # -----------------------------------------------------

        if (
            "Predicted_Specification_Status"
            in data.columns
        ):

            print()
            print(
                "Predicted specification status:"
            )

            print(
                data[
                    "Predicted_Specification_Status"
                ]
                .value_counts()
                .to_string()
            )

        # -----------------------------------------------------
        # Risk statistics
        # -----------------------------------------------------

        if "Risk_Score" in data.columns:

            print()
            print("Risk score summary:")

            print(
                data[
                    "Risk_Score"
                ]
                .describe()
                .to_string()
            )

        # -----------------------------------------------------
        # Saved files
        # -----------------------------------------------------

        print()
        print(
            f"Results : {result_file}"
        )

        print(
            f"Metrics : {metrics_file}"
        )

        print()
        print("=" * 60)

        return data, metrics


# =============================================================
# MAIN ENTRY POINT
# =============================================================

if __name__ == "__main__":

    pipeline = ScreeningPipeline()

    print()
    print(
        "Configuration loaded successfully."
    )

    print(
        "Multi-parameter AI screening pipeline is ready."
    )

    print()
    print(
        "Run using:"
    )

    print(
        "python -c \"from main import ScreeningPipeline; "
        "ScreeningPipeline().run('data/burn_in_measurements.csv')\""
    )