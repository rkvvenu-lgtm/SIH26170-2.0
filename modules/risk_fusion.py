import pandas as pd


class RiskFusion:
    """
    Combines anomaly risk and drift risk into a single screening decision.
    """

    def __init__(
        self,
        anomaly_weight=0.5,
        drift_weight=0.5,
        investigate_threshold=0.25,
        reject_threshold=0.70
    ):
        total_weight = anomaly_weight + drift_weight

        if total_weight <= 0:
            raise ValueError("Risk weights must have a positive total.")

        self.anomaly_weight = anomaly_weight / total_weight
        self.drift_weight = drift_weight / total_weight

        self.investigate_threshold = investigate_threshold
        self.reject_threshold = reject_threshold

    def calculate_risk(self, data):
        result = data.copy()

        anomaly_risk = pd.to_numeric(
            result.get("Anomaly_Risk", 0),
            errors="coerce"
        ).fillna(0).clip(0, 1)

        drift_risk = pd.to_numeric(
            result.get("Drift_Risk", 0),
            errors="coerce"
        ).fillna(0).clip(0, 1)

        result["Risk_Score"] = (
            self.anomaly_weight * anomaly_risk
            + self.drift_weight * drift_risk
        )

        result["Risk_Percentage"] = (
            result["Risk_Score"] * 100
        )

        result["Decision"] = result["Risk_Score"].apply(
            self._get_decision
        )

        result["Explanation"] = result.apply(
            self._generate_explanation,
            axis=1
        )

        return result

    def _get_decision(self, risk_score):
        if risk_score >= self.reject_threshold:
            return "REJECT"

        if risk_score >= self.investigate_threshold:
            return "INVESTIGATE"

        return "PASS"

    def _generate_explanation(self, row):
        reasons = []

        anomaly_risk = float(row.get("Anomaly_Risk", 0) or 0)
        drift_risk = float(row.get("Drift_Risk", 0) or 0)

        anomaly_status = str(
            row.get("Anomaly_Status", "NORMAL")
        )

        drift_status = str(
            row.get("Drift_Status", "NORMAL")
        )

        specification_status = str(
            row.get("Specification_Status", "PASS")
        )

        # Specification reason
        if specification_status == "REJECT":
            spec_reason = str(
                row.get(
                    "Specification_Reason",
                    "Engineering specification exceeded"
                )
            )
            reasons.append(spec_reason)

        elif specification_status == "INVESTIGATE":
            spec_reason = str(
                row.get(
                    "Specification_Reason",
                    "Approaching engineering specification"
                )
            )
            reasons.append(spec_reason)

        # Anomaly explanation
        if anomaly_status == "ANOMALY":
            reasons.append(
                f"Anomalous behaviour detected "
                f"(anomaly risk {anomaly_risk:.2f})"
            )

        # Drift explanation
        if drift_status == "HIGH_RISK":
            reasons.append(
                f"High future drift risk "
                f"(drift risk {drift_risk:.2f})"
            )

        elif drift_status == "EARLY_WARNING":
            reasons.append(
                f"Future drift early warning "
                f"(drift risk {drift_risk:.2f})"
            )

        # If no abnormal condition exists
        if not reasons:
            if anomaly_status == "NORMAL" and drift_status == "NORMAL":
                return "Normal behaviour; within engineering specification"

            return "No significant abnormal condition detected"

        return "; ".join(reasons)