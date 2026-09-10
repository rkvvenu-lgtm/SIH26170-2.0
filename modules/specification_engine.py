import pandas as pd


class SpecificationEngine:
    """
    Engineering specification evaluation engine.

    Supports:
    1. Actual measurement vs engineering limit
    2. Predicted future value vs engineering limit
    3. Component-level specification evaluation
    """

    def __init__(self, specifications=None):
        self.specifications = specifications or {}

    # =========================================================
    # Generic specification evaluation
    # =========================================================

    def evaluate(self, parameter, value):

        if parameter not in self.specifications:

            return {
                "status": "UNKNOWN",
                "reason": (
                    f"No specification available "
                    f"for {parameter}"
                )
            }

        spec = self.specifications[parameter]

        minimum = spec.get("min")
        maximum = spec.get("max")

        warning_min = spec.get("warning_min")
        warning_max = spec.get("warning_max")

        if minimum is not None and value < minimum:

            return {
                "status": "REJECT",
                "reason": (
                    f"{parameter} is below "
                    f"the minimum specification"
                )
            }

        if maximum is not None and value > maximum:

            return {
                "status": "REJECT",
                "reason": (
                    f"{parameter} is above "
                    f"the maximum specification"
                )
            }

        if (
            warning_min is not None
            and value < warning_min
        ):

            return {
                "status": "INVESTIGATE",
                "reason": (
                    f"{parameter} is approaching "
                    f"the lower limit"
                )
            }

        if (
            warning_max is not None
            and value > warning_max
        ):

            return {
                "status": "INVESTIGATE",
                "reason": (
                    f"{parameter} is approaching "
                    f"the upper limit"
                )
            }

        return {
            "status": "PASS",
            "reason": (
                f"{parameter} is within specification"
            )
        }

    # =========================================================
    # Value vs engineering limit
    # =========================================================

    def evaluate_against_limit(
        self,
        parameter,
        value,
        maximum_limit,
        warning_ratio=0.80
    ):

        if pd.isna(value):

            return {
                "status": "UNKNOWN",
                "reason": (
                    f"{parameter} value is missing"
                )
            }

        if pd.isna(maximum_limit):

            return {
                "status": "UNKNOWN",
                "reason": (
                    f"No engineering limit "
                    f"available for {parameter}"
                )
            }

        try:

            value = float(value)
            maximum_limit = float(maximum_limit)

        except (
            ValueError,
            TypeError
        ):

            return {
                "status": "UNKNOWN",
                "reason": (
                    f"Invalid numeric value "
                    f"for {parameter}"
                )
            }

        if maximum_limit <= 0:

            return {
                "status": "UNKNOWN",
                "reason": (
                    f"Invalid engineering limit "
                    f"for {parameter}"
                )
            }

        if value > maximum_limit:

            return {
                "status": "REJECT",
                "reason": (
                    f"{parameter} exceeds the "
                    f"engineering limit "
                    f"({maximum_limit})"
                )
            }

        warning_limit = (
            warning_ratio * maximum_limit
        )

        if value >= warning_limit:

            return {
                "status": "INVESTIGATE",
                "reason": (
                    f"{parameter} is approaching "
                    f"the engineering limit "
                    f"({maximum_limit})"
                )
            }

        return {
            "status": "PASS",
            "reason": (
                f"{parameter} is within the "
                f"engineering limit"
            )
        }

    # =========================================================
    # Predicted future value vs limit
    # =========================================================

    def evaluate_predicted_against_limit(
        self,
        parameter,
        predicted_value,
        maximum_limit,
        warning_ratio=0.80
    ):

        if pd.isna(predicted_value):

            return {
                "status": "UNKNOWN",
                "reason": (
                    f"Predicted value for "
                    f"{parameter} is unavailable"
                )
            }

        if pd.isna(maximum_limit):

            return {
                "status": "UNKNOWN",
                "reason": (
                    f"No engineering limit "
                    f"available for {parameter}"
                )
            }

        try:

            predicted_value = float(
                predicted_value
            )

            maximum_limit = float(
                maximum_limit
            )

        except (
            ValueError,
            TypeError
        ):

            return {
                "status": "UNKNOWN",
                "reason": (
                    f"Invalid predicted value "
                    f"for {parameter}"
                )
            }

        if maximum_limit <= 0:

            return {
                "status": "UNKNOWN",
                "reason": (
                    f"Invalid engineering limit "
                    f"for {parameter}"
                )
            }

        warning_limit = (
            warning_ratio * maximum_limit
        )

        if predicted_value > maximum_limit:

            return {
                "status": "REJECT",
                "reason": (
                    f"Predicted {parameter} at 168h "
                    f"({predicted_value:.3f}) exceeds "
                    f"the engineering limit "
                    f"({maximum_limit:.3f})"
                )
            }

        if predicted_value >= warning_limit:

            return {
                "status": "INVESTIGATE",
                "reason": (
                    f"Predicted {parameter} at 168h "
                    f"({predicted_value:.3f}) is approaching "
                    f"the engineering limit "
                    f"({maximum_limit:.3f})"
                )
            }

        return {
            "status": "PASS",
            "reason": (
                f"Predicted {parameter} at 168h "
                f"({predicted_value:.3f}) is within "
                f"the engineering limit "
                f"({maximum_limit:.3f})"
            )
        }

    # =========================================================
    # Actual component specification evaluation
    # =========================================================

    def evaluate_component(self, row):

        evaluations = {}

        parameter_mapping = {

            "Iddq_168h_uA":
                "Iddq_Max_Limit_uA",

            "Leakage_168h_uA":
                "Leakage_Max_Limit_uA",

            "Delay_168h_ns":
                "Delay_Max_Limit_ns"
        }

        for parameter, limit_column in (
            parameter_mapping.items()
        ):

            if parameter not in row.index:
                continue

            if limit_column not in row.index:
                continue

            evaluations[parameter] = (
                self.evaluate_against_limit(
                    parameter,
                    row[parameter],
                    row[limit_column]
                )
            )

        return evaluations

    # =========================================================
    # Predicted future component evaluation
    # =========================================================

    def evaluate_predicted_component(
        self,
        row
    ):

        evaluations = {}

        parameter_mapping = {

            "Predicted_Iddq_168h_uA":
                (
                    "Iddq",
                    "Iddq_Max_Limit_uA"
                ),

            "Predicted_Leakage_168h_uA":
                (
                    "Leakage",
                    "Leakage_Max_Limit_uA"
                ),

            "Predicted_Delay_168h_ns":
                (
                    "Delay",
                    "Delay_Max_Limit_ns"
                )
        }

        for predicted_column, mapping in (
            parameter_mapping.items()
        ):

            parameter, limit_column = mapping

            if predicted_column not in row.index:
                continue

            if limit_column not in row.index:
                continue

            evaluations[predicted_column] = (
                self.evaluate_predicted_against_limit(
                    parameter,
                    row[predicted_column],
                    row[limit_column]
                )
            )

        return evaluations

    # =========================================================
    # Dataset-level actual specification evaluation
    # =========================================================

    def evaluate_dataset(self, data):

        result = data.copy()

        statuses = []
        reasons = []

        for _, row in result.iterrows():

            evaluations = (
                self.evaluate_component(row)
            )

            final_status = "PASS"
            component_reasons = []

            for (
                parameter,
                evaluation
            ) in evaluations.items():

                status = evaluation["status"]

                if status == "REJECT":

                    final_status = "REJECT"

                elif (
                    status == "INVESTIGATE"
                    and final_status != "REJECT"
                ):

                    final_status = "INVESTIGATE"

                if status != "PASS":

                    component_reasons.append(
                        evaluation["reason"]
                    )

            if not evaluations:

                final_status = "UNKNOWN"

                component_reasons.append(
                    "No applicable engineering "
                    "specifications found"
                )

            statuses.append(
                final_status
            )

            reasons.append(
                "; ".join(component_reasons)
            )

        result[
            "Specification_Status"
        ] = statuses

        result[
            "Specification_Reason"
        ] = reasons

        return result

    # =========================================================
    # Dataset-level predicted specification evaluation
    # =========================================================

    def evaluate_predicted_dataset(
        self,
        data
    ):

        result = data.copy()

        statuses = []
        reasons = []

        for _, row in result.iterrows():

            evaluations = (
                self.evaluate_predicted_component(
                    row
                )
            )

            final_status = "PASS"
            component_reasons = []

            for (
                parameter,
                evaluation
            ) in evaluations.items():

                status = evaluation["status"]

                if status == "REJECT":

                    final_status = "REJECT"

                elif (
                    status == "INVESTIGATE"
                    and final_status != "REJECT"
                ):

                    final_status = "INVESTIGATE"

                if status != "PASS":

                    component_reasons.append(
                        evaluation["reason"]
                    )

            if not evaluations:

                final_status = "UNKNOWN"

                component_reasons.append(
                    "No predicted values available "
                    "for specification evaluation"
                )

            statuses.append(
                final_status
            )

            reasons.append(
                "; ".join(component_reasons)
            )

        result[
            "Predicted_Specification_Status"
        ] = statuses

        result[
            "Predicted_Specification_Reason"
        ] = reasons

        return result