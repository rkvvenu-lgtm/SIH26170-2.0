import pandas as pd


class DataValidator:

    def __init__(self, required_columns=None):
        self.required_columns = required_columns or []

    def validate(self, df):

        errors = []
        warnings = []

        # 1. Empty dataset
        if df.empty:
            errors.append("Dataset is empty.")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings
            }

        # 2. Required columns
        missing_columns = [
            col for col in self.required_columns
            if col not in df.columns
        ]

        if missing_columns:
            errors.append(
                f"Missing required columns: {missing_columns}"
            )

        # 3. Duplicate rows
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            warnings.append(
                f"{duplicate_count} duplicate rows detected."
            )

        # 4. Missing values
        missing_values = df.isnull().sum()
        missing_values = missing_values[
            missing_values > 0
        ]

        if len(missing_values) > 0:
            warnings.append(
                f"Missing values detected: "
                f"{missing_values.to_dict()}"
            )

        # 5. Numeric column validation
        for column in df.columns:

            if column in self.required_columns:

                if not pd.api.types.is_numeric_dtype(
                    df[column]
                ):
                    errors.append(
                        f"Column '{column}' must be numeric."
                    )

        # 6. Final status
        valid = len(errors) == 0

        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings
        }


def load_and_validate(
    file_path,
    required_columns=None
):

    df = pd.read_csv(file_path)

    validator = DataValidator(
        required_columns=required_columns
    )

    result = validator.validate(df)

    return df, result