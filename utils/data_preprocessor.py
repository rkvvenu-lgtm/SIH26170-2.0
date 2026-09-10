import pandas as pd
import numpy as np


class DataPreprocessor:

    def __init__(self):
        self.numeric_columns = []

    def clean(self, data):
        """
        Clean raw engineering data while preserving
        the original information as much as possible.
        """

        if data is None or data.empty:
            raise ValueError("Input dataset is empty.")

        df = data.copy()

        # Remove completely empty rows
        df = df.dropna(how="all").reset_index(drop=True)

        # Convert obvious numeric columns
        for column in df.columns:

            if df[column].dtype == "object":

                converted = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

                valid_ratio = converted.notna().mean()

                # Convert only if most values are numeric
                if valid_ratio >= 0.8:
                    df[column] = converted

        # Store numerical columns
        self.numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        return df

    def get_numeric_data(self, data):
        """
        Return only numerical features suitable
        for statistical/ML processing.
        """

        if data is None or data.empty:
            raise ValueError("Input dataset is empty.")

        numeric_data = data.select_dtypes(
            include=np.number
        ).copy()

        if numeric_data.empty:
            raise ValueError(
                "No numerical features found in the dataset."
            )

        return numeric_data

    def handle_missing_values(self, data):
        """
        Handle missing numerical values using
        column-wise median imputation.
        """

        df = data.copy()

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:

            if df[column].isna().any():
                median_value = df[column].median()

                if pd.notna(median_value):
                    df[column] = df[column].fillna(
                        median_value
                    )

        return df

    def prepare(self, data):
        """
        Complete preprocessing pipeline.
        """

        df = self.clean(data)
        df = self.handle_missing_values(df)

        return df


def preprocess_data(data):
    """
    Simple interface for preprocessing data.
    """

    preprocessor = DataPreprocessor()

    return preprocessor.prepare(data)