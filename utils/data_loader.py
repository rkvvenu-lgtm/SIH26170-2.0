from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Loads input datasets for the screening pipeline.
    Supports CSV and Excel files.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv",
        ".xlsx",
        ".xls"
    }

    def __init__(self):
        pass

    def _standardize_columns(self, data):
        data = data.copy()

        data.columns = [
            str(column).strip().replace(" ", "_")
            for column in data.columns
        ]

        return data

    def load_data(self, file_path):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Input data file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Input data path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: CSV, XLSX, XLS."
            )

        if extension == ".csv":
            data = pd.read_csv(path)

        elif extension in {".xlsx", ".xls"}:
            data = pd.read_excel(path)

        else:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        if data.empty:
            raise ValueError(
                "Input dataset is empty."
            )

        # Remove completely empty rows and columns
        data = data.dropna(
            axis=0,
            how="all"
        )

        data = data.dropna(
            axis=1,
            how="all"
        )

        # Standardize column names
        data = self._standardize_columns(data)

        return data

    # Backward-compatible alias
    def load(self, file_path):
        return self.load_data(file_path)