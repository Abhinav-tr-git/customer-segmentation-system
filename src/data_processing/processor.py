import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List
from loguru import logger
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    # Column dtypes for safe loading — Quantity/CustomerID use float to handle
    # NaN values that appear before the cleaning step, avoiding a crash.
    _LOAD_DTYPES = {
        "InvoiceNo":   str,
        "StockCode":   str,
        "Description": str,
        "Quantity":    float,   # BUG FIX: was int — crashes on NaN rows pre-clean
        "UnitPrice":   float,
        "CustomerID":  float,
        "Country":     str,
    }

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.scaler_path = Path(config["model"]["scaler_save_path"])
        # Persist the top-N items list so train/infer always use the same columns.
        self.top_items_path = self.scaler_path.parent / "top_items.joblib"
        self._top_items: List[str] = []

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads raw transaction CSV with robust encoding handling."""
        path = Path(file_path)
        logger.info(f"Loading data from {path}")
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        # latin-1 encoding handles special European characters (é, ü, etc.)
        # that are invalid in UTF-8, common in UCI Online Retail dataset.
        # engine='python' + on_bad_lines='warn' tolerates rows with embedded
        # commas in Description fields without crashing.
        return pd.read_csv(
            path,
            dtype=self._LOAD_DTYPES,
            encoding="latin-1",
            encoding_errors="replace",
            engine="python",
            on_bad_lines="warn",
            quotechar='"',
        )

    def save_scaler(self) -> None:
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")

    def load_scaler(self) -> None:
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Run training first.")
        self.scaler = joblib.load(self.scaler_path)

    def save_top_items(self) -> None:
        joblib.dump(self._top_items, self.top_items_path)
        logger.info(f"Top-items list saved to {self.top_items_path}")

    def load_top_items(self) -> None:
        if not self.top_items_path.exists():
            raise FileNotFoundError(
                f"Top-items list not found at {self.top_items_path}. Run training first."
            )
        self._top_items = joblib.load(self.top_items_path)

    # ------------------------------------------------------------------
    # Feature-engineering steps
    # ------------------------------------------------------------------

    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops invalid rows and derives the Sales column."""
        logger.info("Cleaning transaction data")
        df = df[df["CustomerID"].notnull()].copy()
        # Cast is safe here because nulls are already removed.
        df["CustomerID"] = df["CustomerID"].astype(int)
        df["Quantity"] = df["Quantity"].astype(int)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], cache=True)
        df["Sales"] = df["Quantity"] * df["UnitPrice"]
        logger.info(f"Cleaned data shape: {df.shape}")
        return df

    def _compute_top_items(self, df: pd.DataFrame) -> List[str]:
        """Returns the 20 most-frequently purchased StockCodes."""
        return df["StockCode"].value_counts().head(20).index.tolist()

    def extract_item_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Builds a customer × top-product pivot table.

        During training  → derives and saves the top-20 list.
        During inference → uses the saved list (same columns, same order).
        """
        logger.info("Extracting item-level features")
        pivot = (
            df[df["StockCode"].isin(self._top_items)]
            .pivot_table(
                index="CustomerID",
                columns="StockCode",
                values="Quantity",
                aggfunc="sum",
                fill_value=0,
            )
        )
        # Prefix columns and reindex to guarantee column order matches training.
        pivot.columns = [f"item_{c}" for c in pivot.columns]
        expected_cols = [f"item_{c}" for c in self._top_items]
        pivot = pivot.reindex(columns=expected_cols, fill_value=0)
        return pivot

    def aggregate_to_customer_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates transactions to one row per customer (RFM + items)."""
        logger.info("Aggregating to customer level")
        reference_date = df["InvoiceDate"].max()

        rfm_df = df.groupby("CustomerID").agg(
            recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            frequency=("InvoiceNo", "nunique"),
            monetary=("Sales", "sum"),
            unique_products=("StockCode", "nunique"),
        )

        item_features = self.extract_item_features(df)
        final_df = rfm_df.join(item_features, how="left").fillna(0)
        final_df = final_df[final_df["monetary"] > 0]
        logger.info(f"Customer-level data shape: {final_df.shape}")
        return final_df

    def scale_features(self, df: pd.DataFrame, training: bool = True) -> np.ndarray:
        """Fits (training) or applies (inference) the StandardScaler."""
        logger.info("Scaling features")
        if training:
            scaled = self.scaler.fit_transform(df)
            self.save_scaler()
        else:
            self.load_scaler()
            scaled = self.scaler.transform(df)
        return scaled

    # ------------------------------------------------------------------
    # Public pipeline entry-points
    # ------------------------------------------------------------------

    def fit_transform_pipeline(self, raw_data_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Full training pipeline: load → clean → aggregate → fit-scale.

        Persists the scaler and top-items list for inference-time reuse.
        """
        raw_df = self.load_data(raw_data_path)
        clean_df = self.clean_transactions(raw_df)

        # Derive and save the top-20 items from the training set.
        self._top_items = self._compute_top_items(clean_df)
        self.save_top_items()

        customer_df = self.aggregate_to_customer_level(clean_df)
        scaled = self.scale_features(customer_df, training=True)
        return customer_df, scaled

    def transform_pipeline(self, raw_data_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Inference-only pipeline: loads saved artifacts then transforms.

        The top-items list and scaler must already be persisted from training.
        """
        raw_df = self.load_data(raw_data_path)
        clean_df = self.clean_transactions(raw_df)

        # Load the training-time top-20 list to guarantee column alignment.
        self.load_top_items()

        customer_df = self.aggregate_to_customer_level(clean_df)
        scaled = self.scale_features(customer_df, training=False)
        return customer_df, scaled
