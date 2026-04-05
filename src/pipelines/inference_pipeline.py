import yaml
import pandas as pd
from loguru import logger

from src.data_processing.processor import DataProcessor
from src.models.clustering import CustomerClustering
from src.utils.logging import setup_logging


def run_inference(
    input_csv: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Orchestrates the inference pipeline.

    Args:
        input_csv:   Path to the raw transaction CSV to segment.
        config_path: Path to the YAML configuration file.

    Returns:
        Customer-level DataFrame with a ``cluster`` column appended.
    """
    setup_logging(config_path)
    logger.info(f"Starting Inference Pipeline for {input_csv}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 1. Transform data using the persisted scaler + top-items list.
    processor = DataProcessor(config)
    customer_df, scaled_data = processor.transform_pipeline(input_csv)

    # 2. Predict clusters.
    model = CustomerClustering(config)
    clusters = model.predict(scaled_data)
    customer_df["cluster"] = clusters

    logger.info("Inference complete.")
    return customer_df
