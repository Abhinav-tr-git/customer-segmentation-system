import yaml
from pathlib import Path
from typing import Optional
from loguru import logger

from src.data_processing.processor import DataProcessor
from src.models.clustering import CustomerClustering
from src.utils.logging import setup_logging


def run_training(
    config_path: str = "config/config.yaml",
    input_path: Optional[str] = None,
) -> None:
    """Orchestrates the training pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        input_path: Override for the raw data CSV. Falls back to
                    ``config['data']['raw_path']`` when not supplied.
    """
    setup_logging(config_path)
    logger.info("Starting Training Pipeline")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_data_path = input_path or config["data"]["raw_path"]
    logger.info(f"Training data: {raw_data_path}")

    # 1. Process data and persist scaler + top-items list.
    processor = DataProcessor(config)
    customer_df, scaled_data = processor.fit_transform_pipeline(raw_data_path)

    # 2. Train model and persist artifacts.
    model = CustomerClustering(config)
    model.train(scaled_data)

    # 3. Predict and save labelled results.
    clusters = model.predict(scaled_data)
    customer_df["cluster"] = clusters

    output_path = Path(config["data"]["processed_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    customer_df.to_csv(output_path)
    logger.info(f"Training complete. Results saved to {output_path}")
