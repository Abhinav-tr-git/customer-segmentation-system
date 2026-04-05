import joblib
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from loguru import logger


class CustomerClustering:
    """Wraps KMeans + PCA with clean train/predict/persist semantics."""

    def __init__(self, config: dict):
        model_cfg = config["model"]
        self.n_clusters: int = model_cfg["n_clusters"]
        self.random_state: int = model_cfg["random_state"]
        self.pca_components: int = model_cfg["pca_components"]

        self.model: Optional[KMeans] = None
        self.pca: Optional[PCA] = None

        self.model_path = Path(model_cfg["model_save_path"])
        self.pca_path = Path(model_cfg["pca_save_path"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialize_fresh_models(self) -> None:
        """Instantiates untrained KMeans and PCA objects."""
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.pca = PCA(n_components=self.pca_components)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, data: np.ndarray) -> None:
        """Fits PCA then KMeans; persists both artifacts to disk."""
        logger.info(f"Training KMeans with {self.n_clusters} clusters")
        self._initialize_fresh_models()

        logger.info(f"Applying PCA with {self.pca_components} components")
        pca_data = self.pca.fit_transform(data)

        self.model.fit(pca_data)
        logger.info("Model training complete.")
        self.save_artifacts()

    def load_artifacts(self) -> None:
        """Loads model and PCA from disk if not already in memory."""
        if self.model is not None and self.pca is not None:
            return  # Already loaded — no-op.

        logger.info("Loading model artifacts from disk...")
        if not self.model_path.exists() or not self.pca_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found. Run training before inference."
            )
        self.model = joblib.load(self.model_path)
        self.pca = joblib.load(self.pca_path)
        logger.info("Artifacts loaded successfully.")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Applies PCA then KMeans prediction. Loads artifacts if needed."""
        self.load_artifacts()
        pca_data = self.pca.transform(data)
        return self.model.predict(pca_data)

    def save_artifacts(self) -> None:
        """Persists in-memory models to disk.

        Raises:
            RuntimeError: If called before training (model or pca is None).
        """
        if self.model is None or self.pca is None:
            raise RuntimeError(
                "save_artifacts() called before training. "
                "Call train() first."
            )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.pca, self.pca_path)
        logger.info(f"Model artifacts saved to {self.model_path.parent}")
