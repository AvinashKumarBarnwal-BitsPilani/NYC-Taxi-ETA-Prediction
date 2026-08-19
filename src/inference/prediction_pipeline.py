"""Phase 5.1 - Reusable prediction pipeline."""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
logger = get_logger(__name__)

DEFAULT_MODEL_PATH = (
    "artifacts/final/model/final_model.joblib"
)

DEFAULT_PREPROCESSOR_PATH = (
    "artifacts/final/preprocessing/preprocessor.joblib"
)

class PredictionPipeline:
    """Reusable inference pipeline for NYC Taxi ETA prediction."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

        self.model = None
        self.preprocessor = None

    def load_artifacts(self) -> None:
        """Load persisted model and preprocessor artifacts."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}"
            )

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                "Preprocessor artifact not found: "
                f"{self.preprocessor_path}"
            )

        logger.info(
            "Loading final model: %s",
            self.model_path,
        )

        self.model = joblib.load(self.model_path)

        logger.info(
            "Loading preprocessor: %s",
            self.preprocessor_path,
        )

        self.preprocessor = joblib.load(
            self.preprocessor_path
        )

        logger.info("Prediction artifacts loaded successfully")

    def predict(
        self,
        features: pd.DataFrame,
    ) -> float:
        """Generate an ETA prediction for one input record."""

        if self.model is None or self.preprocessor is None:
            self.load_artifacts()

        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                "Prediction input must be a pandas DataFrame."
            )

        transformed_features = self.preprocessor.transform(
            features
        )

        prediction = self.model.predict(
            transformed_features
        )

        return float(np.asarray(prediction).reshape(-1)[0])