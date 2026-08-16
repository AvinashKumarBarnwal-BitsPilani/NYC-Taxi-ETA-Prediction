import joblib
import pandas as pd

from src.pipelines.preprocessing import PROCESSED_DIR


def test_persisted_preprocessor_can_transform_new_data():
    """Verify persisted preprocessor can be loaded and reused for inference."""

    preprocessor_path = PROCESSED_DIR / "preprocessor.joblib"

    assert preprocessor_path.exists()

    preprocessor = joblib.load(preprocessor_path)

    # Verify the transformer was fitted on TRAIN data.
    assert hasattr(preprocessor, "transformers_")

    # Simulate new prediction-time feature input.
    X_new = pd.DataFrame(
        {
            "distance_km": [5.2, 10.8],
            "vendor_id": [1.0, 2.0],
            "store_and_fwd_flag": ["N", "Y"],
        }
    )

    # Reuse the fitted transformer; do NOT fit again.
    X_processed = preprocessor.transform(X_new)

    assert X_processed.shape == (2, 5)

    assert list(preprocessor.get_feature_names_out()) == [
        "numerical__distance_km",
        "categorical__vendor_id_1.0",
        "categorical__vendor_id_2.0",
        "categorical__store_and_fwd_flag_N",
        "categorical__store_and_fwd_flag_Y",
    ]