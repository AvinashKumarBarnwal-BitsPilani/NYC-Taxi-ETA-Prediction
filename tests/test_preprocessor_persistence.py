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
        "vendor_id": [1.0, 2.0],
        "passenger_count": [1, 2],
        "store_and_fwd_flag": ["N", "Y"],
        "pickup_hour": [8, 18],
        "pickup_day_of_week": [1, 6],
        "pickup_month": [4, 5],
        "is_weekend": [0, 1],
        "distance_km": [5.2, 10.8],
    }
)

    # Reuse the fitted transformer; do NOT fit again.
    X_processed = preprocessor.transform(X_new)

    assert X_processed.shape == (2, 10)

    assert list(preprocessor.get_feature_names_out()) == [
    "numerical__distance_km",
    "passthrough_numerical__passenger_count",
    "passthrough_numerical__pickup_hour",
    "passthrough_numerical__pickup_day_of_week",
    "passthrough_numerical__pickup_month",
    "passthrough_numerical__is_weekend",
    "categorical__vendor_id_1",
    "categorical__vendor_id_2",
    "categorical__store_and_fwd_flag_N",
    "categorical__store_and_fwd_flag_Y",
]