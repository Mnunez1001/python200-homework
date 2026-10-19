import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    """Loads the saved weather classifier and classifies hypothetical days."""
    assignment_dir = Path(__file__).resolve().parent
    model_dir = assignment_dir / "models"

    model_path = model_dir / "weather_classifier.pkl"
    metadata_path = model_dir / "weather_classifier_metadata.json"

    missing_files = [
        path.name
        for path in (model_path, metadata_path)
        if not path.is_file()
    ]

    if missing_files:
        raise SystemExit(
            "Cannot run predictions. Missing model files: "
            + ", ".join(missing_files)
            + ". Run 'uv run python train_classifier.py' "
            "from assignments_03 first."
        )

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    pipeline = joblib.load(model_path)
    features = metadata["feature_names"]

    print("--- Saved Model ---")
    print("City:", metadata["city"])
    print("Model:", metadata["model_type"])
    print("Features:", ", ".join(features))
    print(f"Test accuracy: {metadata['test_accuracy']:.4f}")

    # Inputs use °C, mm of precipitation, and km/h of wind.
    hypothetical_days = [
        {
            "name": "Mild and calm",
            "temperature_2m_max": 18.0,
            "temperature_2m_min": 8.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 8.0,
        },
        {
            "name": "Hot, wet, and windy",
            "temperature_2m_max": 40.0,
            "temperature_2m_min": 27.0,
            "precipitation_sum": 8.0,
            "wind_speed_10m_max": 45.0,
        },
        {
            "name": "Borderline warm",
            "temperature_2m_max": 26.0,
            "temperature_2m_min": 15.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 15.0,
        },
        {
            "name": "Borderline windy",
            "temperature_2m_max": 20.0,
            "temperature_2m_min": 10.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 29.0,
        },
    ]

    # Selecting and ordering columns using the saved training metadata.
    # The descriptive name is not passed into the model.
    days = pd.DataFrame(hypothetical_days)
    X_new = days[features]

    predictions = pipeline.predict(X_new)
    probabilities = pipeline.predict_proba(X_new)

    # Probability columns follow the model's class order.
    good_column = list(pipeline.classes_).index(1)
    good_probabilities = probabilities[:, good_column]

    print("\n--- Hypothetical Day Predictions ---")

    for index, day in enumerate(hypothetical_days):
        label = "good" if predictions[index] == 1 else "skip"

        print(f"\n{day['name']}:")
        for feature in features:
            print(f"  {feature}: {X_new.iloc[index][feature]:.1f}")

        print("  Predicted label:", label)
        print(f"  Probability of good: {good_probabilities[index]:.4f}")


if __name__ == "__main__":
    main()


# --- Task 8: Reflection ---
#
# 1. Borderline prediction
# For "Borderline warm," the model estimated P(good) = 0.4366, or 43.66%,
# and predicted skip. This is relatively close to 0.5, so the model
# favors skip without a strong preference.
# The day actually satisfies our labeling rule: 26°C is included,
# the low is above 0°C, precipitation is under 3 mm, and wind is under
# 30 km/h. This disagreement shows that the classifier approximates
# the rule rather than reproducing it perfectly.
#
# If the model returned 0.52, it would predict good at the default
# threshold, but I would present it as an uncertain recommendation
# and show the weather conditions so the user could decide.
# These probabilities are model estimates; their calibration has
# not been tested.
#
# 2. Missing saved files
# If predict.py ran before train_classifier.py, the saved model and
# metadata would not exist. Loading them directly would raise
# FileNotFoundError.
# Our script checks for both files before loading and gives a message
# naming the missing files and explaining how to run the training script.
#
# 3. Cloud pipeline
# Replace the hard-coded hypothetical days with weather records
# supplied by the cloud pipeline.
# Validate that incoming records contain the expected features and
# use the same units as training, then select columns in metadata order.
# Store the predicted labels and probabilities in a database rather
# than only printing them.