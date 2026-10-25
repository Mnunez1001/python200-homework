from pathlib import Path

from weather_model import WeatherClassifier


def main() -> None:
    """Classifying hypothetical days using the real saved weather pipeline."""
    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "weather_classifier.pkl"
    )

    classifier = WeatherClassifier(model_path)

    names = [
        "Mild and calm",
        "Hot, wet, and windy",
        "Borderline warm",
        "Borderline windy",
    ]

    days = [
        {
            "temperature_2m_max": 18.0,
            "temperature_2m_min": 8.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 8.0,
        },
        {
            "temperature_2m_max": 40.0,
            "temperature_2m_min": 27.0,
            "precipitation_sum": 8.0,
            "wind_speed_10m_max": 45.0,
        },
        {
            "temperature_2m_max": 26.0,
            "temperature_2m_min": 15.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 15.0,
        },
        {
            "temperature_2m_max": 20.0,
            "temperature_2m_min": 10.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 29.0,
        },
    ]

    predictions = classifier.predict(days)

    print("--- Las Vegas Running Weather Predictions ---")

    for name, day, prediction in zip(
        names, days, predictions, strict=True
    ):
        print(
            f"\n{name}: "
            f"high={day['temperature_2m_max']:.1f}°C, "
            f"low={day['temperature_2m_min']:.1f}°C, "
            f"precipitation={day['precipitation_sum']:.1f} mm, "
            f"wind={day['wind_speed_10m_max']:.1f} km/h "
            f"-> {prediction.label}, "
            f"P(good)={prediction.probability:.4f}"
        )


# Without the guard, an unconditional main() call would also run when
# another script imports this module to reuse a helper. That would
# unexpectedly load the model, make predictions, and print output.
if __name__ == "__main__":
    main()


# --- Task 6: Reflection ---
#
# 1. Fixed feature order
# Suppose a caller supplies wind speed before temperature in a dictionary.
# If we passed values to the model in that order, it could interpret
# 30 km/h of wind as a 30°C daily high, producing incorrect predictions.
# Our component explicitly selects the training feature order, so the
# caller's dictionary insertion order does not affect the model input.
# With named DataFrames, scikit-learn may instead reject mismatched
# column order; explicit selection prevents that error as well.
#
# 2. Temporary fixture model
# Training a tiny model inside a fixture makes the tests independent
# of the real model file, its training data, and changes to its predictions.
# pytest's tmp_path provides an isolated location for each test.
# The suite verifies loading, label mapping, probabilities, ordering,
# and input errors using predictable examples.
# A separate integration check can verify the real saved model.
#
# 3. Cloud readiness
# WeatherClassifier loads its pipeline once and accepts batches of
# dictionaries, returning structured Prediction objects in input order.
# It does not require API requests or retraining to make predictions,
# so a database pipeline can call it directly.
# Before production, I would add input validation for nonnumeric,
# missing-value, and infinite measurements, with errors identifying
# the affected record and feature.