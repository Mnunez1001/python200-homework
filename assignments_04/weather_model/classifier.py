from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class Prediction:
    """Representing a running-weather label and the probability of good weather."""

    label: str
    probability: float


class WeatherClassifier:
    """Loads a saved pipeline and classify daily running weather.

    Raises:
        FileNotFoundError: If the saved model file does not exist.
    """

    FEATURES: tuple[str, ...] = (
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    )

    def __init__(self, model_path: str | Path) -> None:
        """Loads the saved pipeline once.

        Args:
            model_path: Path to the saved classifier pipeline.

        Raises:
            FileNotFoundError: If model_path does not point to a file.
        """
        path = Path(model_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Weather classifier model file not found: {path}. "
                "Copy weather_classifier.pkl from the Week 3 models folder "
                "or run the Week 3 training script to create it."
            )

        self.pipeline = joblib.load(path)

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Classify daily weather records in their original order.

        Args:
            days: Weather dictionaries containing all four required features.
                Temperatures must be in Celsius, precipitation in millimeters,
                and wind speed in kilometers per hour.

        Returns:
            One Prediction per input day, preserving input order.
            Each probability represents the probability of good weather,
            even when the predicted label is skip.
            Returns an empty list when days is empty.

        Raises:
            ValueError: If any day is missing required features.
        """
        if not days:
            return []

        missing_details: list[str] = []

        for index, day in enumerate(days):
            missing = [
                feature for feature in self.FEATURES if feature not in day
            ]

            if missing:
                missing_details.append(
                    f"day at index {index}: {', '.join(missing)}"
                )

        if missing_details:
            raise ValueError(
                "Missing required features — "
                + "; ".join(missing_details)
            )

        # Explicitly selecting columns in the same order used during training.
        # Additional dictionary keys are ignored.
        X = pd.DataFrame(days)[list(self.FEATURES)]

        labels = self.pipeline.predict(X)
        good_probabilities = self.pipeline.predict_proba(X)[:, 1]

        return [
            Prediction(
                label="good" if label == 1 else "skip",
                probability=float(probability),
            )
            for label, probability in zip(
                labels, good_probabilities, strict=True
            )
        ]