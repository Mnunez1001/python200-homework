import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
#from __future__ import annotations
import pytest


# --- Q1: Load Model Information Once ---

class ModelInfo:
    """Load and store model metadata from a JSON file."""

    def __init__(self, path: str | Path) -> None:
        """Reading the JSON file once and store its contents."""
        with open(path, encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        """Returning the number of features listed in the metadata."""
        return len(self.data["features"])


# Loading in __init__ avoids repeated file reads and JSON parsing when
# feature_count() is called many times; it reuses the stored data.


# --- Q2: Forecast Dataclass --------------------------------------------------------------------------------

@dataclass
class DayForecast:
    """Represent a day's date, high temperature in Celsius, and rain flag."""

    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        """Returning a readable description of the forecast."""
        rain_description = (
            "rain expected" if self.will_rain else "no rain expected"
        )
        return (
            f"{self.date}: high {self.high_c:.1f}C, "
            f"{rain_description}"
        )


# @dataclass generated __init__ for assigning the three fields.
# It also generated methods such as __repr__ and field-based __eq__.


# --- Q3: Temperature Parsing and Tests --------------------------------------------------------------------

def parse_temperature(text: str) -> float:
    """Converting numeric text to a float or raising a temperature error."""
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid temperature: {text!r}. Expected a number."
        ) from error


def test_parse_temperature_valid() -> None:
    """Accept a numeric temperature string."""
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises() -> None:
    """Reject nonnumeric text with a useful temperature error."""
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# pytest.raises(ValueError) alone would accept a ValueError with an
# unrelated or unhelpful message. match="temperature" also checks that
# the message includes the expected word.


# --- Demonstrations --------------------------------------------------------------------------------------

def main() -> None:
    """Running the warmup demonstrations and print their outputs."""
    print("--- Q1: ModelInfo ---")

    # Creating a small temporary JSON file for exercise 1.
    with TemporaryDirectory() as temporary_directory:
        metadata_path = Path(temporary_directory) / "model_info.json"

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"features": ["high_c", "low_c", "precipitation_mm"]},
                file,
            )

        model_info = ModelInfo(metadata_path)
        print("Feature count:", model_info.feature_count())

    # The temporary file is now removed, but the loaded data remains available.
    print("Feature count from stored data:", model_info.feature_count())

    print("\n--- Q2: DayForecast ---")

    dry_day = DayForecast("2023-06-15", 24.1, False)
    rainy_day = DayForecast("2023-06-16", 19.5, True)

    print(dry_day.summary())
    print(rainy_day.summary())

    print("\n--- Q3: parse_temperature ---")
    print("Parsed temperature:", parse_temperature("24.1"))

    try:
        parse_temperature("hot")
    except ValueError as error:
        print("Caught error:", error)


if __name__ == "__main__":
    main()