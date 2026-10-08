import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse


def small_response() -> dict:
    """Return a fresh, valid response with three hourly observations."""
    return {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            "temperature_2m": [10.0, 12.0, 14.0],
            "precipitation": [0.0, 0.5, 1.0],
        },
    }


def test_real_response_validates() -> None:
    """Validate the real JSON file and check its observation count."""
    # A plain relative path depends on the terminal's working directory.
    # This path instead locates the JSON relative to this test file.
    data_path = Path(__file__).parent.parent / "weather_raw.json"

    with data_path.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    response = WeatherResponse.model_validate(raw_data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude_raises() -> None:
    """Reject a latitude outside the allowed range."""
    raw_data = small_response()
    raw_data["latitude"] = 200.0

    with pytest.raises(ValidationError) as caught:
        WeatherResponse.model_validate(raw_data)

    assert caught.value.errors()[0]["loc"] == ("latitude",)


def test_mismatched_lengths_raise() -> None:
    """Reject hourly arrays with unequal lengths."""
    raw_data = small_response()
    raw_data["hourly"]["temperature_2m"].pop()

    with pytest.raises(ValidationError, match="same length"):
        WeatherResponse.model_validate(raw_data)


def test_null_temperature_raises() -> None:
    """Reject a null temperature at a specific hourly index."""
    raw_data = small_response()
    raw_data["hourly"]["temperature_2m"][1] = None

    # JSON null becomes Python None when JSON is loaded.
    with pytest.raises(ValidationError) as caught:
        WeatherResponse.model_validate(raw_data)

    assert caught.value.errors()[0]["loc"] == (
        "hourly",
        "temperature_2m",
        1,
    )