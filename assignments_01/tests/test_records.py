import pytest

from weatherkit import HourlyReading, WeatherResponse, to_readings


@pytest.fixture
def response() -> WeatherResponse:
    """Provide three observations with distinct values."""
    return WeatherResponse.model_validate(
        {
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
                "temperature_2m": [10.0, 20.0, 30.0],
                "precipitation": [0.0, 0.5, 1.5],
            },
        }
    )


def test_to_readings_preserves_count_and_order( response: WeatherResponse,) -> None:
    """Return one reading per hour in the original timestamp order."""
    readings = to_readings(response)

    assert len(readings) == len(response.hourly.time)
    assert readings[0].timestamp == response.hourly.time[0]
    assert readings[-1].timestamp == response.hourly.time[-1]
    assert [reading.timestamp for reading in readings] == response.hourly.time


@pytest.mark.parametrize("index", [0, 1, 2])

def test_to_readings_matches_input_index(response: WeatherResponse, index: int) -> None:
    """Keep each timestamp paired with its corresponding measurements."""
    reading = to_readings(response)[index]

    assert reading.timestamp == response.hourly.time[index]
    assert reading.temperature_c == response.hourly.temperature_2m[index]
    assert reading.precipitation_mm == response.hourly.precipitation[index]


def test_hourly_readings_compare_equal() -> None:
    """Compare dataclass instances by their field values."""
    reading_a = HourlyReading("2026-04-08T00:00", 10.0, 0.5)
    reading_b = HourlyReading("2026-04-08T00:00", 10.0, 0.5)

    assert reading_a is not reading_b
    assert reading_a == reading_b