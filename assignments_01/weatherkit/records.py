from dataclasses import dataclass

from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    """Represent one hourly observation.

    timestamp is the API timestamp string in the response's timezone.
    temperature_c is measured in degrees Celsius.
    precipitation_mm is measured in millimeters.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert validated hourly arrays into individual records in source order.

    Args:
        response: A validated weather response whose hourly arrays have
            matching lengths.

    Returns:
        A list containing one HourlyReading per hourly observation,
        preserving the order of the response's hourly arrays.
    """
    hourly = response.hourly

    return [
        HourlyReading(
            timestamp=timestamp,
            temperature_c=temperature,
            precipitation_mm=precipitation,
        )
        #zip() pairs values at the same index, producing a timestamp, temperature, and precipitation value for each hour.
        for timestamp, temperature, precipitation in zip(
            hourly.time,
            hourly.temperature_2m,
            hourly.precipitation,
            strict=True,
        )
    ]


# The boundary is where external API data enters our program through
# WeatherResponse.model_validate(raw_data). Pydantic checks field types,
# coordinate ranges, and the lengths of the hourly arrays there.
# HourlyReading holds internal data created from that validated response,
# so a dataclass supplies __init__, __repr__, and field-based equality
# without repeating Pydantic validation for every hourly record.
# Dataclass type hints describe the fields but do not validate them at runtime.