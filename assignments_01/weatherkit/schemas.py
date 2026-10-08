from typing import Self

from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Represent aligned hourly timestamps, temperatures, and precipitation."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_list_lengths(self) -> Self:
        """Reject hourly arrays whose lengths do not match."""
        n_time = len(self.time)
        n_temperature = len(self.temperature_2m)
        n_precipitation = len(self.precipitation)

        if not (n_time == n_temperature == n_precipitation):
            raise ValueError(
                "Hourly lists must have the same length: "
                f"time={n_time}, "
                f"temperature_2m={n_temperature}, "
                f"precipitation={n_precipitation}."
            )

        return self


class WeatherResponse(BaseModel):
    """Represent weather API metadata and its hourly measurements."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock

    