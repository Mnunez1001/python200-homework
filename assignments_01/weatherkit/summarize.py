from dataclasses import dataclass

from .records import HourlyReading


@dataclass
class DailySummary:
    """Represent daily temperatures in °C and total precipitation in mm."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between maximum and minimum temperatures."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Summarize hourly readings for days meeting an observation threshold."""

    def __init__(self, min_hours: int = 24) -> None:
        """Setting the minimum number of observations needed to report a day."""
        self.min_hours = min_hours

    def _group_by_date(self, readings: list[HourlyReading] ) -> dict[str, list[HourlyReading]]:
        """Group readings by the date portion of their timestamps."""
        grouped: dict[str, list[HourlyReading]] = {}

        for reading in readings:
            date = reading.timestamp[:10]
            grouped.setdefault(date, []).append(reading)

        return grouped

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Compute summaries for qualifying days, sorted by date.

        Args:
            readings: Hourly observations to group by calendar date.

        Returns:
            Daily summaries excluding days with fewer than min_hours
            observations.
        """
        grouped = self._group_by_date(readings)
        summaries: list[DailySummary] = []

        for date, day_readings in sorted(grouped.items()):
            if len(day_readings) < self.min_hours:
                continue

            temperatures = [
                reading.temperature_c for reading in day_readings
            ]

            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(temperatures),
                    temp_min=min(temperatures),
                    precipitation_sum=sum(
                        reading.precipitation_mm for reading in day_readings
                    ),
                    hours_observed=len(day_readings),
                )
            )

        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return sorted dates excluded for having too few observations.

        Args:
            readings: Hourly observations to group by calendar date.

        Returns:
            Dates present in the readings with fewer than min_hours
            observations.
        """
        grouped = self._group_by_date(readings)

        return sorted(
            date
            for date, day_readings in grouped.items()
            if len(day_readings) < self.min_hours
        )