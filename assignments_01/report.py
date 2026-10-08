import json
from pathlib import Path

from weatherkit import DailyAggregator, WeatherResponse, to_readings


def main() -> None:
    """Validate hourly weather data and print daily summaries."""
    data_path = Path(__file__).resolve().parent / "weather_raw.json"

    with data_path.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    weather = WeatherResponse.model_validate(raw_data)
    readings = to_readings(weather)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    dropped_dates = aggregator.incomplete_days(readings)

    print("Weather report")
    print(f"Location: {weather.latitude}, {weather.longitude}")
    print(f"Timezone: {weather.timezone}")
    print(f"Hourly observations: {len(readings)}")
    print(f"Days reported: {len(summaries)}")

    print()
    print(
        f"{'Date':<12}"
        f"{'High (°C)':>12}"
        f"{'Low (°C)':>12}"
        f"{'Precip (mm)':>14}"
        f"{'Range (°C)':>14}"
    )
    print("-" * 64)

    for summary in summaries:
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>12.1f}"
            f"{summary.temp_min:>12.1f}"
            f"{summary.precipitation_sum:>14.1f}"
            f"{summary.temp_range():>14.1f}"
        )

    print()
    if dropped_dates:
        print(
            "WARNING: Dropped days with fewer than "
            f"{aggregator.min_hours} observations: "
            + ", ".join(dropped_dates)
        )
    else:
        print("Incomplete days dropped: none.")


# The guard runs main() only when this file is executed directly.
# If main() were called without the guard, importing report to reuse
# a helper function would also load the JSON, run the pipeline, and
# print the report as an unintended side effect.
if __name__ == "__main__":
    main()


# --- Task 7: Reflection ---
#
# 1. Rejecting null temperatures
# Rejecting the whole file makes sense when a downstream process requires
# complete data, such as preparing a strictly checked dataset for model
# training. It prevents missing measurements from entering that process.
# For a live weather dashboard, I would rather tolerate one missing
# temperature and still display the available observations.
# To allow gaps, I would change the HourlyBlock field to:
# temperature_2m: list[float | None]
# That schema change alone is not enough. I would also update conversion
# and aggregation to handle missing temperatures explicitly. For example,
# I could skip observations with missing temperatures, count only the
# usable observations, and report any resulting incomplete days.
#
# 2. Running the pipeline at noon
# If the input contains only observations collected so far, the current
# day may have about 12 readings at noon. With min_hours=24, that day is
# excluded even though its data is complete up to the current time.
# incomplete_days() makes the exclusion visible so the pipeline can
# warn users and process the day again when more observations arrive.
# Lowering the threshold could provide a partial-day summary, but its
# maximum and minimum would not represent the entire day.
#
# 3. Reusing the package in Week 10
# A pipeline can import WeatherResponse, to_readings, and DailyAggregator
# directly from weatherkit without copying their code or running report.py.
# This lets the pipeline reuse the tested validation and aggregation
# while handling API requests and database storage separately.