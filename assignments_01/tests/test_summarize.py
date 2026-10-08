import pytest

from weatherkit import DailyAggregator, HourlyReading


@pytest.fixture
def hourly_readings() -> list[HourlyReading]:
    """Provides three observations on one date and two on another."""
    # Deliberately placing the later date first to test sorting as well.
    return [
        HourlyReading("2026-04-09T00:00", 20.0, 1.2),
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 15.0, 0.2),
        HourlyReading("2026-04-09T01:00", 25.0, 0.3),
        HourlyReading("2026-04-08T02:00", 5.0, 0.0),
    ]


def test_groups_by_date(hourly_readings: list[HourlyReading]) -> None:
    """Group observations into two summaries sorted by date."""
    summaries = DailyAggregator(min_hours=2).summarize(hourly_readings)

    assert len(summaries) == 2
    assert [summary.date for summary in summaries] == [
        "2026-04-08",
        "2026-04-09",
    ]
    assert [summary.hours_observed for summary in summaries] == [3, 2]


def test_temperature_extremes(hourly_readings: list[HourlyReading],) -> None:
    """Compute each day's maximum and minimum temperatures."""
    summaries = DailyAggregator(min_hours=2).summarize(hourly_readings)

    assert summaries[0].temp_max == 15.0
    assert summaries[0].temp_min == 5.0
    assert summaries[1].temp_max == 25.0
    assert summaries[1].temp_min == 20.0


def test_precipitation_sum(hourly_readings: list[HourlyReading],) -> None:
    """Add precipitation within each date."""
    summaries = DailyAggregator(min_hours=2).summarize(hourly_readings)

    assert summaries[0].precipitation_sum == pytest.approx(0.3)
    assert summaries[1].precipitation_sum == pytest.approx(1.5)


def test_incomplete_day_is_dropped(hourly_readings: list[HourlyReading],) -> None:
    """Exclude and report a date below the observation threshold."""
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(hourly_readings)

    assert [summary.date for summary in summaries] == ["2026-04-08"]
    assert aggregator.incomplete_days(hourly_readings) == ["2026-04-09"]


def test_lower_threshold_keeps_same_day(hourly_readings: list[HourlyReading],) -> None:
    """Keep the previously excluded date when the threshold is lowered."""
    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(hourly_readings)

    assert [summary.date for summary in summaries] == [
        "2026-04-08",
        "2026-04-09",
    ]
    assert aggregator.incomplete_days(hourly_readings) == []


def test_temperature_range(hourly_readings: list[HourlyReading],) -> None:
    """Calculate the difference between daily temperature extremes."""
    summaries = DailyAggregator(min_hours=2).summarize(hourly_readings)

    assert summaries[0].temp_range() == pytest.approx(10.0)
    assert summaries[1].temp_range() == pytest.approx(5.0)


def test_default_threshold_requires_24_observations(hourly_readings: list[HourlyReading],) -> None:
    """Exclude both small sample days under the default threshold."""
    aggregator = DailyAggregator()

    assert aggregator.summarize(hourly_readings) == []
    assert aggregator.incomplete_days(hourly_readings) == [
        "2026-04-08",
        "2026-04-09",
    ]

# Deliberate break check: changed temp_range() from subtraction to addition.
# test_temperature_range caught the change: it received 20.0 instead of 10.0.
# Restored subtraction and reran the suite successfully.

#tests\test_summarize.py::test_temperature_range FAILED                                                               [100%]

#======================================================== FAILURES =========================================================
#_________________________________________________ test_temperature_range __________________________________________________

#hourly_readings = [HourlyReading(timestamp='2026-04-09T00:00', temperature_c=20.0, precipitation_mm=1.2), HourlyReading(timestamp='2026-...re_c=25.0, precipitation_mm=0.3), HourlyReading(timestamp='2026-04-08T02:00', temperature_c=5.0, precipitation_mm=0.0)]

#    def test_temperature_range(hourly_readings: list[HourlyReading],) -> None:
#        """Calculate the difference between daily temperature extremes."""
#        summaries = DailyAggregator(min_hours=2).summarize(hourly_readings)
    
#>       assert summaries[0].temp_range() == pytest.approx(10.0)
#E       assert 20.0 == 10.0 ± 1.0e-05
#E         
#E         comparison failed
#E         Obtained: 20.0
#E         Expected: 10.0 ± 1.0e-05

#tests\test_summarize.py:74: AssertionError
#================================================= short test summary info =================================================
#FAILED tests\test_summarize.py::test_temperature_range - assert 20.0 == 10.0 ± 1.0e-05
#==================================================== 1 failed in 0.25s ====================================================