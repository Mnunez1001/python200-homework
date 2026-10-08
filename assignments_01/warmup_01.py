# --- Classes ---

# Q1
class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = list(readings) if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    # Q2
    def __repr__(self):
        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )


my_thermometer = Thermometer("Greensboro")

for reading in [21.0, 15.0, 55.0, 31.0]:
    my_thermometer.add(reading)

print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())

# average() must handle an empty list because its length would be zero.
# Without the check, calculating sum(readings) / len(readings)
# would raise ZeroDivisionError.


# Q2 demonstration
print("Thermometer object:", my_thermometer)

second_thermometer = Thermometer("Charlotte", [18.0, 20.0])
print("Thermometer list:", [my_thermometer, second_thermometer])

# Without a custom __repr__ (or __str__), Python displays something like
# <__main__.Thermometer object at 0x...>.
# This is unhelpful for debugging because it does not show the location,
# readings count, or average.


# Q3
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]


standard_alert = TemperatureAlert()
higher_alert = TemperatureAlert(33.0)

print("Readings above 30°C:", standard_alert.breaches(my_thermometer))
print("Readings above 33°C:", higher_alert.breaches(my_thermometer))

# The threshold is stored on TemperatureAlert because it represents
# that alert object's configuration.
# With twenty thermometers, we can reuse the same alert object for each
# thermometer without passing the threshold repeatedly. This keeps
# the checks consistent and reduces opportunities for mistakes.

# --- Dataclasses, Type Hints, and Docstrings -----------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field, FrozenInstanceError


# Q1 and Q2
@dataclass(frozen=True)
class Station:
    """Represent a weather station with coordinates and elevation in meters."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station("NC001", "Greensboro", 36.07, -79.79, 222.0)
station_b = Station("NC001", "Greensboro", 36.07, -79.79, 222.0)

print("Stations are equal:", station_a == station_b)

# A dataclass automatically generates __eq__, which compares field values.
# These objects have identical field values, so the result is True.
# With the original class, the result would be False because the default
# equality comparison checks object identity, and these are distinct objects.


# Q2
try:
    station_a.elevation = 300.0
except FrozenInstanceError as error:
    print("Frozen station error:", error)

station_c = Station("NC002", "Wichita", 21.60, -82.55, 620.0)

unique_stations = {station_a, station_b, station_c}
print("Number of unique stations:", len(unique_stations))

# With the default eq=True, frozen=True also generates __hash__.
# All our Station fields are hashable, so Station objects can be set members
# or dictionary keys. The set keeps only one of the two equal stations.


# Q3
@dataclass
class StationBatch:
    """Represent a regional collection of weather stations."""

    region: str
    stations: list[Station] = field(default_factory=list)  

    def add(self, station: Station) -> None:
        """Append a station to this batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda station: station.elevation)


# Using stations: list[Station] = [] raises:
# ValueError: mutable default <class 'list'> for field stations is not
# allowed: use default_factory
#
# Python refuses this mutable default because it would be shared across
# instances. field(default_factory=list) creates a new list for each batch.

#  raise ValueError(f'mutable default {type(f.default)} for field '
#ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory


batch = StationBatch("North Carolina")
print("Highest station in empty batch:", batch.highest())

batch.add(station_a)
batch.add(station_c)

print("Highest station:", batch.highest())
print("Stations in batch:", batch.stations)

another_batch = StationBatch("South Carolina")
print("Separate batch's stations:", another_batch.stations)

# --- Pydantic ---------------------------------------------------------------------------------------------------------------------------

from typing import Self

from pydantic import BaseModel, Field, ValidationError, model_validator


# Q1
class Reading(BaseModel):
    """Represent a validated weather reading from a station."""

    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    # Q4
    @model_validator(mode="after")
    def check_sensor_combination(self) -> Self:
        """Reject the temperature and humidity combination indicating failure."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError(
                "Possible sensor failure: humidity is 0.0 "
                "and temperature is below -40°C."
            )
        return self


valid_reading = Reading(
    station_id="NC001",
    timestamp="2026-10-06T14:00:00",
    temperature_c=24.5,
    humidity=55.0,
)

print("Q1 valid reading:", valid_reading)


# Q2
# Failure 1: missing required timestamp
try:
    Reading(
        station_id="NC001",
        temperature_c=24.5,
        humidity=55.0,
    )
except ValidationError as error:
    print("\nQ2 missing field:")
    print(error)


# Failure 2: temperature outside the allowed range
try:
    Reading(
        station_id="NC001",
        timestamp="2026-10-06T14:00:00",
        temperature_c=150.0,
        humidity=55.0,
    )
except ValidationError as error:
    print("\nQ2 invalid temperature:")
    print(error)


# Failure 3: humidity cannot be converted to a number
try:
    Reading(
        station_id="NC001",
        timestamp="2026-10-06T14:00:00",
        temperature_c=24.5,
        humidity="very humid",
    )
except ValidationError as error:
    print("\nQ2 invalid humidity:")
    print(error)


converted_reading = Reading(
    station_id="NC001",
    timestamp="2026-10-06T15:00:00",
    temperature_c="21.5",
    humidity=40,
)

print("\nQ2 converted reading:", converted_reading)
print("Temperature type:", type(converted_reading.temperature_c))
print("Humidity type:", type(converted_reading.humidity))

# In its default non-strict mode, Pydantic can convert numeric strings
# and integers into floats. "21.5" represents a number, but "very humid"
# does not. Converted values must still satisfy the field constraints.


# Q3
try:
    Reading(
        station_id="NC",
        temperature_c="not a number",
        humidity=50.0,
    )
except ValidationError as error:
    print("\nQ3 multiple errors:")
    for detail in error.errors():
        print("Location:", detail["loc"], "| Message:", detail["msg"])
    print("Number of errors:", len(error.errors()))

# Three errors are reported: station_id is too short, timestamp is
# missing, and temperature_c is not numeric.
# Reporting all errors at once lets us fix several input problems
# together instead of repeatedly fixing one and rerunning the program.


# Q4
working_sensor_reading = Reading(
    station_id="NC002",
    timestamp="2026-10-06T16:00:00",
    temperature_c=-45.0,
    humidity=25.0,
)

print("\nQ4 valid sensor reading:", working_sensor_reading)

try:
    Reading(
        station_id="NC002",
        timestamp="2026-10-06T16:00:00",
        temperature_c=-45.0,
        humidity=0.0,
    )
except ValidationError as error:
    print("\nQ4 failed sensor combination:")
    print(error)

# Field constraints alone check each field independently.
# Both -45.0°C and 0.0% humidity pass their individual limits,
# but this rule rejects their combination. A model validator
# can examine both fields together.

# --- pytest ----------------------------------------------------------------------------------------------------------------------------------

import pytest


# Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return celsius * (9 / 5) + 32


def test_celsius_to_fahrenheit() -> None:
    """Check freezing, boiling, and body-temperature conversions."""
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)


# Floating-point arithmetic can introduce tiny rounding differences.
# The formula above returns 98.60000000000001 for 37°C.
# pytest.approx allows a small tolerance instead of requiring exact equality.


# Q2
def mean(values: list[float]) -> float:
    """Return the arithmetic mean; raise ValueError if values is empty."""
    if not values:
        raise ValueError("Cannot calculate the mean of an empty list.")
    return sum(values) / len(values)


def test_mean_of_empty_raises() -> None:
    """Check that an empty list raises an error with a useful message."""
    with pytest.raises(ValueError, match="empty"):
        mean([])


# pytest.raises(ValueError) alone accepts any ValueError message.
# match="empty" also checks that the message contains the expected word,
# catching an incorrect or unhelpful message that omits it.


# Q3
@pytest.mark.parametrize(
    "values, expected",
    [
        ([10.0], 10.0),
        ([2.0, 4.0, 6.0], 4.0),
        ([-6.0, -3.0, 0.0], -3.0),
        ([-5.0, 5.0], 0.0),
    ],
)
def test_mean_values(values: list[float], expected: float) -> None:
    """Check the mean for several lists of values."""
    assert mean(values) == pytest.approx(expected)


# Parametrization reuses the same test logic for four separate cases.
# It reduces duplicated code, makes new cases easy to add, and still
# reports each case separately so we can identify any failing input.

# =========================================================================== test session starts ===========================================================================
#platform win32 -- Python 3.12.15, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\alexp\OneDrive\Documents\Code_The_Dream\Cloud_AI_26.4\python200-homework\.venv\Scripts\python.exe
#cachedir: .pytest_cache
#rootdir: C:\Users\alexp\OneDrive\Documents\Code_The_Dream\Cloud_AI_26.4\python200-homework
#configfile: pyproject.toml
#collected 6 items                                                                                                                                                          

#warmup_01.py::test_celsius_to_fahrenheit PASSED                                                                                                                      [ 16%]
#warmup_01.py::test_mean_of_empty_raises PASSED                                                                                                                       [ 33%]
#warmup_01.py::test_mean_values[values0-10.0] PASSED                                                                                                                  [ 50%]
#warmup_01.py::test_mean_values[values1-4.0] PASSED                                                                                                                   [ 66%]
#warmup_01.py::test_mean_values[values2--3.0] PASSED                                                                                                                  [ 83%]
#warmup_01.py::test_mean_values[values3-0.0] PASSED                                                                                                                   [100%]

#============================================================================ 6 passed in 0.18s ============================================================================


# Q4
#=========================================================================== test session starts ===========================================================================
#platform win32 -- Python 3.12.15, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\alexp\OneDrive\Documents\Code_The_Dream\Cloud_AI_26.4\python200-homework\.venv\Scripts\python.exe
#cachedir: .pytest_cache
#rootdir: C:\Users\alexp\OneDrive\Documents\Code_The_Dream\Cloud_AI_26.4\python200-homework
#configfile: pyproject.toml
#collected 6 items                                                                                                                                                          

#warmup_01.py::test_celsius_to_fahrenheit FAILED                                                                                                                      [ 16%]
#warmup_01.py::test_mean_of_empty_raises PASSED                                                                                                                       [ 33%]
#warmup_01.py::test_mean_values[values0-10.0] PASSED                                                                                                                  [ 50%]
#warmup_01.py::test_mean_values[values1-4.0] PASSED                                                                                                                   [ 66%]
#warmup_01.py::test_mean_values[values2--3.0] PASSED                                                                                                                  [ 83%]
#warmup_01.py::test_mean_values[values3-0.0] PASSED                                                                                                                   [100%]

#================================================================================ FAILURES =================================================================================
#_______________________________________________________________________ test_celsius_to_fahrenheit ________________________________________________________________________

 #   def test_celsius_to_fahrenheit() -> None:
 #        """Check freezing, boiling, and body-temperature conversions."""
 #     assert celsius_to_fahrenheit(0) == 32
#>       assert celsius_to_fahrenheit(100) == 212
#E       assert 257.0 == 212
#E        +  where 257.0 = celsius_to_fahrenheit(100)

#warmup_01.py:319: AssertionError
#========================================================================= short test summary info =========================================================================
#FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
#======================================================================= 1 failed, 5 passed in 0.32s =======================================================================

# Pytest showed the actual result, 257.0, and the expected result, 212,
# along with the failing assertion and the input 100.
# These details help identify the incorrect conversion factor instead
# of only telling us that an assertion failed.