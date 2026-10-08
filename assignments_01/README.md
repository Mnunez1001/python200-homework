# Assignment 1: Python Foundations and Weatherkit

This assignment practices classes, dataclasses, type hints, docstrings,
Pydantic validation, pytest, and organizing Python code into a package.

## Part 1: Warmup

`warmup_01.py` contains:

- Thermometer and temperature alert classes
- Weather station dataclasses
- Pydantic models and validation examples
- Tests covering conversions, error cases, and parametrized inputs

## Part 2: Weatherkit

The `weatherkit` package converts a raw Open-Meteo weather response
into daily summaries.

The pipeline:

1. Loads `weather_raw.json`
2. Validates the response and checks hourly array lengths
3. Converts parallel arrays into hourly reading objects
4. Groups readings by date
5. Reports days that meet the minimum observation threshold
6. Identifies incomplete days that were excluded

The supplied dataset contains 168 hourly observations across seven days
for Charlotte, North Carolina. Its timestamps use GMT.

## Files

- `weatherkit/schemas.py`: Pydantic boundary validation
- `weatherkit/records.py`: Hourly dataclass and conversion
- `weatherkit/summarize.py`: Daily summaries and aggregation
- `weatherkit/__init__.py`: Package exports
- `tests/`: Project test suite
- `conftest.py`: Assignment-level pytest configuration file
- `report.py`: Daily report script and reflection
- `weather_raw.json`: Supplied input data
- `warmup_01.py`: Warmup exercises and tests

## Setup and execution

Requires Python 3.12 and uv.

From the repository root:

```bash
uv sync
uv run python assignments_01/report.py
uv run pytest assignments_01/warmup_01.py -v
```

Run the project tests from the assignment folder:

```bash
cd assignments_01
uv run pytest -v
```

## Verification

The report produces seven daily summaries for the supplied dataset.
The project suite contains 16 test cases covering validation,
conversion, equality, aggregation, and incomplete-day handling.

A deliberate change from subtraction to addition in `temp_range()`
was caught by `test_temperature_range`. The correct implementation
was restored.