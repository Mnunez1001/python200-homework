import os
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from supabase import Client, create_client


CITY = "Las Vegas, Nevada"
LATITUDE = 36.17
LONGITUDE = -115.14
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

WEATHER_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


def get_client() -> Client:
    """Load configuration and create a Supabase client."""
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()

    if not url or not key:
        raise ValueError(
            "Set SUPABASE_URL and SUPABASE_KEY in the repository-root .env."
        )

    return create_client(url, key)


# --- Step 1: Extract --------------------------------------------------------------------------------------------------

def extract_weather() -> dict:
    """Fetch the full year of 2023 daily weather for Las Vegas."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": WEATHER_COLUMNS,
        "timezone": "America/Los_Angeles",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "wind_speed_unit": "kmh",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily")
    if not isinstance(daily, dict) or not daily.get("time"):
        raise ValueError("The API response contains no daily observations.")

    print("City:", CITY)
    print("HTTP status:", response.status_code)
    print("Timezone:", payload.get("timezone"))
    print("Daily units:", payload.get("daily_units"))
    print("Daily observations fetched:", len(daily["time"]))

    return daily


# --- Step 2: Transform -------------------------------------------------------------------------------------------------

def transform_weather(daily: dict) -> list[dict]:
    """Converting parallel daily arrays into validated row dictionaries."""
    required = ["time", *WEATHER_COLUMNS]

    for column in required:
        if column not in daily or not isinstance(daily[column], list):
            raise ValueError(f"Missing or invalid daily array: {column}")

    lengths = {column: len(daily[column]) for column in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Daily arrays have mismatched lengths: {lengths}")

    records = [
        {
            "date": daily["time"][i],
            **{
                column: daily[column][i]
                for column in WEATHER_COLUMNS
            },
        }
        for i in range(len(daily["time"]))
    ]

    # Verifying that every requested calendar date occurs exactly once.
    start = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)
    expected_count = (end - start).days + 1
    expected_dates = {
        (start + timedelta(days=i)).isoformat()
        for i in range(expected_count)
    }
    actual_dates = [record["date"] for record in records]

    if (
        len(actual_dates) != expected_count
        or set(actual_dates) != expected_dates
    ):
        raise ValueError(
            f"Expected {expected_count} unique daily dates covering "
            f"{START_DATE} through {END_DATE}; received {len(records)} rows."
        )

    for record in records:
        missing_values = [
            column
            for column in WEATHER_COLUMNS
            if record[column] is None
        ]
        if missing_values:
            raise ValueError(
                f"Missing measurements on {record['date']}: {missing_values}"
            )

    records.sort(key=lambda record: record["date"])

    print("Expected records:", expected_count)
    print("Transformed records:", len(records))
    print("First record:", records[0])
    print("Last record:", records[-1])

    return records


# Transform observation:
# I expected 365 records because 2023 was not a leap year.
# Both runs fetched and transformed exactly 365 records, covering
# 2023-01-01 through 2023-12-31 with no missing or duplicate dates.
# The response confirmed Celsius, millimeters, and km/h, matching
# the units used by my weather classifier.


# --- Step 3: Load -------------------------------------------------------------------------------------------------

def load_weather(supabase: Client, records: list[dict]) -> None:
    """Inserting weather rows or updating existing rows with matching dates."""
    if not records:
        raise ValueError("There are no records to load.")

    response = (
        supabase.table("weather_raw")
        .upsert(
            records,
            on_conflict="date",
            returning="representation",
        )
        .execute()
    )

    print(f"Upserted {len(response.data)} rows into weather_raw.")


# Using date as the conflict key makes repeated loads avoid duplicate
# dates. Existing rows are updated, while missing dates are inserted.
# loaded_at is omitted so we do not explicitly replace it on each retry.
#
# Idempotency observation:
# After the first run, weather_raw contained 366 rows:
# 365 historical observations and one warmup record dated 2026-11-26.
# After the second run, both counts remained unchanged.
#
# Each run reported 365 upserted rows, but that does not mean
# 365 new rows were added. The second run matched existing dates
# and updated those records rather than creating duplicates.
# This demonstrates idempotency for repeated loading of these inputs.


# --- Step 4: Verify -------------------------------------------------------------------------------------------------

def verify_weather(supabase: Client) -> None:
    """Print table counts, date boundaries, and a selected weather row."""
    count_response = (
        supabase.table("weather_raw")
        .select("date", count="exact", head=True)
        .execute()
    )
    print("Total rows in weather_raw:", count_response.count)

    year_response = (
        supabase.table("weather_raw")
        .select("date", count="exact", head=True)
        .gte("date", START_DATE)
        .lte("date", END_DATE)
        .execute()
    )
    print("Rows in the 2023 date range:", year_response.count)

    earliest = (
        supabase.table("weather_raw")
        .select("date")
        .order("date")
        .limit(1)
        .execute()
    )
    latest = (
        supabase.table("weather_raw")
        .select("date")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )

    print(
        "Earliest date in table:",
        earliest.data[0]["date"] if earliest.data else None,
    )
    print(
        "Latest date in table:",
        latest.data[0]["date"] if latest.data else None,
    )

    target = "2023-07-04"
    target_response = (
        supabase.table("weather_raw")
        .select("*")
        .eq("date", target)
        .execute()
    )

    if target_response.data:
        print(f"Record for {target}:")
        print(target_response.data[0])
        return

    # If the target is missing, compare the closest earlier/later rows.
    before = (
        supabase.table("weather_raw")
        .select("*")
        .lte("date", target)
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    after = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", target)
        .order("date")
        .limit(1)
        .execute()
    )

    candidates = before.data + after.data
    if not candidates:
        print("No records are available for the nearest-date check.")
        return

    target_date = date.fromisoformat(target)
    nearest = min(
        candidates,
        key=lambda row: (
            abs((date.fromisoformat(row["date"]) - target_date).days),
            row["date"],
        ),
    )

    print(f"{target} is missing. Nearest available record:")
    print(nearest)

# Verification observation:
# Both runs returned 366 total rows and exactly 365 rows for 2023.
# I also confirmed the total in the Supabase Table Editor.
# The earliest date was 2023-01-01; the latest was 2026-11-26
# because the warmup record remains in the table.
#
# The 2023-07-04 observation was retrieved successfully:
# high 39.9°C, low 27.3°C, precipitation 0.0 mm, and wind 32.4 km/h.
# Its loaded_at timestamp remained unchanged after the second run,
# consistent with omitting that field from the upsert payload.

def main() -> None:
    """Extract, transform, load, and verify Las Vegas weather."""
    supabase = get_client()

    print("\n--- Step 1: Extract ---")
    daily = extract_weather()

    print("\n--- Step 2: Transform ---")
    records = transform_weather(daily)

    print("\n--- Step 3: Load ---")
    load_weather(supabase, records)

    print("\n--- Step 4: Verify ---")
    verify_weather(supabase)


if __name__ == "__main__":
    main()