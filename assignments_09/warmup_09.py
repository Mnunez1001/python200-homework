import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from postgrest.exceptions import APIError
from supabase import Client, create_client


# --- Supabase Connection --------------------------------------------------------------------------------------

# Q1
# supabase-py needs the project's HTTPS API URL and an API key.
# I can find them in the project's Connect dialog; API keys are also
# available under Settings > API Keys, and the URL under Settings > API.
#
# Environment variables keep configuration separate from code and make
# it easier to switch projects without editing the script.
# Secret/service-role keys must never be committed because they provide
# elevated access. Publishable keys are intended for public applications,
# but permissions and RLS must still protect the underlying data.


# Q2
def get_client() -> Client:
    """Load project configuration and return a Supabase client.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY is missing or empty.
    """
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()

    missing = []
    if not url:
        missing.append("SUPABASE_URL")
    if not key:
        missing.append("SUPABASE_KEY")

    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Add them to the .env file at the repository root."
        )

    return create_client(url, key)


# Q3
# Row Level Security uses database policies to control which rows a
# user can read, insert, update, or delete, alongside table permissions.
# We disabled RLS for this course exercise so our Python scripts can
# practice reading and writing without first configuring user policies.
# Disabling RLS does not remove primary keys or other data constraints.
#
# In a real application with multiple users, I would keep it enabled.
# For example, a personal fitness app should let users access their
# own workout records without exposing other users' records.


# --- supabase-py CRUD ----------------------------------------------------------------------------------------------

# Q1
def insert_test_record(supabase: Client) -> list[dict]:
    """Insert today's synthetic weather observation and return the row."""
    record = {
        "date": date.today().isoformat(),
        "temperature_2m_max": 18.5,
        "temperature_2m_min": 7.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12.0,
    }

    response = supabase.table("weather_raw").insert(record).execute()
    return response.data


# Running insert_test_record twice on the same date would raise a
# duplicate-primary-key error because weather_raw.date must be unique.
# It would not create two rows for that date.
#
# To allow retries that update existing weather values, replace:
#     .insert(record)
# with:
#     .upsert(record, on_conflict="date")
#
# These are invented practice measurements,
# loaded_at is omitted so the database supplies its default timestamp.


# Q2
def get_records_by_date_range(supabase: Client, start: str, end: str,) -> list[dict]:
    """Return all weather rows within an inclusive ISO date range.

    Args:
        supabase: Connected Supabase client.
        start: First date to include, formatted as YYYY-MM-DD.
        end: Last date to include, formatted as YYYY-MM-DD.

    Returns:
        Row dictionaries ordered by date.

    Raises:
        ValueError: If dates are invalid or start is after end.
    """
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    if start_date > end_date:
        raise ValueError("start must be on or before end.")

    records = []
    offset = 0
    page_size = 100

    # Fetching pages so the function is not limited to one response page.
    while True:
        response = (
            supabase.table("weather_raw")
            .select("*")
            .gte("date", start_date.isoformat())
            .lte("date", end_date.isoformat())
            .order("date")
            .range(offset, offset + page_size - 1)
            .execute()
        )

        page = response.data
        if not page:
            break

        records.extend(page)
        offset += len(page)

    return records


# Q3
# insert adds new rows. I would choose it when an existing key should
# be treated as an error, such as accidentally recording the same
# uniquely identified transaction twice.
#
# upsert inserts new rows and updates matching rows when the specified
# unique conflict key already exists. I would choose it when repeatedly
# downloading weather observations that may contain corrected values.
#
# Upsert is not automatically better for every task: it can overwrite
# existing values when I actually wanted duplicate input to be rejected.
# It can also execute an update even when the supplied values are
# unchanged.


def safe_upsert(supabase: Client, records: list[dict]) -> None:
    """Upsert weather records using date as the conflict key."""
    if not records:
        print("Rows affected: 0")
        return

    response = (
        supabase.table("weather_raw")
        .upsert(
            records,
            on_conflict="date",
            returning="representation",
        )
        .execute()
    )

    print("Rows affected:", len(response.data))


# --- Idempotency ----------------------------------------------------------------------------------------------

# Q1
# Idempotency matters because a pipeline may need to retry after a
# network failure, crash, or scheduled rerun. Repeating the same input
# should leave the intended stored data in the same state.
#
# For example, a pipeline might insert 50 of 100 weather observations
# before crashing. If it restarts with plain inserts, a table without
# a unique constraint could accumulate duplicate observations, causing
# incorrect counts and totals.
# In my table, the primary key prevents duplicates, but the retry
# could fail immediately on a date inserted before the crash.
#
# Upserting by date lets the retry update those existing dates and
# insert the remaining ones without creating duplicate observations.
# This assumes the input values are stable; changing timestamps,
# incrementing counters, or other side effects need separate care.


def main() -> None:
    """Demonstrate connecting, inserting, selecting, and upserting."""
    supabase = get_client()
    today = date.today()

    print("--- Connection Q2: Client Created ---")
    print("Supabase client initialized; the next request checks access.")

    print("\n--- CRUD Q1: Insert Test Record ---")
    try:
        inserted = insert_test_record(supabase)
        print("Inserted:", inserted)
    except APIError as error:
        # Handle the expected duplicate-date error on subsequent runs.
        # Other database errors should still be raised.
        if error.code != "23505":
            raise
        print("A row already exists for today; the insert was rejected.")

    print("\n--- CRUD Q2: Inclusive Date Range ---")
    start = (today - timedelta(days=1)).isoformat()
    end = (today + timedelta(days=1)).isoformat()

    rows = get_records_by_date_range(supabase, start, end)
    print(f"Records from {start} through {end}:")
    for row in rows:
        print(row)

    print("\n--- CRUD Q3: Repeatable Upsert ---")
    practice_records = [
        {
            "date": today.isoformat(),
            "temperature_2m_max": 18.5,
            "temperature_2m_min": 7.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 12.0,
        }
    ]

    safe_upsert(supabase, practice_records)
    safe_upsert(supabase, practice_records)

    stored = get_records_by_date_range(
        supabase,
        today.isoformat(),
        today.isoformat(),
    )
    print("Rows stored for today after both upserts:", len(stored))


if __name__ == "__main__":
    main()

# Connection observation:
# My first insert reached Supabase but failed because the anon role
# did not have the required table privileges.
# Disabling RLS did not automatically grant those privileges.
# I granted SELECT, INSERT, and UPDATE on weather_raw so the course
# script could read, insert, and upsert its practice records.

# Warmup execution observations:
# After granting SELECT, INSERT, and UPDATE to anon, the script
# successfully inserted the practice record for 2026-11-26.
# Supabase automatically supplied loaded_at through DEFAULT now().
#
# The inclusive date-range query returned the inserted record because
# its date fell between 2026-11-25 and 2026-11-27.
#
# Both upserts reported one affected row, but only one row remained
# stored for that date. "Rows affected" does not mean "new rows added":
# each upsert matched the existing date and updated that record.
# This demonstrates repeatable loading without duplicate dates.