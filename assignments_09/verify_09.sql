-- Query 1: Row count and date range
SELECT
    COUNT(*) AS total_rows,
    MIN(date) AS earliest_date,
    MAX(date) AS latest_date
FROM public.weather_raw;

-- total_rows | earliest_date | latest_date
-- 366        | 2023-01-01    | 2026-11-26


-- Query 2: Rows with at least one missing weather measurement
SELECT
    COUNT(*) AS rows_with_missing_values
FROM public.weather_raw
WHERE temperature_2m_max IS NULL
   OR temperature_2m_min IS NULL
   OR precipitation_sum IS NULL
   OR wind_speed_10m_max IS NULL;

-- rows_with_missing_values
-- 0


-- Reflection:
--
-- The SQL row count and date boundaries match the Python results
-- from Step 4: 366 rows, earliest date 2023-01-01, and latest date
-- 2026-11-26.
--
-- A full year of daily observations for 2023 contains 365 rows.
-- The additional row is the synthetic warmup record for 2026-11-26.
-- That record also explains why the latest date is not 2023-12-31.
--
-- The warmup inserted that practice record and then upserted it.
-- The project upserted 365 historical observations on each run.
-- Because date is the primary key and conflict key, repeating the
-- upserts did not create additional rows for existing dates.
--
-- Query 2 checks something the Step 4 output did not explicitly
-- count: rows with missing weather measurements. The expected count
-- is zero because the historical records passed validation and the
-- warmup record supplied all four measurements.


-- Video Link: https://youtu.be/N-mslrfXRWE