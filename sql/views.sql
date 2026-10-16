-- Keep only days with at least 75% hourly coverage
CREATE OR REPLACE VIEW clean_daily AS
SELECT location_name, day, value AS pm25
FROM raw_openaq_daily
WHERE pct_complete >= 75;

-- One row per station-day: today's readings and weather, plus the next-day target.
-- Joining on day + 1 (not LEAD) so a gap in the data never pairs two non-consecutive days.
CREATE OR REPLACE VIEW features AS
SELECT t.location_name, t.day,
       t.pm25            AS pm25_today,
       y.pm25            AS pm25_yesterday,
       n.pm25            AS target_next_day,
       w.temp_mean, w.precip_sum, w.wind_max, w.wind_dir
FROM clean_daily t
JOIN clean_daily n ON n.location_name = t.location_name AND n.day = t.day + 1
LEFT JOIN clean_daily y ON y.location_name = t.location_name AND y.day = t.day - 1
LEFT JOIN raw_weather_daily w ON w.day = t.day;