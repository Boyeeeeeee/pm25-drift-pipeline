import duckdb

con = duckdb.connect("data/pm25.duckdb")
for stmt in open("sql/views.sql", encoding="utf-8").read().split(";"):
    if stmt.strip():
        con.execute(stmt)

print("Coverage check (NULL coverage, <75% coverage):")
print(con.sql("""SELECT COUNT(*) FILTER (WHERE pct_complete IS NULL) AS null_cov,
                        COUNT(*) FILTER (WHERE pct_complete < 75)   AS low_cov
                 FROM raw_openaq_daily"""))

print("Persistence baseline (tomorrow = today), per station:")
print(con.sql("""SELECT location_name, COUNT(*) AS n,
                        ROUND(AVG(ABS(target_next_day - pm25_today)), 2) AS mae,
                        ROUND(100 * SUM(ABS(target_next_day - pm25_today)) / SUM(target_next_day), 1) AS wape_pct
                 FROM features GROUP BY 1 ORDER BY 1"""))

print("Persistence baseline by month, all stations:")
print(con.sql("""SELECT strftime(day, '%Y-%m') AS month, COUNT(*) AS n,
                        ROUND(AVG(ABS(target_next_day - pm25_today)), 2) AS mae
                 FROM features GROUP BY 1 ORDER BY 1"""))