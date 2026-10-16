import duckdb
import pandas as pd


def load_features(db="data/pm25.duckdb"):
    con = duckdb.connect(db)
    for stmt in open("sql/views.sql", encoding="utf-8").read().split(";"):
        if stmt.strip():
            con.execute(stmt)
    df = con.sql("SELECT * FROM features").df()
    con.close()
    df["day"] = pd.to_datetime(df["day"])
    df["month"] = df["day"].dt.to_period("M")
    df["station"] = df["location_name"].astype("category").cat.codes
    return df