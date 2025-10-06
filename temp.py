import duckdb
import pandas as pd
import awswrangler as wr
import streamlit as st
import os
import pydeck as pdk
from datetime import timedelta

# --- 1️⃣ Connect to DuckDB (in-memory or persistent DB)
con = duckdb.connect()

# --- 2️⃣ Load and configure S3 settings
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

source_prefix = "s3://sumant28-lastly/output/dt=*/*.parquet"
json_files = wr.s3.list_objects(source_prefix, suffix=".parquet")
uri = json_files[len(json_files) - 1]
# --- 3️⃣ Query the Parquet file directly from S3
query1 = f"""
SELECT 
a.altitude,
a.course,
a.courseAccuracy,
a.latitude,
a.longitude,
a.speed,
a.speedAccuracy,
a.timestamp,
a.verticalAccuracy
FROM (SELECT * FROM '{uri}') w
CROSS JOIN UNNEST (w.route) AS t(a);
"""

query2 = f"""
SELECT 
a.Avg,
a.Max,
a.Min,
a.date,
a.source,
a.units
FROM (SELECT * FROM '{uri}') w
CROSS JOIN UNNEST (w.heartRateData) AS t(a);
"""

query3 = f"""
SELECT 
a.Avg,
a.Max,
a.Min,
a.date,
a.source,
a.units
FROM (SELECT * FROM '{uri}') w
CROSS JOIN UNNEST (w.heartRateRecovery) AS t(a);
"""

query4 = f"""
SELECT * FROM '{uri}'
"""

df1 = con.execute(query1).fetchdf()
df2 = con.execute(query2).fetchdf()
df3 = con.execute(query3).fetchdf()
df4 = con.execute(query4).fetchdf()

delta1 = df4['end'][0] - df4['start'][0]

from datetime import timedelta

td = timedelta(hours=1, minutes=23, seconds=45)

td_str = f"{int(delta1.total_seconds() // 3600):02}:{int((delta1.total_seconds() % 3600) // 60):02}"
print(td_str)  # "01:23"
#td_str = f"{int(td.total_seconds() // 3600):02}:{int((td.total_seconds() % 3600) // 60):02}"

s2 = df4['start'][0]
ts_str = s2.strftime("%H:%M")

# --- Simulated GPS + Time Data ---
df = pd.DataFrame({
    "timestamp": pd.date_range("2025-10-06 09:00:00", periods=10, freq="min"),
    "latitude": [-36.8485, -36.8490, -36.8495, -36.8500, -36.8510,
                 -36.8515, -36.8520, -36.8525, -36.8530, -36.8535],
    "longitude": [174.7633, 174.7640, 174.7645, 174.7650, 174.7660,
                  174.7665, 174.7670, 174.7675, 174.7680, 174.7685],
    "speed_kmh": [10, 12, 14, 15, 14, 13, 12, 11, 10, 9],
})

# Format timestamps to show only time (HH:MM)
df["time"] = df["timestamp"].dt.strftime("%H:%M")

# --- Compute metrics ---
avg_speed = df["speed_kmh"].mean()
max_speed = df["speed_kmh"].max()
distance_km = 1.2  # example placeholder

# --- Define map ---
midpoint = (df1["latitude"].mean(), df1["longitude"].mean())

line_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": df1[["longitude", "latitude"]].values.tolist()}],
    get_color=[255, 0, 0],
    width_scale=10,
    width_min_pixels=2,
)

point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[longitude, latitude]',
    get_color='[0, 0, 255, 160]',
    get_radius=20,
)

view_state = pdk.ViewState(
    latitude=midpoint[0],
    longitude=midpoint[1],
    zoom=14,
    pitch=0,
)

deck = pdk.Deck(
    map_provider="carto",
    map_style="light",
    layers=[line_layer, point_layer],
    initial_view_state=view_state,
)

# --- Page Layout: Three Columns ---
col_map, col_chart, col_metrics = st.columns([2, 2, 3])

# 1️⃣ MAP PANEL
with col_map:
    st.subheader("Route Map")
    st.pydeck_chart(deck)

# 2️⃣ TIME SERIES PANEL
with col_chart:
    st.subheader("Average Heart Rate")
    st.line_chart(df2, x="date", y="Avg", height=300)

# 3️⃣ METRICS PANEL
with col_metrics:
    st.subheader("Metrics")
    st.metric(label="Day", value=df4['start'][0].strftime("%Y-%m-%d"))
    st.metric(label="Time", value=ts_str)
    st.metric(label="Workout Type", value=df4['name'][0])
    st.metric(label="Workout Time", value=td_str)
    st.metric(label="Burned kJ", value=round(df4['activeEnergyBurned'][0]['qty']))