import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
import time
import duckdb
import awswrangler as wr

st.set_page_config(layout="wide")

# --- Connect to DuckDB and load data from S3 ---
@st.cache_resource
def init_duckdb():
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    
    # Get AWS credentials from Streamlit secrets
    try:
        aws_access_key = st.secrets["aws_access_key_id"]
        aws_secret_key = st.secrets["aws_secret_access_key"]
        aws_region = st.secrets.get("aws_region", "ap-southeast-2")
    except KeyError:
        st.error("AWS credentials not found in Streamlit secrets. Please add them to .streamlit/secrets.toml")
        st.stop()
    
    con.execute(f"""
    SET s3_region='{aws_region}';
    SET s3_access_key_id='{aws_access_key}';
    SET s3_secret_access_key='{aws_secret_key}';
    """)
    return con

con = init_duckdb()

# Load data from S3
@st.cache_data
def load_data():
    source_prefix = "s3://sumant28-testbucket/output/dt=*/*.parquet"
    query = f"SELECT * FROM '{source_prefix}'"
    df_raw = con.execute(query).fetchdf()
    return df_raw

df_raw = load_data()

# --- Process timezone and dates ---
local_tz = "Pacific/Auckland"
for col in df_raw:
    if col == 'dt':
        df_raw['dt'] = df_raw['dt'].dt.date
    elif df_raw[col].dtype == 'datetime64[us]':
        df_raw[col] = pd.to_datetime(df_raw[col].dt.tz_localize("UTC").dt.tz_convert(local_tz))

# --- Extract metrics by activity type ---
stand_data = (
    df_raw.groupby(["dt", "name"], as_index=False)
    .agg(total=("qty", "sum"), has_stand=("name", lambda x: x.str.contains("stand_hour").any()))
    .query("has_stand")[["dt", "total"]]
)

move_data = (
    df_raw.groupby(["dt", "name"], as_index=False)
    .agg(total=("qty", "sum"), has_move=("name", lambda x: x.str.contains("active_ener").any()))
    .query("has_move")[["dt", "total"]]
)

exercise_data = (
    df_raw.groupby(["dt", "name"], as_index=False)
    .agg(total=("qty", "sum"), has_exercise=("name", lambda x: x.str.contains("exer").any()))
    .query("has_exercise")[["dt", "total"]]
)

# --- Normalize values to percentages ---
stand_data["stand"] = round(stand_data["total"] / 12 * 100)
move_data["move"] = round(move_data["total"] / (550 * 4) * 100)
exercise_data["exercise"] = round(exercise_data["total"] / 30 * 100)

# --- Merge data into single DataFrame ---
stand_data = stand_data.rename(columns={"dt": "date"}).reset_index(drop=True)
move_data = move_data.rename(columns={"dt": "date"}).reset_index(drop=True)
exercise_data = exercise_data.rename(columns={"dt": "date"}).reset_index(drop=True)

data = stand_data[["date", "stand"]].merge(
    move_data[["date", "move"]], on="date", how="outer"
).merge(
    exercise_data[["date", "exercise"]], on="date", how="outer"
).fillna(0)

# Convert date to datetime for proper comparison
data["date"] = pd.to_datetime(data["date"])

# --- Get date range from actual data ---
month_days = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq="D")
YEAR = data['date'].min().year
MONTH = data['date'].min().month

# --- Function to draw one day's rings ---
def make_rings(move, exercise, stand, day, progress):
    bg = "#F0F2F5"
    colors = ["#00C7E0", "#34C759", "#FF453A"]  # Stand (inner), Exercise, Move (outer)
    labels = ["Stand", "Exercise", "Move"]
    vals = [stand, exercise, move]
    holes = [0.15, 0.5, 0.75]

    fig = go.Figure()

    is_empty = move == 0 and exercise == 0 and stand == 0
    
    for v, c, h, label in zip(vals, colors, holes, labels):
        if is_empty:
            fig.add_trace(go.Pie(
                values=[0, 100],
                hole=h, sort=False, rotation=90,
                marker=dict(colors=[bg, bg]),
                textinfo='none', showlegend=False,
                hoverinfo='skip',
                name=label
            ))
        else:
            current_val = v * progress / 100
            fig.add_trace(go.Pie(
                values=[current_val, max(100 - current_val, 0)],
                hole=h, sort=False, rotation=90,
                marker=dict(colors=[c, bg]),
                textinfo='none', showlegend=False,
                hovertemplate=f"<b>{label}</b><br>Progress: {v:.0f}%<extra></extra>",
                name=label
            ))

    fig.update_layout(
        width=100, height=100, margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(text=str(day), x=0.5, y=0.5,
                          font=dict(size=16, color="black"), showarrow=False)],
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    fig.update_traces(marker_line=dict(width=0))
    return fig

# --- Navigation ---
page = st.sidebar.radio("Navigation", ["Activity Rings", "Hourly Heatmap"], key="page_nav")

if page == "Activity Rings":
    st.markdown(f"### {datetime.date(YEAR, MONTH, 1).strftime('%B %Y')} Apple Fitness Monthly Challenge")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Daily Activity Ring Close/Monthly Target", f"{(data[['stand','move','exercise']] >= 100).all(axis=1).sum()}/10")
    with metric_cols[1]:
        st.metric("Daily Move Target/Monthly Average", f"450/{round(sum(move_data['total'])*0.25/len(move_data),0)}")
    with metric_cols[2]:
        st.metric("Daily Exercise Target/Monthly Average", f"30/{round(sum(exercise_data['total'])/len(exercise_data),0)}")
    with metric_cols[3]:
        st.metric("Daily Stand Target/Monthly Average", f"12/{round(sum(stand_data['total'])/len(stand_data),1)}")

    st.divider()

    st.markdown("### Daily Progress Rings")

    first_day = datetime.date(YEAR, MONTH, 1)
    first_weekday = first_day.weekday()

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    header_cols = st.columns(7)
    for i, col in enumerate(header_cols):
        with col:
            st.markdown(f"<div style='text-align: center; font-weight: bold;'>{day_names[i]}</div>", unsafe_allow_html=True)

    placeholders = []
    cols_per_row = 7
    current_col = 0
    current_row_cols = st.columns(cols_per_row)

    for _ in range(first_weekday):
        placeholders.append(None)
        current_col += 1

    for day in range(1, len(month_days) + 1):
        if current_col >= cols_per_row:
            current_row_cols = st.columns(cols_per_row)
            current_col = 0
            
        placeholders.append(current_row_cols[current_col].empty())
        current_col += 1

    for idx, date in enumerate(month_days):
        placeholder_idx = idx + first_weekday
        day_data_match = data[data['date'] == date]
            
        if day_data_match.empty:
            continue
            
        day_data = day_data_match.iloc[0]
        fig = make_rings(day_data['move'], day_data['exercise'], day_data['stand'], date.day, 100)
        placeholders[placeholder_idx].plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")

elif page == "Hourly Heatmap":
    st.title("Heat Map of Hourly Daily Stand Count")
    
    # Prepare hourly data
    df_hourly = df_raw.copy()
    
    result = (
        df_hourly.groupby(["dt", pd.Grouper(key="date", freq="H"), "name"], as_index=False)
        .agg(total=("qty", "sum"), has_r=("name", lambda x: x.str.contains("stand_hour").any()))
        .query("has_r")
    )

    result['hours'] = result["date"].dt.strftime("%Y-%m-%d %H:00")
    result = result[['hours','total']]
    result = result.rename(columns={"hours": "day_hour", "total": "value"})

    hourly_data = result

    # Convert day_hour to datetime
    hourly_data["day_hour"] = pd.to_datetime(hourly_data["day_hour"])
    hourly_data["date"] = hourly_data["day_hour"].dt.date
    hourly_data["hour"] = hourly_data["day_hour"].dt.hour

    # Create a pivot table: rows=dates, columns=hours (0-23)
    pivot_data = hourly_data.pivot_table(
        index="date",
        columns="hour",
        values="value",
        aggfunc="max",
        fill_value=0
    )

    # Ensure all hours 0-23 are represented
    for hour in range(24):
        if hour not in pivot_data.columns:
            pivot_data[hour] = 0

    pivot_data = pivot_data.sort_index(axis=1)

    # Add pass/fail indicator: 1 if sum >= 12, else 0
    pivot_data["pass_fail"] = (pivot_data.iloc[:, :-1].sum(axis=1) >= 12).astype(int)
    hour_counts = pivot_data.iloc[:, :-1].sum(axis=1)

    # Create heatmap with pass/fail column
    heatmap_data = pivot_data.drop("pass_fail", axis=1)

    # Add status labels to each row
    status_labels = ["✓" if x == 1 else "✗" for x in pivot_data["pass_fail"].values]
    y_labels = [f"{str(d)} {label}" for d, label in zip(heatmap_data.index, status_labels)]

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"{h:02d}:00" for h in heatmap_data.columns],
        y=y_labels,
        colorscale=[[0, "#F0F2F5"], [1, "#00C7E0"]],
        showscale=True,
        colorbar=dict(title="Active"),
        hovertemplate="Date: %{y}<br>Hour: %{x}<br>Value: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title="Hourly Activity by Day",
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        height=max(400, len(heatmap_data) * 30),
        width=1000,
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, use_container_width=True)
