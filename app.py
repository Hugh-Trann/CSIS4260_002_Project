from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# STREAMLIT APP
# Flight arrival delay prediction dashboard
# Works with the artifacts saved by the training script:
# - outputs/best_model.pkl
# - outputs/model_results.csv
# - nycflights_with_weather.csv
# ============================================================

st.set_page_config(page_title="Flight Delay Prediction Dashboard", layout="wide")

BASE_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = BASE_FOLDER / "outputs"
MODEL_PATH = OUTPUT_FOLDER / "best_model.pkl"
RESULTS_PATH = OUTPUT_FOLDER / "model_results.csv"
METADATA_PATH = OUTPUT_FOLDER / "model_metadata.json"
DATA_PATH = BASE_FOLDER / "nycflights_with_weather.csv"

TARGET_COLUMN = "arr_delay"


# ------------------------------------------------------------
# Reuse feature engineering logic from training
# ------------------------------------------------------------
def add_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "weather_datetime" in df.columns:
        df["weather_datetime"] = pd.to_datetime(df["weather_datetime"], errors="coerce")
    else:
        df["weather_datetime"] = pd.NaT

    required_cols = ["year", "month", "day", "hour", "minute"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for datetime creation: {missing}")

    date_part = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")

    hour_fixed = pd.to_numeric(df["hour"], errors="coerce").fillna(0)
    minute_fixed = pd.to_numeric(df["minute"], errors="coerce").fillna(0)

    mask_24 = hour_fixed == 24
    hour_fixed = hour_fixed.where(~mask_24, 0)
    date_part = date_part.where(~mask_24, date_part + pd.Timedelta(days=1))

    df["dep_datetime"] = (
        date_part
        + pd.to_timedelta(hour_fixed, unit="h")
        + pd.to_timedelta(minute_fixed, unit="m")
    )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dt = df["dep_datetime"]
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["day_of_year"] = dt.dt.dayofyear

    month = dt.dt.month
    df["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["winter", "spring", "summer", "fall"],
        default="unknown",
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["departure_period"] = pd.cut(
        pd.to_numeric(df["hour"], errors="coerce").fillna(0),
        bins=[-1, 5, 11, 17, 21, 24],
        labels=["overnight", "morning", "afternoon", "evening", "late_night"],
    ).astype(str)

    return df


def add_weather_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_gusts_10m",
        "weather_code",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["has_precipitation"] = (df["precipitation"].fillna(0) > 0).astype(int)
    df["has_rain"] = (df["rain"].fillna(0) > 0).astype(int)
    df["has_snow"] = (df["snowfall"].fillna(0) > 0).astype(int)
    df["strong_wind"] = (df["wind_speed_10m"].fillna(0) >= 25).astype(int)
    df["strong_gust"] = (df["wind_gusts_10m"].fillna(0) >= 35).astype(int)
    df["freezing_temp_flag"] = (df["temperature_2m"] <= 0).fillna(False).astype(int)
    df["heavy_precip_flag"] = (df["precipitation"].fillna(0) >= 2.5).astype(int)
    df["heavy_rain_flag"] = (df["rain"].fillna(0) >= 2.5).astype(int)
    df["snow_risk_flag"] = ((df["snowfall"].fillna(0) > 0) | (df["freezing_temp_flag"] == 1)).astype(int)

    severe_codes = {
        51, 53, 55, 56, 57, 61, 63, 65, 66, 67,
        71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99,
    }
    df["severe_weather"] = df["weather_code"].isin(severe_codes).astype(int)

    df["weather_risk_score"] = (
        df["has_precipitation"]
        + df["has_rain"]
        + df["has_snow"]
        + df["strong_wind"]
        + df["strong_gust"]
        + df["freezing_temp_flag"]
        + df["heavy_precip_flag"]
        + df["heavy_rain_flag"]
        + df["snow_risk_flag"]
        + df["severe_weather"]
    )

    return df


def get_feature_columns() -> list:
    return [
        "month", "day", "hour", "minute", "day_of_week", "is_weekend",
        "week_of_year", "quarter", "day_of_year", "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "carrier", "origin", "dest", "route", "carrier_route", "departure_period", "season",
        "distance",
        "temperature_2m", "precipitation", "rain", "snowfall",
        "wind_speed_10m", "wind_gusts_10m", "weather_code",
        "has_precipitation", "has_rain", "has_snow", "strong_wind",
        "strong_gust", "freezing_temp_flag", "heavy_precip_flag",
        "heavy_rain_flag", "snow_risk_flag", "severe_weather", "weather_risk_score",
        "route_hist_mean_delay", "carrier_hist_mean_delay", "carrier_route_hist_mean_delay",
        "origin_hist_mean_delay", "dest_hist_mean_delay", "hour_hist_mean_delay",
        "origin_hour_traffic_count", "route_hour_traffic_count", "carrier_origin_hour_count",
    ]


# ------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("best_model.pkl not found. Run the training script first.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_reference_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("nycflights_with_weather.csv not found. Put it beside app.py")

    df = pd.read_csv(DATA_PATH)
    df = add_datetime_columns(df)
    df = add_time_features(df)
    df = add_weather_risk_features(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column 'arr_delay' not found in reference dataset.")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=["dep_datetime", TARGET_COLUMN, "origin", "dest", "carrier"]).copy()

    df["route"] = df["origin"].astype(str) + "_" + df["dest"].astype(str)
    df["carrier_route"] = df["carrier"].astype(str) + "_" + df["route"].astype(str)
    df["dep_hour_floor"] = df["dep_datetime"].dt.floor("h")

    return df


# ------------------------------------------------------------
# Historical lookup helpers for single prediction
# ------------------------------------------------------------
def safe_mean(series: pd.Series, default_value: float) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if len(cleaned) == 0:
        return float(default_value)
    return float(cleaned.mean())


@st.cache_data
def build_lookup_tables():
    df = load_reference_data().copy()

    global_delay_mean = safe_mean(df[TARGET_COLUMN], 0.0)

    lookup = {
        "global_delay_mean": global_delay_mean,
        "route_hist": df.groupby("route")[TARGET_COLUMN].mean().to_dict(),
        "carrier_hist": df.groupby("carrier")[TARGET_COLUMN].mean().to_dict(),
        "carrier_route_hist": df.groupby("carrier_route")[TARGET_COLUMN].mean().to_dict(),
        "origin_hist": df.groupby("origin")[TARGET_COLUMN].mean().to_dict(),
        "dest_hist": df.groupby("dest")[TARGET_COLUMN].mean().to_dict(),
        "hour_hist": df.groupby("hour")[TARGET_COLUMN].mean().to_dict(),
        "route_hour_traffic": df.groupby(["route", "hour"]).size().to_dict(),
        "origin_hour_traffic": df.groupby(["origin", "hour"]).size().to_dict(),
        "carrier_origin_hour": df.groupby(["carrier", "origin", "hour"]).size().to_dict(),
        "distance_by_route": df.groupby("route")["distance"].median().to_dict() if "distance" in df.columns else {},
        "median_temperature": safe_mean(df.get("temperature_2m", pd.Series(dtype=float)), 0.0),
        "median_precipitation": safe_mean(df.get("precipitation", pd.Series(dtype=float)), 0.0),
        "median_rain": safe_mean(df.get("rain", pd.Series(dtype=float)), 0.0),
        "median_snowfall": safe_mean(df.get("snowfall", pd.Series(dtype=float)), 0.0),
        "median_wind_speed": safe_mean(df.get("wind_speed_10m", pd.Series(dtype=float)), 0.0),
        "median_wind_gusts": safe_mean(df.get("wind_gusts_10m", pd.Series(dtype=float)), 0.0),
        "median_weather_code": safe_mean(df.get("weather_code", pd.Series(dtype=float)), 0.0),
    }

    return lookup


# ------------------------------------------------------------
# Build one prediction row
# ------------------------------------------------------------
def build_input_dataframe(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    carrier: str,
    origin: str,
    dest: str,
    temperature_2m: float,
    precipitation: float,
    rain: float,
    snowfall: float,
    wind_speed_10m: float,
    wind_gusts_10m: float,
    weather_code: int,
    distance: float | None = None,
) -> pd.DataFrame:
    lookup = build_lookup_tables()
    route = f"{origin}_{dest}"
    carrier_route = f"{carrier}_{route}"

    if distance is None:
        distance = lookup["distance_by_route"].get(route, 0.0)

    weather_datetime = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)

    row = pd.DataFrame(
        {
            "year": [year],
            "month": [month],
            "day": [day],
            "hour": [hour],
            "minute": [minute],
            "carrier": [carrier],
            "origin": [origin],
            "dest": [dest],
            "distance": [distance],
            "weather_datetime": [weather_datetime],
            "temperature_2m": [temperature_2m],
            "precipitation": [precipitation],
            "rain": [rain],
            "snowfall": [snowfall],
            "wind_speed_10m": [wind_speed_10m],
            "wind_gusts_10m": [wind_gusts_10m],
            "weather_code": [weather_code],
        }
    )

    row = add_datetime_columns(row)
    row = add_time_features(row)
    row = add_weather_risk_features(row)

    row["route"] = route
    row["carrier_route"] = carrier_route

    global_delay_mean = lookup["global_delay_mean"]
    row["route_hist_mean_delay"] = lookup["route_hist"].get(route, global_delay_mean)
    row["carrier_hist_mean_delay"] = lookup["carrier_hist"].get(carrier, global_delay_mean)
    row["carrier_route_hist_mean_delay"] = lookup["carrier_route_hist"].get(carrier_route, global_delay_mean)
    row["origin_hist_mean_delay"] = lookup["origin_hist"].get(origin, global_delay_mean)
    row["dest_hist_mean_delay"] = lookup["dest_hist"].get(dest, global_delay_mean)
    row["hour_hist_mean_delay"] = lookup["hour_hist"].get(hour, global_delay_mean)

    row["origin_hour_traffic_count"] = lookup["origin_hour_traffic"].get((origin, hour), 1)
    row["route_hour_traffic_count"] = lookup["route_hour_traffic"].get((route, hour), 1)
    row["carrier_origin_hour_count"] = lookup["carrier_origin_hour"].get((carrier, origin, hour), 1)

    feature_columns = get_feature_columns()
    for col in feature_columns:
        if col not in row.columns:
            row[col] = np.nan

    return row[feature_columns].copy()


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
try:
    model = load_model()
    results_df = load_results()
    metadata = load_metadata()
    reference_df = load_reference_data()
    lookup_tables = build_lookup_tables()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

st.title("✈️ Flight Arrival Delay Prediction Dashboard")
st.caption("Predict arrival delay using the trained model and flight + weather inputs.")

with st.sidebar:
    st.header("Model Info")
    if metadata.get("best_model_name"):
        st.success(f"Best model: {metadata['best_model_name']}")
    elif not results_df.empty:
        st.success(f"Best model: {results_df.iloc[0]['model']}")
    else:
        st.info("Model results file not found.")

    st.write("Artifacts expected:")
    st.code("outputs/best_model.pkl\noutputs/model_results.csv\nnycflights_with_weather.csv")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Enter Flight Details")

    available_carriers = sorted(reference_df["carrier"].dropna().astype(str).unique().tolist())
    available_origins = sorted(reference_df["origin"].dropna().astype(str).unique().tolist())
    available_dests = sorted(reference_df["dest"].dropna().astype(str).unique().tolist())

    with st.form("prediction_form"):
        date_col1, date_col2, date_col3 = st.columns(3)
        with date_col1:
            year = st.number_input("Year", min_value=2000, max_value=2100, value=2013, step=1)
        with date_col2:
            month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1)
        with date_col3:
            day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)

        time_col1, time_col2 = st.columns(2)
        with time_col1:
            hour = st.slider("Departure Hour", min_value=0, max_value=23, value=12)
        with time_col2:
            minute = st.slider("Departure Minute", min_value=0, max_value=59, value=0)

        trip_col1, trip_col2, trip_col3 = st.columns(3)
        with trip_col1:
            carrier = st.selectbox("Carrier", available_carriers)
        with trip_col2:
            origin = st.selectbox("Origin", available_origins)
        with trip_col3:
            dest = st.selectbox("Destination", available_dests)

        suggested_route = f"{origin}_{dest}"
        distance = float(lookup_tables["distance_by_route"].get(suggested_route, 0.0))

        st.subheader("Weather Inputs")
        weather_col1, weather_col2, weather_col3 = st.columns(3)

        with weather_col1:
            temperature_2m = st.number_input(
                "Temperature (°C)",
                value=float(lookup_tables["median_temperature"])
            )
            precipitation = st.number_input(
                "Precipitation",
                min_value=0.0,
                value=float(max(0.0, lookup_tables["median_precipitation"]))
            )

        with weather_col2:
            rain = st.number_input(
                "Rain",
                min_value=0.0,
                value=float(max(0.0, lookup_tables["median_rain"]))
            )
            snowfall = st.number_input(
                "Snowfall",
                min_value=0.0,
                value=float(max(0.0, lookup_tables["median_snowfall"]))
            )

        with weather_col3:
            wind_speed_10m = st.number_input(
                "Wind Speed",
                min_value=0.0,
                value=float(max(0.0, lookup_tables["median_wind_speed"]))
            )
            wind_gusts_10m = st.number_input(
                "Wind Gusts",
                min_value=0.0,
                value=float(max(0.0, lookup_tables["median_wind_gusts"]))
            )

        weather_type = st.selectbox(
            "Weather Condition",
            ["Clear", "Cloudy", "Rain", "Snow", "Storm"]
        )

        weather_map = {
            "Clear": 0,
            "Cloudy": 2,
            "Rain": 63,
            "Snow": 73,
            "Storm": 95,
        }

        weather_code = weather_map[weather_type]

        submitted = st.form_submit_button("Predict Delay")

    if submitted:
        try:
            input_df = build_input_dataframe(
                year=int(year),
                month=int(month),
                day=int(day),
                hour=int(hour),
                minute=int(minute),
                carrier=str(carrier),
                origin=str(origin),
                dest=str(dest),
                temperature_2m=float(temperature_2m),
                precipitation=float(precipitation),
                rain=float(rain),
                snowfall=float(snowfall),
                wind_speed_10m=float(wind_speed_10m),
                wind_gusts_10m=float(wind_gusts_10m),
                weather_code=int(weather_code),
                distance=float(distance),
            )

            prediction = float(model.predict(input_df)[0])

            st.subheader("Prediction Result")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Predicted Arrival Delay", f"{prediction:.2f} min")
            with metric_col2:
                if prediction <= 15:
                    st.success("Status: On time / minor delay")
                elif prediction <= 60:
                    st.warning("Status: Moderate delay")
                else:
                    st.error("Status: Severe delay")

            with st.expander("Show model input row"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with col2:
    st.subheader("Model Performance")
    if not results_df.empty:
        st.dataframe(results_df, use_container_width=True)
    else:
        st.info("model_results.csv not found yet.")

    st.subheader("Quick Notes")
    st.markdown(
        """
        - This app loads the trained `best_model.pkl`.
        - Historical route/carrier features are estimated from the training dataset.
        - For unseen routes, the app falls back to global average delay values.
        - For best results, keep `app.py`, `nycflights_with_weather.csv`, and the `outputs` folder together.
        """
    )

    if RESULTS_PATH.exists():
        st.download_button(
            label="Download model results CSV",
            data=RESULTS_PATH.read_bytes(),
            file_name="model_results.csv",
            mime="text/csv",
        )