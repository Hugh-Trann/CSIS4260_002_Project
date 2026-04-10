import pandas as pd
import requests
from time import sleep
from pathlib import Path


# ------------------------------------------------------------
# Step 1: read the flights data
# ------------------------------------------------------------
def load_flights(csv_file: str) -> pd.DataFrame:
    flights = pd.read_csv(csv_file)

    numeric_cols = [
        "year", "month", "day",
        "dep_time", "dep_delay",
        "arr_time", "arr_delay",
        "flight", "air_time", "distance",
        "hour", "minute",
    ]

    for col in numeric_cols:
        if col in flights.columns:
            flights[col] = pd.to_numeric(flights[col], errors="coerce")

    string_cols = ["carrier", "tailnum", "origin", "dest"]
    for col in string_cols:
        if col in flights.columns:
            flights[col] = flights[col].astype(str).str.strip()

    # keep rows that have the basic date/time information needed for joining
    flights = flights.dropna(subset=["year", "month", "day", "hour", "minute", "origin"]).copy()

    flights["year"] = flights["year"].astype(int)
    flights["month"] = flights["month"].astype(int)
    flights["day"] = flights["day"].astype(int)
    flights["hour"] = flights["hour"].astype(int)
    flights["minute"] = flights["minute"].astype(int)

    flights["flight_date"] = pd.to_datetime(flights[["year", "month", "day"]], errors="coerce")
    flights = flights.dropna(subset=["flight_date"]).copy()

    # create a proper scheduled departure datetime
    flights["dep_hour_fixed"] = flights["hour"]
    flights["dep_minute_fixed"] = flights["minute"]
    flights["dep_datetime"] = flights["flight_date"]

    # sometimes hour can be 24 in schedule data, so move it to the next day
    mask_24 = flights["dep_hour_fixed"] == 24
    flights.loc[mask_24, "dep_datetime"] = flights.loc[mask_24, "dep_datetime"] + pd.Timedelta(days=1)
    flights.loc[mask_24, "dep_hour_fixed"] = 0

    flights["dep_datetime"] = (
        flights["dep_datetime"]
        + pd.to_timedelta(flights["dep_hour_fixed"], unit="h")
        + pd.to_timedelta(flights["dep_minute_fixed"], unit="m")
    )

    return flights


# ------------------------------------------------------------
# Step 2: download historical weather from Open-Meteo
# ------------------------------------------------------------
def get_weather_for_airport(
    airport_code: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        # visibility removed because it was not reliably returned in the current merged dataset
        "hourly": "temperature_2m,precipitation,rain,snowfall,wind_speed_10m,wind_gusts_10m,weather_code",
        "timezone": "America/New_York",
        "models": "era5",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"No hourly weather data returned for {airport_code}")

    hourly = data["hourly"]

    weather_df = pd.DataFrame(
        {
            "weather_datetime": pd.to_datetime(hourly["time"]),
            "temperature_2m": hourly["temperature_2m"],
            "precipitation": hourly["precipitation"],
            "rain": hourly["rain"],
            "snowfall": hourly["snowfall"],
            "wind_speed_10m": hourly["wind_speed_10m"],
            "wind_gusts_10m": hourly["wind_gusts_10m"],
            "weather_code": hourly["weather_code"],
        }
    )

    weather_df["origin"] = airport_code
    return weather_df


# ------------------------------------------------------------
# Step 3: merge flights and weather together
# ------------------------------------------------------------
def build_merged_dataset(
    input_csv: str = "nycflights.csv",
    output_csv: str = "nycflights_with_weather.csv",
) -> pd.DataFrame:
    airports = {
        "JFK": {"latitude": 40.6413, "longitude": -73.7781},
        "LGA": {"latitude": 40.7769, "longitude": -73.8740},
        "EWR": {"latitude": 40.6895, "longitude": -74.1745},
    }

    flights = load_flights(input_csv)
    flights = flights[flights["origin"].isin(airports.keys())].copy()

    start_date = flights["flight_date"].min().strftime("%Y-%m-%d")
    end_date = flights["flight_date"].max().strftime("%Y-%m-%d")

    print("Flight date range:", start_date, "to", end_date)

    all_weather = []
    for code, coords in airports.items():
        print(f"Downloading weather for {code}...")
        airport_weather = get_weather_for_airport(
            airport_code=code,
            latitude=coords["latitude"],
            longitude=coords["longitude"],
            start_date=start_date,
            end_date=end_date,
        )
        all_weather.append(airport_weather)
        sleep(1)

    weather = pd.concat(all_weather, ignore_index=True)

    # round flight time down to the hour so it can match hourly weather data
    flights["weather_datetime"] = flights["dep_datetime"].dt.floor("h")

    flights_with_weather = pd.merge(
        flights,
        weather,
        on=["origin", "weather_datetime"],
        how="left",
    )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    flights_with_weather.to_csv(output_csv, index=False)

    print("Done!")
    print("Merged file saved as:", output_csv)
    print("Final shape:", flights_with_weather.shape)
    print("Missing weather rows:", flights_with_weather["temperature_2m"].isna().sum())

    return flights_with_weather


if __name__ == "__main__":
    build_merged_dataset()
