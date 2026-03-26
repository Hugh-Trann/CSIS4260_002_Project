import pandas as pd
import requests
from time import sleep


flights = pd.read_csv("nycflights.csv")

print("Original shape:", flights.shape)
print("\nOriginal dtypes:")
print(flights.dtypes)

numeric_cols = [
    "year", "month", "day",
    "dep_time", "dep_delay",
    "arr_time", "arr_delay",
    "flight", "air_time", "distance",
    "hour", "minute"
]

for col in numeric_cols:
    flights[col] = pd.to_numeric(flights[col], errors="coerce")

string_cols = ["carrier", "tailnum", "origin", "dest"]

for col in string_cols:
    flights[col] = flights[col].astype(str).str.strip()

flights = flights.dropna(subset=["year", "month", "day", "hour", "minute", "origin"]).copy()

flights["year"] = flights["year"].astype(int)
flights["month"] = flights["month"].astype(int)
flights["day"] = flights["day"].astype(int)
flights["hour"] = flights["hour"].astype(int)
flights["minute"] = flights["minute"].astype(int)

flights["flight_date"] = pd.to_datetime(flights[["year", "month", "day"]], errors="coerce")

flights = flights.dropna(subset=["flight_date"]).copy()


flights["dep_hour_fixed"] = flights["hour"]
flights["dep_minute_fixed"] = flights["minute"]
flights["dep_datetime"] = flights["flight_date"]

# If hour = 24, move date to next day and set hour to 0
mask_24 = flights["dep_hour_fixed"] == 24
flights.loc[mask_24, "dep_datetime"] = flights.loc[mask_24, "dep_datetime"] + pd.Timedelta(days=1)
flights.loc[mask_24, "dep_hour_fixed"] = 0

# Add hour and minute offsets
flights["dep_datetime"] = (
    flights["dep_datetime"]
    + pd.to_timedelta(flights["dep_hour_fixed"], unit="h")
    + pd.to_timedelta(flights["dep_minute_fixed"], unit="m")
)

airports = {
    "JFK": {"latitude": 40.6413, "longitude": -73.7781},
    "LGA": {"latitude": 40.7769, "longitude": -73.8740},
    "EWR": {"latitude": 40.6895, "longitude": -74.1745},
}

flights = flights[flights["origin"].isin(airports.keys())].copy()


# download hourly weather data
def get_weather_for_airport(airport_code, latitude, longitude, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,rain,snowfall,wind_speed_10m,wind_gusts_10m,visibility,weather_code",
        "timezone": "America/New_York",
        "models": "era5"
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"No hourly weather data returned for {airport_code}")

    hourly = data["hourly"]

    weather_df = pd.DataFrame({
        "weather_datetime": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "precipitation": hourly["precipitation"],
        "rain": hourly["rain"],
        "snowfall": hourly["snowfall"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "wind_gusts_10m": hourly["wind_gusts_10m"],
        "visibility": hourly["visibility"],
        "weather_code": hourly["weather_code"],
    })

    weather_df["origin"] = airport_code

    return weather_df


# find min and max dates in flight data
start_date = flights["flight_date"].min().strftime("%Y-%m-%d")
end_date = flights["flight_date"].max().strftime("%Y-%m-%d")

print("\nFlight date range:", start_date, "to", end_date)

# download weather for each airport
all_weather = []

for code, coords in airports.items():
    print(f"Downloading weather for {code}...")

    airport_weather = get_weather_for_airport(
        airport_code=code,
        latitude=coords["latitude"],
        longitude=coords["longitude"],
        start_date=start_date,
        end_date=end_date
    )

    all_weather.append(airport_weather)
    sleep(1)

# combine all weather into one dataframe
weather = pd.concat(all_weather, ignore_index=True)

flights["weather_datetime"] = flights["dep_datetime"].dt.floor("h")

flights_with_weather = pd.merge(
    flights,
    weather,
    on=["origin", "weather_datetime"],
    how="left"
)

print("\nMissing weather values after merge:")
print(
    flights_with_weather[
        ["temperature_2m", "precipitation", "rain", "snowfall", "wind_speed_10m", "wind_gusts_10m", "visibility", "weather_code"]
    ].isnull().sum()
)

output_file = "nycflights_with_weather.csv"
flights_with_weather.to_csv(output_file, index=False)

print("\nDone!")
print("New file saved as:", output_file)

print("\nFinal shape:", flights_with_weather.shape)

print("\nFirst 5 rows:")
print(flights_with_weather.head())

print("\nFinal dtypes:")
print(flights_with_weather.dtypes)