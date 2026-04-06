import requests
import pandas as pd

API_KEY = "DEMO_KEY"
START_DATE = "2026-04-01"
END_DATE = "2026-04-06"

url = "https://api.nasa.gov/neo/rest/v1/feed"
params = {
    "start_date": START_DATE,
    "end_date": END_DATE,
    "api_key": API_KEY
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()
data = response.json()

asteroid_rows = []

near_earth_objects = data["near_earth_objects"]

for date in near_earth_objects:
    asteroids_for_day = near_earth_objects[date]

    for asteroid in asteroids_for_day:
        diameter_info = asteroid["estimated_diameter"]["meters"]
        diameter_min_m = diameter_info["estimated_diameter_min"]
        diameter_max_m = diameter_info["estimated_diameter_max"]

        close_approach_data = asteroid["close_approach_data"]

        if len(close_approach_data) == 0:
            continue

        approach = close_approach_data[0]

        velocity_kph = float(approach["relative_velocity"]["kilometers_per_hour"])
        miss_distance_km = float(approach["miss_distance"]["kilometers"])
        close_approach_date = approach["close_approach_date"]

        row = {
            "name": asteroid["name"],
            "hazardous": int(asteroid["is_potentially_hazardous_asteroid"]),
            "diameter_min_m": diameter_min_m,
            "diameter_max_m": diameter_max_m,
            "velocity_kph": velocity_kph,
            "miss_distance_km": miss_distance_km,
            "close_approach_date": close_approach_date
        }

        asteroid_rows.append(row)

df = pd.DataFrame(asteroid_rows)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataframe shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())