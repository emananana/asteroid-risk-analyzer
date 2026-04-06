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

print("Top-level keys:", data.keys())
print("Dates returned:", list(data["near_earth_objects"].keys()))

first_date = list(data["near_earth_objects"].keys())[0]
first_asteroid = data["near_earth_objects"][first_date][0]

print("\nSample asteroid:")
print("Name:", first_asteroid["name"])
print("Hazardous:", first_asteroid["is_potentially_hazardous_asteroid"])

diameter_info = first_asteroid["estimated_diameter"]["meters"]
print("Estimated diameter min (m):", diameter_info["estimated_diameter_min"])
print("Estimated diameter max (m):", diameter_info["estimated_diameter_max"])

approach = first_asteroid["close_approach_data"][0]
print("Relative velocity (km/h):", approach["relative_velocity"]["kilometers_per_hour"])
print("Miss distance (km):", approach["miss_distance"]["kilometers"])
print("Orbiting body:", approach["orbiting_body"])