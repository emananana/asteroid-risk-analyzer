import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

API_KEY = "DEMO_KEY"

url = "https://api.nasa.gov/neo/rest/v1/feed"

# We will pull several 7-day windows and combine them
date_ranges = [
    ("2026-03-01", "2026-03-07"),
    ("2026-03-08", "2026-03-14"),
    ("2026-03-15", "2026-03-21"),
    ("2026-03-22", "2026-03-28"),
    ("2026-03-29", "2026-04-04"),
]

asteroid_rows = []

for start_date, end_date in date_ranges:
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": API_KEY
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

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

# Remove duplicate asteroids if any appear across windows
df = df.drop_duplicates(subset=["name", "close_approach_date"])

print("\nFirst 5 rows:")
print(df.head())

print("\nDataframe shape:")
print(df.shape)

print("\nHazardous value counts:")
print(df["hazardous"].value_counts())

print("\nSummary statistics:")
print(df[["diameter_max_m", "velocity_kph", "miss_distance_km"]].describe())

hazardous_df = df[df["hazardous"] == 1]
non_hazardous_df = df[df["hazardous"] == 0]

plt.figure(figsize=(10, 6))
plt.scatter(non_hazardous_df["miss_distance_km"], non_hazardous_df["diameter_max_m"], label="Not Hazardous")
plt.scatter(hazardous_df["miss_distance_km"], hazardous_df["diameter_max_m"], label="Hazardous")
plt.xlabel("Miss Distance (km)")
plt.ylabel("Max Estimated Diameter (m)")
plt.title("Asteroid Risk: Size vs Distance from Earth")
plt.legend()
plt.tight_layout()
plt.savefig("asteroid_risk_plot_v2.png")

feature_columns = ["diameter_max_m", "velocity_kph", "miss_distance_km"]
X = df[feature_columns]
y = df["hazardous"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

results_df = X_test.copy()
results_df["actual_hazardous"] = y_test.values
results_df["predicted_hazardous"] = y_pred

print("\nPrediction Results:")
print(results_df)

plt.show()