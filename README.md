# 🚀 Asteroid Risk Analyzer 🚀

A machine learning project using NASA Near-Earth Object data to analyze and classify potentially hazardous asteroids.

## Problem

Monitoring near-Earth objects is critical for space safety, but identifying potentially hazardous asteroids requires analyzing multiple factors such as size, velocity, and proximity to Earth.

This project builds a machine learning system to automatically assess asteroid risk using real NASA data.

## Features
- Fetches real asteroid data from NASA API
- Processes telemetry (size, velocity, distance)
- Applies machine learning for risk analysis
- Visualizes asteroid risk patterns

## Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib
- NASA NeoWs API

## Pipeline

1. Fetch asteroid data from NASA NeoWs API
2. Parse and clean nested JSON into structured features
3. Perform exploratory data analysis and visualization
4. Train a logistic regression model for classification
5. Address class imbalance using weighted loss
6. Evaluate model performance using accuracy and classification metrics

## Model Performance

- Accuracy: ~86%
- Hazardous asteroid recall: 100%
- Achieves high recall on hazardous asteroids (100%), prioritizing detection of critical cases

Note: Dataset was initially imbalanced, addressed using class-weighted logistic regression and expanded data collection.

## Visualization

![Asteroid Risk Plot](asteroid_risk_plot_v2.png)

## Key Insight

Hazardous asteroids tend to be larger and closer to Earth, but no single feature fully determines risk. A machine learning model is effective for capturing these multi-factor relationships.

## Limitations & Future Work

- Dataset size is limited due to API constraints; future work could aggregate larger historical datasets
- Model performance could be improved using more advanced models (e.g., random forests, gradient boosting)
- Real-time streaming and alert systems could be added for operational use