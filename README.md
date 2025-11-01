# NexGen — Predictive Delivery Optimizer

## Overview

This Streamlit app predicts delivery delays and recommends corrective actions using historical orders, delivery performance, fleet, routes, costs and customer feedback.

## Features

- Train a delay classifier (and delay-regression if data exists)
- Interactive dashboard with 5+ visualizations
- Order-level scoring and batch scoring
- Rule-based action recommendations
- Business impact simulation (cost savings)

## Requirements

Python 3.9+
Install:
pip install -r requirements.txt

## Files expected (place in same folder or upload via sidebar)

- orders.csv
- delivery_performance.csv
- routes_distance.csv
- vehicle_fleet.csv
- warehouse_inventory.csv
- customer_feedback.csv
- cost_breakdown.csv

## Run

streamlit run app.py

## Notes

- The app tries to be robust to missing columns. For model training you need historical rows with `promised_time` and `actual_time` or a labelled `is_delayed`.
- Trained models saved as `delay_classifier.joblib` and `delay_regressor.joblib`.

## Extensions

- Add geolocation maps (lat/lon) for hot-spot mapping.
- Integrate with carrier APIs to trigger reroutes automatically.
- Add prescriptive optimizer for vehicle assignment with capacity constraints.
