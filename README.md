# NYC Taxi Demand Forecasting

An end-to-end machine learning pipeline for forecasting hourly taxi demand across key NYC zones, built to support zone-level capacity planning, driver deployment, and surge pricing decisions.

## Overview

This project constructs a panel dataset at hourly × zone resolution for five representative NYC taxi zones (2019–2024), integrating trip records, weather, holiday, and major event data to build and evaluate demand forecasting and regime classification models.

**Key Results:**

| Metric | Value |
|--------|-------|
| XGBoost Test R² | **0.932 (RMSE: 36.41)** |
| Regime Classifier Accuracy | **89%** |
| Regime Classifier Macro-F1 | **0.90** |
| Dataset Size | **~183,000 hourly observations** |

---

## Zones Covered

| Zone ID | Zone Name | Type |
|---------|-----------|------|
| 230 | Times Sq / Theatre District | Tourism & Entertainment |
| 261 | World Trade Center | Financial District |
| 79 | East Village | Residential & Nightlife |
| 237 | Upper East Side South | High-income Residential |
| 132 | JFK Airport | Transportation Hub |

---

## Project Structure

```
nyc-taxi-demand-forecasting/
│
├── LoadData.ipynb          # Data ingestion pipeline (TLC + Weather)
├── Master_Notebook.ipynb   # Full ML pipeline: EDA, modeling, evaluation
└── README.md
```

---

## Data Sources

- **NYC TLC Yellow Taxi Trip Records** (2019–2024): Monthly parquet files from the [TLC CloudFront archive](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Weather Data**: Hourly NYC weather from [Open-Meteo Historical Weather Archive API](https://open-meteo.com/en/docs/historical-forecast-api)
- **Holiday Calendar**: US/NY public holidays via the `holidays` Python library
- **Major NYC Events**: NYC Marathon, Times Square NYE (manually flagged by time window)

---

## Pipeline

### 1. Data Ingestion (`LoadData.ipynb`)
- Downloads monthly TLC Yellow Taxi parquet files (2019–2024)
- Filters to 5 target zones and business hours (6am–10pm)
- Aggregates to hourly `(zone_id, timestamp)` level
- Fetches hourly weather from Open-Meteo API

### 2. Data Processing
- Merges taxi demand and weather on shared timestamp
- Adds calendar features: holidays, major events, time components
- Handles missing values and duplicate key integrity checks

### 3. Feature Engineering
- **Cyclical time encodings**: hour, day-of-week, month (sin/cos transforms)
- **Demand memory**: lag-1h, lag-24h, lag-168h, 4h rolling average
- **Weather flags**: `is_raining`, `is_snowing`, `is_extreme_cold`, `is_windy`
- **Calendar flags**: `is_holiday`, `is_major_event`
- **Zone indicators**: one-hot encoded

### 4. Modeling

#### Demand Forecasting (Regression)
Time-aware train/test split using `TimeSeriesSplit` to prevent data leakage.

| Model | Test RMSE | Test R² |
|-------|-----------|---------|
| Linear Regression | 60.67 | 0.812 |
| Random Forest | 38.36 | 0.925 |
| **XGBoost** | **36.41** | **0.932** |

#### Demand Regime Classification
- **KMeans clustering** (k=3) → Low / Medium / High demand tiers
- **Random Forest Classifier** to predict regime labels
  - Test Accuracy: **89%**, Macro-F1: **0.90**

#### Key Drivers (Feature Importance)
1. 1-hour demand lag
2. 4-hour rolling mean
3. Weekly demand lag (168h)
4. Hour-of-day encodings
5. Zone indicators

---

## Key Findings

- **Demand is momentum-driven** — recent demand lags dominate feature importance; weather and events act as secondary modifiers, not primary drivers
- **Strong time structure** — demand peaks 5–7 PM daily, weekdays consistently outperform weekends, patterns stable across 2019–2024
- **Zone heterogeneity is significant** — Upper East Side South shows highest average demand; WTC is concentrated in business hours; JFK generates highest revenue per hour
- **Weather & events fine-tune the baseline** — rain increases demand in residential zones; holidays suppress commuter zone activity; major events cause localized spikes in Times Square
- **Nonlinearity matters** — XGBoost (R²=0.932) significantly outperforms Linear Regression (R²=0.812), confirming strong nonlinear interactions between features
