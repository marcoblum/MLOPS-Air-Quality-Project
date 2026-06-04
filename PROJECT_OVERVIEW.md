# Air Quality Prediction Project

This project implements a fully automated MLOps pipeline to forecast air quality (PM2.5 levels) using real-time telemetry from OpenAQ and historical data.

For detailed setup instructions and reproducibility steps, please refer to the [README](README.md) file.

---

## Short Description & Architecture

This system operates on a three-tier **FTI (Feature, Training, Inference)** architecture.

A GitHub Actions workflow fetches live air quality telemetry hourly via the **OpenAQ API**, engineering features directly into the **Hopsworks Feature Store**. An offline training pipeline extracts these features to train an optimized **XGBoost regressor**, tracking artifacts back to the model registry. Finally, a containerized **Streamlit web application** serves live inference by streaming the latest state directly from the cloud feature store, ensuring real-time predictions without relying on local data persistence.

---



## Project Structure

The repository follows a modular MLOps architecture, separating feature engineering, model training, and inference services into dedicated components.

```text
MLOPS-AIR-QUALITY-PROJECT/
|
├── .github/
|   └── workflows/
|       └── feature_pipeline.yml        # scheduled feature ingestion
|
├── data/
|   └── processed/
|       ├── history/                    # historical backfill data
|       └── features_latest.parquet     # latest feature snapshot
|
├── models/
|   └── air_quality_model.pkl           # trained model artifact
|
├── src/
|   ├── feature_pipeline/               # fetches and transforms sensor data
|   ├── training_pipeline/              # model training and evaluation
|   └── app.py                          # Streamlit frontend
|
├── tests/                              # unit tests for transform_batch
|
├── .env.example                        # environment variable template
├── .gitignore
├── Dockerfile
└── requirements.txt
```

### Directory Overview

| Path                     | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| `.github/workflows/`     | Automated pipeline orchestration using GitHub Actions |
| `data/`                  | Runtime data storage (excluded from Git)              |
| `models/`                | Serialized model artifacts                            |
| `src/feature_pipeline/`  | Data ingestion and feature engineering                |
| `src/training_pipeline/` | Model training and optimization                       |
| `src/app.py`             | Streamlit-based prediction interface                  |
| `.env.example`           | Template for environment variables                    |
| `Dockerfile`             | Container deployment specification                    |
| `requirements.txt`       | Pinned Python dependencies                            |
| `tests/`                 | Unit tests for feature pipeline transformation

---

## Project Pipeline Overview

1. **Feature Pipeline**

   * Collects hourly air quality telemetry from OpenAQ.
   * Performs feature engineering.
   * Stores features in Hopsworks Feature Store.

2. **Training Pipeline**

   * Retrieves historical features from Hopsworks.
   * Trains and validates an XGBoost model.
   * Registers model artifacts.

3. **Inference Service**

   * Streamlit-based web application.
   * Retrieves latest features from Hopsworks.
   * Generates real-time PM2.5 forecasts.

---

## Tech Stack

| Area | Technology |
|------|------------|
| Data source | OpenAQ API |
| Data processing | Python, Pandas |
| Feature storage | Hopsworks Feature Store |
| Model training | XGBoost, scikit-learn |
| Frontend | Streamlit |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |
| Automation | GitHub Actions |
| CI | pytest |



---

## Automated Unit Testing
The core feature engineering and data transformation logic (`transform_batch`) 
is covered by automated unit tests, split across multiple files:

- **test_input_validation** – empty and invalid inputs
- **test_lag_features** – lag correctness (1h, 6h, 24h)
- **test_rolling_features** – rolling average and variance
- **test_imputation** – missing value handling via ffill
- **test_feature_completeness** – required columns and time features
- **test_app_fallback** – validates model schema compliance and defensive fallback imputation under incomplete live data streams

This ensures that features like rolling averages, lags, and data imputations 
remain mathematically correct even if structural source modifications occur.

