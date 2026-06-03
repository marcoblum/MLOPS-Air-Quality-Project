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
├── tests/
|   └── test_features.py                # unit tests for transform_batch
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

* Python 3.10
* OpenAQ API
* Hopsworks Feature Store
* XGBoost
* Streamlit
* GitHub Actions
* Docker
* Hugging Face Spaces





---

