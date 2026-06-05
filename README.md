# Air Quality Prediction Project
This repository implements a cloud-based MLOps pipeline for short-term PM2.5 air quality forecasting.

Developed as part of the I.BA_MLOPS_MM.F2601 Machine Learning Operations module at HSLU (Spring Semester 2026).

For a brief project overview, see the project [summary](PROJECT_OVERVIEW.md).

Also, make sure to check out the Deployed Application: https://huggingface.co/spaces/Balumi13/Air-Quality



## Reproducibility (Clone and Run)

Follow these steps to replicate the environment and run the pipeline components locally or via containerization.

### Prerequisites

* Python 3.10
* Docker
* Hopsworks Account
* Hugging Face Account
* OpenAQ Account

---

## 1. Clone the Repository

```bash
git clone https://github.com/marcoblum/MLOPS-Air-Quality-Project.git
cd MLOPS-AIR-QUALITY-PROJECT
```

---

## 2. Environment Setup

Duplicate the provided template to create your local environment file:

```bash
cp .env.example .env
```

Open the newly created `.env` file and fill in your private values. The complete set of variables is:

```env
OPENAQ_API_KEY=your_key_here
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT=your_unique_project_name
HF_TOKEN=your_token_here
HF_REPO_ID=your_username/your_space_name
```

### Required Credentials

#### OpenAQ API Key

1. Register on OpenAQ.
2. Generate an API key. (You may use this guide: https://docs.openaq.org/using-the-api/quick-start)
3. Add it to:

```env
OPENAQ_API_KEY=your_key_here
```

#### Hopsworks API Key

1. Log in to your Hopsworks account.
2. Navigate to **Account Settings -> API Keys**.
3. Create a key with project access.
4. Add it to:

```env
HOPSWORKS_API_KEY=your_key_here
```

#### Hopsworks Project Name

> **Important:** On Hopsworks Serverless (`app.hopsworks.ai`), project names are **globally unique across the entire platform**, not just within your own account. You therefore **cannot** reuse the original author's project name (`AeroPredict`) - it already exists and is owned by another account. Trying to create it manually will fail with a *"project already exists"* error, and the pipeline will not be able to log in to it.

1. Log in to your Hopsworks account.
2. Create a **new project** with a name that is unique to you (e.g., `AeroPredict_<yourname>`).
3. Add that exact name to your `.env` file:

```env
HOPSWORKS_PROJECT=your_unique_project_name
```

The feature pipeline, the training pipeline, and the app all read this variable when connecting to the Feature Store, so make sure it matches the project you created.

#### Hugging Face Token

1. Log in to your Hugging Face account.
2. Open **Settings -> Access Tokens** in Hugging Face.
3. Create a token with **Write** permissions.
4. Add it to:

```env
HF_TOKEN=your_token_here
```

#### Hugging Face Space (HF_REPO_ID)

> **Important:** The upload calls in the pipeline and training scripts push files into an **existing** Space - they do **not** create one automatically. You must create your own Space first (see Section 3, Phase 1) before referencing it here, otherwise the upload step will fail (the rest of the pipeline still runs and only prints a warning).

1. Create a Hugging Face Space under your own account (Section 3, Phase 1).
2. Add its identifier in the form `username/space_name` to your `.env` file:

```env
HF_REPO_ID=your_username/your_space_name
```

---


## 3. Cloud Deployment (Hugging Face Spaces)

To deploy the inference application to the cloud, follow these step-by-step instructions to mirror the production environment.

### Phase 1: Create the Space on Hugging Face
1. Log in to your **Hugging Face** account and click on **New Space**.
2. Name your Space (e.g., `Air-Quality`) and select **Streamlit** as the SDK.
3. Choose the **Blank** template and set the Space visibility to **Public**.
4. Click **Create Space** to initialize the infrastructure.

> The full identifier of this Space (e.g., `your_username/Air-Quality`) is the value you put in `HF_REPO_ID` in your `.env` file.

### Phase 2: Upload Project Files
To get the application running, you need to upload the core project files into the main directory (Root) of your new Space:

1. In your Space, click on the **Files** tab and select **Add file -> Upload files**.
2. Drag and drop the following files and folders from your local file explorer:
   * The `src/` folder (which contains your code and the `app.py`)
   * `Dockerfile`
   * `requirements.txt`
   * `.gitignore`
3. Scroll down and click **Commit changes to main**.

Hugging Face will automatically detect the `Dockerfile`, install the pinned dependencies, and launch the active Streamlit app in the cloud.

### Phase 3: Configure Environment Variables (Secrets)
Because your private `.env` file is excluded via `.gitignore` for security reasons, you must inject your API keys directly into the Hugging Face infrastructure so the cloud container can authenticate with your data services:

1. In your Hugging Face Space, navigate to the **Settings** tab.
2. Scroll down to the **Variables and secrets** section and click on **New secret**.
3. Add the following secrets using your personal values:
   * **Key:** `OPENAQ_API_KEY` / **Value:** *Your OpenAQ API Key*
   * **Key:** `HOPSWORKS_API_KEY` / **Value:** *Your Hopsworks API Key*
   * **Key:** `HOPSWORKS_PROJECT` / **Value:** *Your unique Hopsworks project name*
   * **Key:** `HF_TOKEN` / **Value:** *Your Hugging Face Token (with Write access)*
   * **Key:** `HF_REPO_ID` / **Value:** *Your Space identifier (`username/space_name`)*
4. Click **Save** for each entry.

> The inference app itself only needs `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` to read from the Feature Store. `HF_TOKEN` and `HF_REPO_ID` are used by the feature and training pipelines (locally or via the scheduled GitHub Action) that push data and the model back to the Space.

Once the secrets are saved, click on the **Factory rebuild** button at the top of the settings page to restart the container with the active environment variables.

---

## 4. Local Execution

If you prefer to run the complete pipeline locally, follow the steps below.

> **Before you start:** Make sure you have (1) created your own **Hopsworks project** and set `HOPSWORKS_PROJECT`, and (2) created your own **Hugging Face Space** and set `HF_REPO_ID` (see Sections 2 and 3). Both must exist before the first run.

### Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv .venv
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows (Command Prompt)

```bash
.venv\Scripts\activate.bat
```

#### Windows (PowerShell)

```bash
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 1 - Initial Backfill (run once)

When you set up the project for the first time, your Hopsworks Feature Store is still empty. Before any hourly run or model training will work, you must populate it **once** with the historical dataset (~2 years of data). This is what the `--backfill` flag is for:

```bash
python src/feature_pipeline/run_feature_pipeline.py --backfill
```

This step iterates over the full history in chunks and uploads the aggregated dataset to the `air_quality_features_1` feature group inside the project defined by `HOPSWORKS_PROJECT`. It only needs to be run **once** during initial setup and takes considerably longer than a normal run.

### Step 2 - Hourly Feature Ingestion & Engineering

After the initial backfill, the regular (live) run fetches only the most recent days, computes features, and appends them to the same feature group. This is the command used by the scheduled GitHub Action:

```bash
python src/feature_pipeline/run_feature_pipeline.py
```

### Step 3 - Run Model Training & Validation

Once the Feature Store contains data, train the forecasting model:

```bash
python src/training_pipeline/train_model.py
```

### Step 4 - Launch the Streamlit Inference App

```bash
streamlit run src/app.py
```

---

## 5. Containerized Execution (Docker)

To run the inference application inside a production-like environment (e.g., similar to Hugging Face Spaces), use the provided Docker configuration.

### Build the Docker Image

```bash
docker build -t air-quality-app .
```

### Run the Container

```bash
docker run --env-file .env -p 7860:7860 air-quality-app
```

---

## Access the Application

Once the container has started successfully, open your browser and navigate to:

```text
http://localhost:7860
```

The Streamlit interface will connect directly to the cloud feature store and provide real-time PM2.5 predictions.

---

## 6. Automated Unit Testing

To execute the test suite locally, run the following command in your terminal:

```bash
pytest
```
For a detailed breakdown of the test suite, refer to the [Project Overview](PROJECT_OVERVIEW.md#automated-unit-testing).