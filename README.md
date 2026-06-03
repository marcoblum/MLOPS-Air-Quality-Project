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

Open the newly created `.env` file and fill in your private API keys.

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
2. Navigate to **Account Settings → API Keys**.
3. Create a key with project access.
4. Add it to:

```env
HOPSWORKS_API_KEY=your_key_here
```

#### Hugging Face Token

1. Log in to your Hugging Face account.
2. Open **Settings → Access Tokens** in Hugging Face.
3. Create a token with **Write** permissions.
4. Add it to:

```env
HF_TOKEN=your_token_here
```

---


## 3. Cloud Deployment (Hugging Face Spaces)

To deploy the inference application to the cloud, follow these step-by-step instructions to mirror the production environment.

### Phase 1: Create the Space on Hugging Face
1. Log in to your **Hugging Face** account and click on **New Space**.
2. Name your Space (e.g., `Air-Quality`) and select **Streamlit** as the SDK.
3. Choose the **Blank** template and set the Space visibility to **Public**.
4. Click **Create Space** to initialize the infrastructure.

### Phase 2: Upload Project Files
To get the application running, you need to upload the core project files into the main directory (Root) of your new Space:

1. In your Space, click on the **Files** tab and select **Add file → Upload files**.
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
3. Add the following three secrets using your personal keys:
   * **Key:** `OPENAQ_API_KEY` / **Value:** *Your OpenAQ API Key*
   * **Key:** `HOPSWORKS_API_KEY` / **Value:** *Your Hopsworks API Key*
   * **Key:** `HF_TOKEN` / **Value:** *Your Hugging Face Token (with Write access)*
4. Click **Save** for each entry.

Once the secrets are saved, click on the **Factory rebuild** button at the top of the settings page to restart the container with the active environment variables.

---

## 4. Local Execution

If you prefer to run the complete pipeline locally, follow the steps below.

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

### Run Feature Ingestion & Engineering

```bash
python src/feature_pipeline/run_feature_pipeline.py
```

### Run Model Training & Validation

```bash
python src/training_pipeline/train_model.py
```

### Launch the Streamlit Inference App

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

## Automated Unit Testing

To execute the test suite locally, run the following command in your terminal:

```bash
pytest
```
For a detailed breakdown of the test suite, refer to the [Project Overview](PROJECT_OVERVIEW.md#automated-unit-testing).
