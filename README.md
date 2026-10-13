# 🔍 Fraud Detection Platform

A machine-learning platform for credit-card fraud detection, combining a trained model, a Flask REST API, PostgreSQL persistence, a Streamlit dashboard, MLflow, Docker Compose, automated tests, and an optional AWS Lambda/API Gateway deployment.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/API-Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

**Repository:** [Wassef-Ben-Rgaya/fraud-detection-platform](https://github.com/Wassef-Ben-Rgaya/fraud-detection-platform)

## Overview

The project detects potentially fraudulent credit-card transactions using a Random Forest model served by a Flask API. Predictions can be recorded in PostgreSQL and accessed through API endpoints; a Streamlit dashboard provides a user interface. Docker Compose orchestrates PostgreSQL, MLflow, the API, and the dashboard. GitHub Actions runs project validation checks on pushes and pull requests.

The project uses the Credit Card Fraud Detection dataset, containing **284,807 transactions**, including **492 fraudulent transactions** (approximately **0.173%**).

## Verified local model results

The following metrics are the values currently returned by the local API's `/api/model` endpoint. They describe the configured local model; they should not automatically be assumed to describe a separately deployed AWS model.

| Item | Value |
|---|---:|
| Model | RandomForest |
| Version | `1.0.0` |
| Input features | 32 |
| F1-score | 0.8404 |
| Precision | 0.8778 |
| Recall | 0.8061 |
| ROC-AUC | 0.9796 |
| Dataset transactions | 284,807 |
| Fraudulent transactions | 492 |
| Fraud rate | 0.173% |

## Architecture

```text
Credit-card transaction dataset
             |
             v
     Data preparation / SQL
             |
             v
   Trained RandomForest model
             |
             v
       Flask REST API  <------> PostgreSQL
             |
             v
      Streamlit dashboard

Docker Compose services:
  PostgreSQL | MLflow | API | Dashboard

CI:
  GitHub Actions -> dependency installation and Python validation
                   plus model tests when required artifacts are available

Optional cloud component:
  API Gateway -> AWS Lambda -> model artifact in S3
```

The local Docker stack and the AWS serverless component are separate execution paths. Validate the AWS model artifact and cloud configuration independently before interpreting the cloud endpoint's metrics.

## Tech stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python | Data processing, model and API |
| Machine learning | Scikit-learn / RandomForest | Fraud classification |
| API | Flask, Gunicorn | HTTP prediction endpoints |
| Database | PostgreSQL 15 | Persisting transaction predictions and history |
| Experiment tracking | MLflow | Tracking model experiments |
| Dashboard | Streamlit | Interactive dashboard |
| Containers | Docker, Docker Compose | Local service orchestration |
| CI | GitHub Actions | Automated project checks |
| Optional cloud deployment | AWS Lambda, API Gateway, S3 | Serverless prediction endpoint and model storage |

## Project structure

```text
fraud-detection-platform/
├── .github/
│   └── workflows/
│       └── ci.yml
├── aws/
│   ├── lambda_function.py
│   ├── s3_upload.py
│   └── deploy_lambda.ps1
├── data/
│   ├── raw/                  # Dataset; keep large/private data out of Git
│   └── processed/            # Preprocessing artifacts
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── Dockerfile.mlflow
├── mlflow/                   # Local model artifacts / model metadata
├── sql/
│   ├── init.sql
│   ├── queries.sql
│   └── views.sql
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data/
│   └── models/
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── docker-compose.yml
├── requirements-api.txt
├── requirements-dashboard.txt
└── README.md
```

> Model files and `.env` configuration may be excluded from Git. Make sure the model artifacts exist locally before building/running services that depend on them.

## Prerequisites

- Windows with PowerShell, or Linux/macOS with equivalent commands
- Git
- Docker Desktop with Docker Compose
- Python 3.12+ for running local tests (the current development machine also ran the tests with Python 3.14)
- The project dataset and required model/preprocessing artifacts

## Quick start with Docker Compose

### 1. Clone the repository

```powershell
git clone https://github.com/Wassef-Ben-Rgaya/fraud-detection-platform.git
cd fraud-detection-platform
```

### 2. Configure environment variables

Create a local `.env` file based on the variables expected by `docker-compose.yml` and the project configuration. Do not commit `.env` or place real AWS credentials in source control. At minimum, configure the PostgreSQL credentials and the MLflow/database variables used by the compose services.

If your repository contains `.env.example`, use it as the starting point:

```powershell
Copy-Item .env.example .env
```

Review the values in `.env` before starting the stack.

### 3. Check and start the services

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

Wait until PostgreSQL, MLflow, and the API report healthy status and the dashboard has started.

### 4. Open the local services

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| Flask API health | http://localhost:5000/api/health |
| Model information | http://localhost:5000/api/model |
| MLflow UI | http://localhost:5001 |

Dashboard health check:

```powershell
Invoke-RestMethod http://localhost:8501/_stcore/health
```

### 5. Stop the services

```powershell
docker compose down
```

This stops and removes the containers but preserves named volumes. **Do not use `docker compose down -v` unless you intentionally want to delete the persisted PostgreSQL and MLflow volumes.**

## API reference

The local Flask blueprint is served under `/api`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | API and model health check |
| `GET` | `/api/model` | Model metadata, dataset summary and metrics |
| `GET` | `/api/metrics` | Model metrics |
| `POST` | `/api/predict` | Predict one transaction |
| `POST` | `/api/predict/batch` | Predict a batch of transactions |
| `GET` | `/api/predictions` | Retrieve prediction history |
| `GET` | `/api/database/health` | Check database connectivity |

### Example: health check

```powershell
Invoke-RestMethod http://localhost:5000/api/health
```

### Example: model information

```powershell
Invoke-RestMethod http://localhost:5000/api/model
```

### Example: prediction request

The endpoint expects a JSON object with a `features` array containing **32 numeric values** in the feature order expected by the trained model.

```powershell
$body = @{
    features = @(0.1, -0.2, 0.3) # Replace with all 32 numeric feature values
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
    -Uri "http://localhost:5000/api/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The abbreviated feature array above is illustrative only; a valid prediction request must contain all 32 values.

## Run tests

Keep Docker Compose running before executing the API integration tests, because those tests call the local API and database.

```powershell
python -m pytest tests -v
```

Latest verified local result: **13 tests passed** — 8 model tests and 5 API/database integration tests.

The GitHub Actions workflow has also completed successfully for the latest CI runs. Review the workflow logs in the repository's **Actions** tab for the exact checks performed by each run.

## Optional AWS deployment

The repository includes AWS-related deployment resources. The configured API Gateway endpoint previously used by this project is:

`https://u85vpnda58.execute-api.eu-north-1.amazonaws.com`

The prediction route is `POST /predict`. Cloud resources, permissions, the model artifact stored in S3, and Lambda configuration must remain consistent. Do not publish AWS access keys, secret keys, database passwords, or other credentials in this README or in Git.

The local API metrics listed above are not presented as AWS metrics. Before publishing a public demo, re-test the live endpoint and confirm which model artifact it is serving.

## Dataset

The project uses the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Download the dataset according to Kaggle's terms and place `creditcard.csv` in `data/raw/` if you intend to run the data-ingestion workflow locally. Do not commit the dataset unless you have confirmed that doing so is permitted and appropriate for your repository.

## Security notes

- Keep `.env`, AWS credentials, and database passwords out of Git.
- Use `.env.example` only for placeholder values, never real secrets.
- Rotate any AWS keys that have ever been committed or shared in plain text.
- Avoid exposing PostgreSQL or MLflow publicly without suitable network and access controls.

## Roadmap

- [ ] Add screenshots of the actual dashboard, MLflow UI and API response.
- [ ] Add a reproducible setup guide for generating/downloading model artifacts.
- [ ] Add CI coverage for service integration where it can run reliably in GitHub Actions.
- [ ] Add API authentication and rate limiting if the API is exposed publicly.
- [ ] Add monitoring for data quality and model drift.

## Author

**Wassef Ben Rgaya** — AI & Data Science Engineer

- [GitHub](https://github.com/Wassef-Ben-Rgaya)
- [LinkedIn](https://www.linkedin.com/in/wassef-ben-rgaya-600817188)
- [Email](mailto:wassefbenrgaya24@gmail.com)

---

If you find this project useful, consider giving the repository a star.
