# Wine Quality Prediction with MLFlow

An end-to-end machine learning pipeline for predicting wine quality using ElasticNet regression, with MLFlow for experiment tracking and Docker for deployment.

## Overview

This project builds a production-ready ML pipeline that predicts wine quality scores (0–10) from physicochemical properties such as acidity, alcohol content, and pH. MLFlow is used throughout for experiment tracking, model versioning, and reproducibility.

## Dataset

**Red Wine Quality** — UCI Machine Learning Repository

| Property | Value |
|----------|-------|
| Samples | 1,599 |
| Features | 11 (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free SO₂, total SO₂, density, pH, sulphates, alcohol) |
| Target | Quality score (0–10, integer) |

## Model

**ElasticNet Regression** — combines L1 (Lasso) and L2 (Ridge) regularisation.

Key hyperparameters tracked via MLFlow:
- `alpha` — regularisation strength
- `l1_ratio` — balance between L1 and L2 penalty

## Project Structure

```
Wine-Quality-MLFlow/
    app.py                  Flask web app for inference
    main.py                 Pipeline entry point
    params.yaml             Model hyperparameters
    schema.yaml             Feature schema definitions
    config/
        config.yaml         Pipeline configuration (paths, URIs)
    research/
        trials.ipynb        Exploratory training experiments
    src/
        Wine-Quality-MLFlow/
            components/     Data ingestion, validation, transformation, training
            config/         Configuration manager
            entity/         Data classes
            pipeline/       Training & prediction pipelines
            utils/          Common utilities
    templates/              HTML templates for web UI
    Dockerfile              Container definition for deployment
    requirements.txt
```

## Pipeline Stages

1. **Data Ingestion** — download and split dataset
2. **Data Validation** — schema checks against `schema.yaml`
3. **Data Transformation** — feature scaling and preprocessing
4. **Model Training** — ElasticNet with params from `params.yaml`
5. **Model Evaluation** — log metrics (RMSE, MAE, R²) to MLFlow

## Setup

```bash
git clone https://github.com/Chirag-Mokashi/Wine-Quality-MLFlow
cd Wine-Quality-MLFlow
pip install -r requirements.txt
```

## Run

```bash
# Run full training pipeline
python main.py

# Launch web app
python app.py
```

Visit `http://localhost:5000` to predict wine quality via the web interface.

## Docker

```bash
docker build -t wine-quality .
docker run -p 5000:5000 wine-quality
```

## MLFlow Tracking

MLFlow logs are stored locally. To view the experiment dashboard:

```bash
mlflow ui
```

Open `http://localhost:5000` in your browser to compare runs, parameters, and metrics.

## Tech Stack

- Python, Scikit-learn, Pandas, NumPy
- MLFlow (experiment tracking & model registry)
- Flask (web interface)
- Docker (containerisation)
- PyYAML (config management)