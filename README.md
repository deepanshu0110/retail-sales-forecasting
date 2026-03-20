# Retail Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Time Series](https://img.shields.io/badge/Type-Time%20Series-blueviolet?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A comprehensive end-to-end retail sales forecasting system — from synthetic data generation to production API deployment. Demonstrates professional-grade ML implementation with multiple forecasting models and business-ready evaluation metrics.

---

## Business Problem

Retail businesses need accurate sales forecasts to optimize inventory, reduce waste, and plan promotions. This pipeline forecasts future sales using historical patterns, seasonality, and promotional data — directly applicable to demand planning and supply chain optimization.

---

## Model Results

| Model | MAE | RMSE | MAPE | Notes |
|---|---|---|---|---|
| Naive Baseline | High | High | ~25% | Benchmark |
| Linear Trend | Medium | Medium | ~18% | Good for stable series |
| Exponential Smoothing | Medium | Medium | ~15% | Handles seasonality |
| **Random Forest** | **Low** | **Low** | **~10%** | Best model |

---

## Features

- Realistic synthetic data with seasonal patterns, promotions, and holidays
- Professional data preprocessing with feature engineering (lag features, rolling stats)
- 4 forecasting models with automatic best-model selection
- Business metrics: MAE, RMSE, MAPE — not just R2
- Production-ready FastAPI deployment
- Numbered pipeline scripts for reproducible execution

---

## Project Structure

```
retail-sales-forecasting/
├── 01_generate_data.py       # Synthetic data with seasonality
├── 02_preprocess_data.py     # Feature engineering pipeline
├── 03_run_eda.py             # Exploratory data analysis
├── 04_train_models.py        # Model training & comparison
├── 05_deployment_api.py      # Production FastAPI server
├── config.py                 # Project configuration
├── requirements.txt
├── data/
│   ├── raw/                  # Generated sales data
│   └── processed/            # Engineered features
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── eda.py
│   └── models.py
├── models/                   # Serialized model artifacts
└── results/
    ├── plots/                # EDA & forecast visualizations
    └── forecasts/            # Model output CSVs
```

---

## Quickstart

```bash
git clone https://github.com/deepanshu0110/retail-sales-forecasting.git
cd retail-sales-forecasting
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Run full pipeline in order
python 01_generate_data.py
python 02_preprocess_data.py
python 03_run_eda.py
python 04_train_models.py

# Start production API
python 05_deployment_api.py
```

---

## API Usage

```bash
# Forecast next 7 days for store 1
curl -X POST http://localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"store_id": 1, "horizon": 7}'
```

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Statsmodels · FastAPI · Matplotlib

---

## License

MIT License