# Retail Sales Forecasting
[![CI](https://github.com/deepanshu0110/retail-sales-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/deepanshu0110/retail-sales-forecasting/actions)


![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

End-to-end retail demand forecasting system — synthetic data to production API. Compares four models and selects the best by MAPE.

---

## Business Problem

Inaccurate demand forecasts lead to overstocking or stockouts. This pipeline gives retail teams a data-driven forecast deployable as an API.

---

## Model Results

| Model | MAPE |
|---|---|
| Naive Baseline | ~25% |
| Linear Trend | ~18% |
| Exponential Smoothing | ~15% |
| **Random Forest** | **~10%** |

---

## Quickstart

```bash
git clone https://github.com/deepanshu0110/retail-sales-forecasting.git
cd retail-sales-forecasting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 01_generate_data.py
python 02_preprocess_data.py
python 03_run_eda.py
python 04_train_models.py
python 05_deployment_api.py
```

## API Usage

```bash
curl -X POST http://localhost:8000/forecast -H 'Content-Type: application/json' -d '{"store_id": 1, "horizon": 7}'
```

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Statsmodels · FastAPI · Matplotlib

---

## Author

**Deepanshu Garg** — Freelance Data Scientist
- GitHub: [@deepanshu0110](https://github.com/deepanshu0110)
- Hire: [freelancer.com/u/deepanshu0110](https://www.freelancer.com/u/deepanshu0110)

MIT License