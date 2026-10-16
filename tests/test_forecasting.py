import importlib.util
import sys

import pandas as pd

from tests.helpers import ROOT, run_script

sys.path.insert(0, str(ROOT))


def test_features_never_use_the_target_day():
    from src.forecast_features import build_features

    dates = pd.date_range("2021-01-01", "2022-12-31", freq="D")
    sales = pd.Series(range(len(dates)), index=dates, dtype=float)
    target_day = pd.Timestamp("2022-06-15")
    before = build_features([target_day], sales, dates[0])
    changed = sales.copy()
    changed[target_day] = 1e9  # a wildly different value on the day being forecast
    after = build_features([target_day], changed, dates[0])
    pd.testing.assert_frame_equal(before, after)


def test_pipeline_runs_and_random_forest_wins():
    run_script("01_generate_data.py")
    run_script("02_preprocess_data.py")
    output = run_script("04_train_models.py", timeout=1500)
    assert "BEST MODEL: Random Forest" in output
    assert "leakage-free" in output


def test_api_forecast_changes_day_to_day():
    spec = importlib.util.spec_from_file_location("api", ROOT / "05_deployment_api.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    api.initialize_app()
    client = api.app.test_client()

    body = client.post("/forecast", json={"store_id": 1, "days_ahead": 7}).get_json()
    assert body["model_used"] == "random_forest"
    assert len(set(body["forecast_values"])) > 1, "forecast should not be flat"

    promo = client.post("/forecast", json={
        "store_id": 1, "days_ahead": 7, "model": "Random_Forest",
        "promotions": [0, 0, 1, 0, 0, 0, 0],
    }).get_json()
    assert promo["model_used"] == "random_forest"
    assert promo["forecast_values"][2] > body["forecast_values"][2]
