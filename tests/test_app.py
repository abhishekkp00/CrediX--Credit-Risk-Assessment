from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module


class StubModel:
    def predict_proba(self, frame):
        self.last_frame = frame.copy()
        return np.array([[0.25, 0.75]])


@contextmanager
def configured_client(monkeypatch, *, saved_threshold=0.567, env_threshold=None):
    model = StubModel()

    def fake_load(path):
        if str(path).endswith("credit_risk_model.pkl"):
            return model
        if str(path).endswith("best_threshold.pkl"):
            return saved_threshold
        raise AssertionError(f"Unexpected model artifact: {path}")

    monkeypatch.setattr(app_module.joblib, "load", fake_load)
    monkeypatch.delenv("CREDIX_THRESHOLD", raising=False)
    if env_threshold is not None:
        monkeypatch.setenv("CREDIX_THRESHOLD", str(env_threshold))

    with TestClient(app_module.app) as client:
        yield client, model


VALID_APPLICATION = {
    "person_age": 30,
    "person_income": 600000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 100000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6,
}


def test_health_endpoint_returns_healthy(monkeypatch):
    with configured_client(monkeypatch) as (client, _):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_serves_frontend(monkeypatch):
    with configured_client(monkeypatch) as (client, _):
        response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()


def test_openapi_docs_are_available(monkeypatch):
    with configured_client(monkeypatch) as (client, _):
        response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]


def test_saved_threshold_is_used_by_default(monkeypatch):
    with configured_client(monkeypatch, saved_threshold=0.567) as (client, _):
        response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 200
    assert response.json()["threshold"] == pytest.approx(0.567)


def test_environment_threshold_overrides_saved_value(monkeypatch):
    with configured_client(
        monkeypatch, saved_threshold=0.567, env_threshold=0.30
    ) as (client, _):
        response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 200
    assert response.json()["threshold"] == pytest.approx(0.30)


def test_prediction_returns_positive_class_probability(monkeypatch):
    with configured_client(monkeypatch) as (client, _):
        response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 200
    assert response.json()["default_probability"] == pytest.approx(0.75)


def test_probability_above_threshold_is_high_risk(monkeypatch):
    with configured_client(monkeypatch, env_threshold=0.567) as (client, _):
        response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 200
    assert response.json()["default_prediction"] == 1
    assert response.json()["result"] == "High Risk"


def test_probability_below_threshold_is_low_risk(monkeypatch):
    with configured_client(monkeypatch, env_threshold=0.80) as (client, _):
        response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 200
    assert response.json()["default_prediction"] == 0
    assert response.json()["result"] == "Low Risk"


def test_invalid_application_is_rejected(monkeypatch):
    invalid_application = {**VALID_APPLICATION, "person_age": 0}
    with configured_client(monkeypatch) as (client, _):
        response = client.post("/predict", json=invalid_application)
    assert response.status_code == 422


def test_prediction_returns_503_when_model_is_unavailable(monkeypatch):
    with configured_client(monkeypatch) as (client, _):
        app_module.ml_model.clear()
        response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 503


def test_threshold_outside_valid_range_fails_startup(monkeypatch):
    with pytest.raises(ValueError, match="CREDIX_THRESHOLD"):
        with configured_client(monkeypatch, env_threshold=1.0):
            pass
    assert app_module.ml_model == {}
