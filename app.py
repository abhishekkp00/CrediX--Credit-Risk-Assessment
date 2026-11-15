from contextlib import asynccontextmanager
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_path = BASE_DIR / "credit_risk_model.pkl"
        threshold_path = BASE_DIR / "best_threshold.pkl"

        saved_threshold = float(joblib.load(threshold_path))
        configured_threshold = os.getenv("CREDIX_THRESHOLD")
        threshold = (
            saved_threshold
            if configured_threshold is None
            else float(configured_threshold)
        )
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "CREDIX_THRESHOLD must be a number strictly between 0 and 1."
            )

        model = joblib.load(model_path)
        ml_model["model"] = model
        ml_model["threshold"] = threshold
        yield
    finally:
        ml_model.clear()


app = FastAPI(
    title="CrediX - Credit Risk Assessment API",
    description="API for predicting loan default risk using a trained ML model.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoanApplication(BaseModel):
    person_age: int = Field(gt=0, le=120)
    person_income: float = Field(gt=0)
    person_home_ownership: str
    person_emp_length: float = Field(ge=0)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(gt=0, le=100)
    loan_percent_income: float = Field(ge=0)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(ge=0)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: LoanApplication):
    if "model" not in ml_model or "threshold" not in ml_model:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is not loaded.",
        )

    try:
        input_df = pd.DataFrame([data.model_dump()])
        probability = float(
            ml_model["model"].predict_proba(input_df)[0, 1]
        )
        threshold = ml_model["threshold"]
        prediction = int(probability >= threshold)

        return {
            "default_probability": probability,
            "default_prediction": prediction,
            "threshold": threshold,
            "result": "High Risk" if prediction else "Low Risk",
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc


app.mount(
    "/",
    StaticFiles(directory=str(BASE_DIR / "static"), html=True),
    name="static",
)
