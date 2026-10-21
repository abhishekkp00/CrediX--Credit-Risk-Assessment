from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# Load ML artifacts during application startup
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["model"] = joblib.load("credit_risk_model.pkl")
    ml_model["threshold"] = joblib.load("best_threshold.pkl")

    yield

    ml_model.clear()

app = FastAPI(
    title="CrediX - Credit Risk Assessment API",
    description="API for predicting loan default risk using a trained ML model.",
    version="1.0.0",
    lifespan=lifespan,
)


# Enable CORS if your frontend runs on a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your frontend origin in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input schema: features supplied by the user.
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

        # Probability of the positive class (default).
        probability = float(
            ml_model["model"].predict_proba(input_df)[0, 1]
        )

        threshold = float(ml_model["threshold"])
        prediction = int(probability >= threshold)

        return {
            "default_probability": probability,
            "default_prediction": prediction,
            "threshold": threshold,
            "result": "High Risk" if prediction == 1 else "Low Risk",
        }

    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc


# Serve the frontend from the static/ directory.
app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static",
)