# CrediX - Credit Risk Assessment

CrediX is a credit risk assessment application that predicts whether a loan applicant is likely to default based on their financial and borrower profile. The project combines a machine learning model with a FastAPI backend and a frontend for real-time risk scoring.

## Project Overview

The application accepts applicant and loan details and returns a predicted default probability, risk classification, and the decision threshold used.

## Tech Stack

- Python 3.10+
- FastAPI and Uvicorn
- Pandas, scikit-learn, XGBoost, and Joblib
- HTML, CSS, and JavaScript

## Project Structure

```text
CrediX--Credit-Risk-Assessment/
├── app.py
├── Credit_Risk.ipynb
├── credit_risk_dataset.csv
├── credit_risk_model.pkl
├── best_threshold.pkl
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
└── README.md
```

## Model and Prediction Flow

The API loads the serialized calibrated model and saved decision threshold at startup. The `POST /predict` endpoint returns:

- `default_probability`: predicted probability of default
- `default_prediction`: binary decision using the active threshold
- `threshold`: threshold applied by this running API instance
- `result`: `High Risk` or `Low Risk`

The default threshold is loaded from `best_threshold.pkl`. You can override it without modifying the model artifact by setting `CREDIX_THRESHOLD` to a number strictly between 0 and 1.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:

- Frontend: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Example threshold override:

```bash
CREDIX_THRESHOLD=0.30 uvicorn app:app --reload
```

Without the variable, the saved threshold remains in use.

## Run with Docker

Build the image from the repository root:

```bash
docker build -t credix .
```

Run the container:

```bash
docker run --rm -p 8000:8000 credix
```

To override the threshold inside the container:

```bash
docker run --rm -p 8000:8000 -e CREDIX_THRESHOLD=0.30 credix
```

The image includes the API, frontend assets, dependencies, and the two model artifacts. The notebook and training dataset are excluded from the image to keep it smaller.

## Prediction Request

```http
POST /predict
Content-Type: application/json
```

```json
{
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
  "cb_person_cred_hist_length": 6
}
```

Example response (illustrative):

```json
{
  "default_probability": 0.3124,
  "default_prediction": 0,
  "threshold": 0.567,
  "result": "Low Risk"
}
```

## Machine Learning Notebook

`Credit_Risk.ipynb` contains the data analysis, preprocessing, model training, evaluation, threshold selection, probability calibration, SHAP interpretation, and model serialization workflow.

The SHAP section uses a model-agnostic explainer around the calibrated model's `predict_proba` method. Its feature contributions therefore target the calibrated default probability returned by the API, rather than only the underlying uncalibrated XGBoost score. To keep notebook execution manageable, it explains a reproducible sample of up to 100 test applicants. Re-run the notebook cells from calibration onward to refresh saved plots and outputs.

## Tests

Install the development dependencies and run the API regression suite:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers the health check, frontend serving, API schema, prediction output, threshold selection and validation, request validation, and the unavailable-model response.

## Notes

This project is an educational demonstration. Its risk predictions should not be used as the sole basis for real lending decisions without appropriate validation, monitoring, governance, and regulatory review.
