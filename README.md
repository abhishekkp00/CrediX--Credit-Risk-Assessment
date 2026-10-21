# CrediX - Credit Risk Assessment

CrediX is a credit risk assessment application that predicts whether a loan applicant is likely to default based on their financial and borrower profile. The project combines a machine learning model with a lightweight FastAPI backend and a clean frontend for real-time risk scoring.

## Project Overview

This application allows a user to enter applicant and loan details through a web form, sends the data to a backend API, and receives:

- predicted default probability
- risk classification
- decision threshold
- final verdict (Low Risk / High Risk)

The project is designed as a practical ML + web app demo for loan underwriting and risk evaluation.

## Objective

The main goal is to build a usable credit risk model that can estimate the probability of default and help in understanding whether a loan application is financially safe.

## Tech Stack

- Python 3.10+
- FastAPI for the API layer
- Uvicorn for serving the API
- Pandas for data handling
- scikit-learn for preprocessing and modeling
- XGBoost for model training
- Joblib for saving/loading ML artifacts
- HTML, CSS, and JavaScript for the frontend UI

## Project Structure

```text
CrediX - Credit Risk Assessment/
├── app.py                     # FastAPI backend and prediction endpoint
├── Credit_Risk.ipynb         # Notebook with data analysis, modeling, and experimentation
├── credit_risk_dataset.csv   # Loan dataset used for model training and evaluation
├── credit_risk_model.pkl     # Trained model artifact
├── best_threshold.pkl         # Optimal decision threshold for classification
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignored local/generated files
├── static/
│   ├── index.html            # Frontend form UI
│   ├── script.js             # Frontend logic to call the API
│   └── style.css             # UI styling
└── README.md                 # Project documentation
```

## Model and Prediction Flow

The backend loads the trained model and threshold during application startup and exposes a prediction API at `/predict`.

### Input fields

The app expects the following borrower and loan details:

- `person_age`
- `person_income`
- `person_home_ownership`
- `person_emp_length`
- `loan_intent`
- `loan_grade`
- `loan_amnt`
- `loan_int_rate`
- `loan_percent_income`
- `cb_person_default_on_file`
- `cb_person_cred_hist_length`

### Output

The backend returns:

- `default_probability`: probability of default
- `default_prediction`: binary prediction based on threshold
- `threshold`: selected threshold value
- `Result`: `High Risk` or `Low Risk`

## Machine Learning Pipeline

The notebook (`Credit_Risk.ipynb`) demonstrates a full data science workflow, including:

- dataset loading
- exploratory data analysis (EDA)
- missing value checks
- target distribution analysis
- outlier handling
- feature engineering and preprocessing
- train-test splitting
- class imbalance handling
- model training and evaluation
- threshold optimization
- model serialization using Joblib

The training process uses preprocessing pipelines for numeric and categorical features, then trains a model suitable for binary classification of loan default risk.

## Data Description

The project uses a credit risk dataset containing borrower features and a target variable indicating whether the applicant defaulted or repaid.

Typical fields in the dataset include:

- personal and financial profile
- employment length
- home ownership status
- loan intent and grade
- loan amount and interest rate
- prior default history
- credit history length

## Frontend

The frontend is served from the `static` directory and gives the user a form-based interface for entering loan application information.

It includes:

- a clean underwriting-style UI
- validation for required fields
- API call to the backend
- risk gauge and verdict display

## API Usage

### Start the API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The app will run locally at:

- `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

### Predict endpoint

```http
POST /predict
```

Request body example:

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

Example response:

```json
{
  "default_probability": 0.3124,
  "default_prediction": 0,
  "threshold": 0.38,
  "Result": "Low Risk"
}
```

## Installation

1. Clone the repository.
2. Navigate to the project folder.
3. Create a virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
uvicorn app:app --reload
```

## Notes

- The project is intended as a demo and can be extended with more advanced features such as:
  - model monitoring
  - user authentication
  - database storage of predictions
  - improved explainability
  - deployment to cloud platforms
- The model files (`.pkl`) are generated artifacts and may be re-created by retraining the notebook if needed.

## License

This project is for educational and demonstration purposes.

## Summary

CrediX demonstrates a complete pipeline for credit risk assessment, from raw data exploration and model training to deployment as a web application. It combines data science, backend APIs, and frontend design into a practical loan risk scoring solution.
