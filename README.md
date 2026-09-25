# Telecom Customer Churn Risk Analysis

An end-to-end analysis and Streamlit app for identifying telecom customers with a higher likelihood of churn. The project includes exploratory analysis, model comparison, threshold selection, interpretation, and a serialized model pipeline for predictions.

## Project files

| File | Purpose |
|---|---|
| `Customer_Churn_01.ipynb` | Ordered analysis with a Markdown summary after each section |
| `app.py` | Streamlit interface for churn-risk predictions |
| `churn_model.pkl` | Fitted preprocessing and logistic-regression pipeline |
| `requirements.txt` | Dependencies for the notebook and app |

## Dataset

The notebook expects the Telco Customer Churn dataset as `Customer_Churn.csv`. It checks these locations in order:

1. The project folder, next to the notebook.
2. The adjacent `DATASET` folder (`../../DATASET/Customer_Churn.csv` when run from the project root).

The CSV is not included in this repository. The target is `Churn` (`Yes` or `No`); `customerID` is excluded from model features. Blank `TotalCharges` values are converted to numeric and filled with zero.

## Workflow

The notebook follows this sequence, with a Markdown section summary after every section:

1. Load and inspect the data.
2. Clean `TotalCharges` and validate the target.
3. Explore churn prevalence, contracts, tenure, charges, internet service, and payment method.
4. Create a stratified holdout split and define reusable preprocessing.
5. Compare logistic regression, decision tree, and random forest.
6. Compare probability thresholds and select 0.4 to prioritize churn recall.
7. Review the confusion matrix, false negatives, and logistic-regression coefficients.
8. Export the fitted pipeline for Streamlit.
9. Report final holdout metrics and summarize the work.

Preprocessing and estimation are kept in scikit-learn pipelines. Numerical charges and tenure are standardized; categorical variables are one-hot encoded with unknown categories ignored.

## Model results

The recorded stratified 80/20 holdout run selected logistic regression at a churn-probability threshold of 0.4:

| Metric | Result |
|---|---:|
| Accuracy | 77.7% |
| Precision | 56.8% |
| Recall | 66.8% |
| True churners identified | 250 of 374 |
| False negatives | 124 |
| False positives | 190 |

At the default threshold of 0.5, recorded recall was 55.9% and precision was 65.7%. The 0.4 threshold finds more churners while increasing false positives. Metrics are recomputed in the notebook and can vary with data or library versions. They describe predictive performance, not causal effects.

## Run locally

Create and activate a virtual environment if desired, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the notebook from top to bottom in Jupyter. Place the CSV in one of the supported locations first. Running the model-export section refreshes `churn_model.pkl`.

Start the Streamlit app from the project folder:

```bash
python -m streamlit run app.py
```

The app uses the 0.4 decision threshold. Its risk labels are Low below 0.4, Medium from 0.4 to below 0.7, and High at or above 0.7.

## Dependencies and implementation

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, and Streamlit.
The pandas, NumPy, scikit-learn, and joblib versions are pinned in `requirements.txt` to match the serialized model artifact.

## Live demo

[Open the Streamlit demo](https://customer-churn-prediction-yash1912.streamlit.app/)

## Author

Yaswanth — AI & Data Science Student
