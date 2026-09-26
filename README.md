# Telecom Customer Churn Risk Analysis

An end-to-end learning project for identifying telecom customers with a higher likelihood of churn. The notebook compares logistic regression, a decision tree, and a random forest, tunes their parameters, checks probability quality, and retains LIME and SHAP explanations. The Streamlit app uses the saved pipeline to score customers.

## Project files

| File | Purpose |
|---|---|
| `Customer_Churn_01.ipynb` | Ordered analysis, model selection, evaluation, and explainability |
| `app.py` | Streamlit interface for churn-risk predictions |
| `churn_model.pkl` | Fitted preprocessing and model pipeline used by the app |
| `requirements.txt` | Notebook and app dependencies |
| `LICENSE` | MIT license |

## Dataset

Place `Customer_Churn.csv` in the project folder or the adjacent `DATASET` folder. The CSV is intentionally not included in this repository. The notebook converts blank `TotalCharges` values to zero and excludes `customerID` from the predictors.

Dataset credit: [IBM Telco Customer Churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d), also distributed on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

## Workflow

1. Load and validate the data and clean `TotalCharges`.
2. Explore churn prevalence and patterns in existing customer attributes.
3. Create a stratified train/test holdout and leakage-safe preprocessing pipelines.
4. Compare default logistic regression, decision tree, and random forest with five-fold stratified cross-validation on training data.
5. Tune model-specific parameters with `GridSearchCV` using average precision, then compare default and tuned holdout performance.
6. Select the model with the strongest mean training-fold average precision. Inspect the full out-of-fold precision-recall curve and the 0.4 threshold trade-off.
7. Check calibration on training-only out-of-fold predictions; apply sigmoid calibration when the ten-bin calibration gap exceeds 0.05.
8. Report holdout ROC-AUC, average precision, ROC and precision-recall curves, a reliability diagram, and false-negative rates by contract and internet service.
9. Explain predictions with LIME and SHAP and export the selected fitted pipeline for Streamlit.

Numerical features are standardized, categorical values are one-hot encoded, and unseen categories are ignored. Model selection and threshold/calibration decisions use training data; the test set is reserved for final evaluation.

## Model results

Measured with the adjacent IBM/Kaggle CSV, a stratified 80/20 split (`random_state=42`), and the pinned dependencies. The selected model was logistic regression (`C=1`); its default and tuned parameter settings matched.

| Model | Mean 5-fold training PR-AUC | Tuned holdout ROC-AUC | Tuned holdout PR-AUC |
|---|---:|---:|---:|
| Logistic regression | 0.661 | 0.842 | 0.634 |
| Decision tree | 0.615 | 0.827 | 0.621 |
| Random forest | 0.661 | 0.843 | 0.659 |

Tuning maximizes average precision on the training folds. On the holdout at threshold 0.5, tuned logistic regression scored 80.6% accuracy, 65.7% precision, and 55.9% recall; the default model had the same scores because the best `C` remained 1. At the selected 0.4 cutoff, holdout accuracy was 77.7%, precision was 56.8%, and recall was 66.8% (250 true positives, 124 false negatives, and 190 false positives). Holdout ROC-AUC was 0.842 and average precision was 0.634.

The training-only out-of-fold calibration gap was 0.011, below the 0.05 threshold, so calibration was not applied. Results depend on the input data and software versions and are recomputed by running the notebook.

## Explainable AI

LIME provides a local explanation for a held-out customer. SHAP summarizes feature contributions across a test subset and for an individual prediction. Both explainers operate on the selected model's transformed feature space.

## Run locally

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run `Customer_Churn_01.ipynb` from top to bottom after placing the CSV in a supported location. The notebook updates `churn_model.pkl` only if the selected model family or estimator parameters differ from the existing artifact.

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

The app applies the 0.4 churn cutoff. Its risk labels are Low below 0.4, Medium from 0.4 to below 0.7, and High at or above 0.7.

## Dependencies

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, Streamlit, LIME, and SHAP. Several versions are pinned in `requirements.txt` to support the serialized model and explainability APIs.

## Live demo

[Open the Streamlit demo](https://customer-churn-prediction-yash1912.streamlit.app/)

## Author

Yaswanth — AI & Data Science Student
