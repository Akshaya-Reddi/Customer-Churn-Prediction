"""
train_model.py
------------------------------------------------------------------
Trains the FINAL Customer Churn model (Gradient Boosting Classifier,
selected as the best model in the notebook) and exports it as a
single pickle file that can accept RAW customer data.

Why a Pipeline?
    In the notebook, encoding + scaling were done as separate manual
    steps on the whole dataframe. That is hard to reproduce for a
    single new customer inside a web app. Here we bundle
        (encoding + scaling + Gradient Boosting)
    into ONE scikit-learn Pipeline. The saved pickle therefore takes
    raw values (e.g. gender="Female", Contract="Two year") directly,
    so the Streamlit UI does not have to redo any preprocessing.

Outputs:
    churn_model.pkl   -> the trained Pipeline (preprocessing + model)
    model_meta.pkl    -> metadata for the UI (dropdown options, ranges,
                         evaluation metrics, feature importances)
------------------------------------------------------------------
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ------------------------------------------------------------------
# 1. Load the raw dataset
# ------------------------------------------------------------------
DATA_PATH = "Telco_Customer_Churn.xlsx"
df = pd.read_excel(DATA_PATH)
print(f"[1] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ------------------------------------------------------------------
# 2. Clean the data (same decisions as the notebook)
# ------------------------------------------------------------------
# 'TotalCharges' has blank strings for brand-new customers (tenure = 0),
# which makes pandas read it as text. Convert to numeric; blanks -> NaN.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill the few missing TotalCharges with the median (robust to outliers).
median_total = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

# 'customerID' is just an identifier -> not useful for prediction.
df = df.drop(columns=["customerID"])

# ------------------------------------------------------------------
# 3. Define features (X) and target (y)
# ------------------------------------------------------------------
# Target: Churn (Yes/No) -> 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

y = df["Churn"]
X = df.drop(columns=["Churn"])

# Continuous numeric columns -> will be standard-scaled.
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# Everything else is categorical (SeniorCitizen 0/1 is treated as a category too).
categorical_features = [c for c in X.columns if c not in numeric_features]

print(f"[3] Numeric features   : {numeric_features}")
print(f"    Categorical features: {categorical_features}")

# ------------------------------------------------------------------
# 4. Build the preprocessing + model Pipeline
# ------------------------------------------------------------------
# ColumnTransformer applies the right transformation to each column group:
#   - numeric  -> StandardScaler (center to mean 0, unit variance)
#   - category -> OneHotEncoder  (handle_unknown='ignore' keeps the app
#                                 from crashing on an unseen category)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Final estimator = the notebook's BEST model with the SAME hyperparameters.
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)

# Bundle preprocessing + model so the pickle accepts raw input.
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", model),
])

# ------------------------------------------------------------------
# 5. Train / test split and fit
# ------------------------------------------------------------------
# stratify=y keeps the same churn ratio in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
print("[5] Pipeline trained successfully.")

# ------------------------------------------------------------------
# 6. Evaluate on the held-out test set
# ------------------------------------------------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of churn

metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall":    recall_score(y_test, y_pred),
    "F1-Score":  f1_score(y_test, y_pred),
    "ROC-AUC":   roc_auc_score(y_test, y_proba),
}

print("\n[6] Test-set performance (Gradient Boosting):")
for name, value in metrics.items():
    print(f"    {name:10s}: {value:.4f}")

# ------------------------------------------------------------------
# 7. Feature importances (mapped back to readable names)
# ------------------------------------------------------------------
# Get the expanded feature names produced by the ColumnTransformer.
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
importances = pipeline.named_steps["classifier"].feature_importances_

# Keep the top 15 most influential features for the dashboard.
importance_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)
# Tidy the names (drop the 'num__' / 'cat__' prefixes) for display.
importance_df["feature"] = (
    importance_df["feature"].str.replace("num__", "", regex=False)
                            .str.replace("cat__", "", regex=False)
)

# ------------------------------------------------------------------
# 8. Build metadata the Streamlit UI needs (dropdown options + ranges)
# ------------------------------------------------------------------
# For every categorical field, list its valid options for the UI dropdowns.
categorical_options = {
    col: sorted([str(v) for v in X[col].dropna().unique()])
    for col in categorical_features
}

# For numeric fields, capture sensible min / max / default for the sliders.
numeric_ranges = {
    col: {
        "min": float(X[col].min()),
        "max": float(X[col].max()),
        "mean": float(X[col].mean()),
    }
    for col in numeric_features
}

meta = {
    "model_name": "Gradient Boosting Classifier",
    "feature_order": list(X.columns),        # exact column order the model expects
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "categorical_options": categorical_options,
    "numeric_ranges": numeric_ranges,
    "metrics": metrics,
    "importances": importance_df.to_dict(orient="records"),
    "n_samples": int(df.shape[0]),
    "churn_rate": float(y.mean()),
}

# ------------------------------------------------------------------
# 9. Save the model and metadata with pickle
# ------------------------------------------------------------------
with open("churn_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

with open("model_meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("\n[9] Saved -> churn_model.pkl  (trained pipeline)")
print("    Saved -> model_meta.pkl   (UI metadata)")
print("\nDone. You can now run the Streamlit app:  streamlit run app.py")
