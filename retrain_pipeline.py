"""
retrain_pipeline.py
────────────────────
Run this ONCE on your local machine to retrain and resave
the pipeline with your current scikit-learn version.

Run with: python retrain_pipeline.py
"""

import pandas as pd
import numpy as np
import joblib
import sklearn

print(f"scikit-learn version: {sklearn.__version__}")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# ── Load dataset ─────────────────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── Fix known data issues ────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.drop("customerID", axis=1, inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ── Split features and target ────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols   = X.select_dtypes(include=np.number).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── Build pipeline ───────────────────────────────────────────
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

pipeline_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(random_state=42))
])

# ── GridSearchCV ─────────────────────────────────────────────
print("\nRunning GridSearchCV (this takes a few minutes)...")

param_grid = {
    "classifier__n_estimators":      [100, 200],
    "classifier__max_depth":         [None, 10, 20],
    "classifier__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    pipeline_rf, param_grid,
    cv=5, scoring="f1", n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest params : {grid_search.best_params_}")
print(f"Best CV F1  : {grid_search.best_score_:.4f}")

# ── Save pipeline ────────────────────────────────────────────
best_rf = grid_search.best_estimator_
joblib.dump(best_rf, "pipeline_model.joblib")
print("\nPipeline saved as pipeline_model.joblib")

# ── Quick sanity check ───────────────────────────────────────
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

y_pred  = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print(f"\nTest Set Results:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nDone. Now run: python test_pipeline.py")
