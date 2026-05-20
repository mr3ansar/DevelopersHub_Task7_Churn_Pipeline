# Task 2: End-to-End ML Pipeline — Customer Churn Prediction

## Objective
Build a reusable, production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The pipeline covers data preprocessing, model training, hyperparameter tuning, and export using `joblib`.

---

## Dataset
**Telco Customer Churn Dataset**
- Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 7,043 rows × 21 columns
- Target column: `Churn` (Yes / No)

---

## Methodology / Approach

### 1. Exploratory Data Analysis
- Inspected shape, data types, missing values
- Visualized class distribution of the target variable (`Churn`)
- Fixed `TotalCharges` column (stored as string with whitespace)

### 2. Preprocessing Pipeline
- **Numerical features**: Median imputation → Standard Scaling
- **Categorical features**: Mode imputation → One-Hot Encoding
- Used `ColumnTransformer` to apply transformations per column type
- Used `Pipeline` to chain preprocessing with the classifier

### 3. Models Trained
| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble model, tuned with GridSearchCV |

### 4. Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation
- Optimized for **F1-score** (better than accuracy for imbalanced data)
- Parameters searched for Random Forest:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]

### 5. Evaluation Metrics
- Accuracy, F1-Score, ROC-AUC
- Confusion Matrix
- ROC Curve

### 6. Model Export
- Best pipeline saved using `joblib` as `pipeline_model.joblib`
- Verified by reloading and running predictions

---

## Key Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~0.80 | ~0.58 | ~0.84 |
| Random Forest (Tuned) | ~0.80 | ~0.59 | ~0.85 |

---

## Key Observations
- **Contract type** is the strongest predictor of churn — month-to-month customers churn at a significantly higher rate
- **Tenure** is inversely related to churn — newer customers are more likely to leave
- **Random Forest** marginally outperforms Logistic Regression on F1 and ROC-AUC after tuning
- The exported pipeline is fully production-ready — load it in 2 lines and predict on new data

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/developershub-aiml-internship.git
cd task2-churn-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in the same folder
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 4. Launch notebook
jupyter notebook churn_pipeline.ipynb
```

### Load the Saved Pipeline (Production Use)
```python
import joblib
pipeline = joblib.load('pipeline_model.joblib')
predictions = pipeline.predict(new_data)
```

---

## Project Structure
```
task2-churn-pipeline/
│
├── churn_pipeline.ipynb       ← Main notebook
├── pipeline_model.joblib      ← Saved trained pipeline
├── requirements.txt           ← Python dependencies
└── README.md                  ← Project documentation
```

---

## Skills Demonstrated
- ML pipeline construction with `scikit-learn` Pipeline API
- Hyperparameter tuning with `GridSearchCV`
- Model export and reusability with `joblib`
- Production-readiness practices
