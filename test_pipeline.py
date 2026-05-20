"""
test_pipeline.py
────────────────
Test the saved churn pipeline on raw, unseen customer data.
Run with: python test_pipeline.py
"""

import joblib
import pandas as pd

# ── Load the saved pipeline ──────────────────────────────────
pipeline = joblib.load("pipeline_model.joblib")
print("Pipeline loaded successfully.\n")

# ── Raw test customers (exactly as they would come in production)
# No preprocessing needed — the pipeline handles everything internally
raw_data = pd.DataFrame({
    "gender":           ["Female", "Male",   "Male",   "Female"],
    "SeniorCitizen":    [0,         1,        0,        1       ],
    "Partner":          ["Yes",     "No",     "Yes",    "No"    ],
    "Dependents":       ["No",      "No",     "Yes",    "No"    ],
    "tenure":           [1,         60,       24,       3       ],
    "PhoneService":     ["No",      "Yes",    "Yes",    "Yes"   ],
    "MultipleLines":    ["No phone service", "No", "Yes", "Yes" ],
    "InternetService":  ["DSL",     "Fiber optic", "DSL", "Fiber optic"],
    "OnlineSecurity":   ["No",      "Yes",    "Yes",    "No"    ],
    "OnlineBackup":     ["Yes",     "No",     "No",     "No"    ],
    "DeviceProtection": ["No",      "Yes",    "Yes",    "No"    ],
    "TechSupport":      ["No",      "Yes",    "No",     "No"    ],
    "StreamingTV":      ["No",      "Yes",    "No",     "Yes"   ],
    "StreamingMovies":  ["No",      "Yes",    "No",     "Yes"   ],
    "Contract":         ["Month-to-month", "Two year", "One year", "Month-to-month"],
    "PaperlessBilling": ["Yes",     "No",     "No",     "Yes"   ],
    "PaymentMethod":    ["Electronic check", "Bank transfer (automatic)",
                         "Credit card (automatic)", "Electronic check"],
    "MonthlyCharges":   [29.85,     65.60,    53.45,    95.75   ],
    "TotalCharges":     [29.85,     3873.20,  1306.92,  270.45  ],
})

# ── Run predictions ──────────────────────────────────────────
predictions  = pipeline.predict(raw_data)
probabilities = pipeline.predict_proba(raw_data)[:, 1]  # probability of churn

# ── Display results ──────────────────────────────────────────
print("=" * 58)
print(f"  {'Customer':<12} {'Tenure':>8} {'Contract':<20} {'Result':<12} {'Churn %'}")
print("=" * 58)

contracts = raw_data["Contract"].tolist()
tenures   = raw_data["tenure"].tolist()

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label  = "⚠ CHURN"    if pred == 1 else "✓ STAY"
    pct    = f"{prob*100:.1f}%"
    print(f"  Customer {i+1:<4}  {tenures[i]:>6}mo   {contracts[i]:<20} {label:<12} {pct}")

print("=" * 58)

# ── Business interpretation ──────────────────────────────────
print("""
Interpretation:
  ⚠ CHURN  → High churn risk. Consider a retention offer.
  ✓ STAY   → Low churn risk. No immediate action needed.

Note: Probabilities above 50% = predicted churn.
A lower threshold (e.g. 30%) can be used if the business
wants to catch more at-risk customers early.
""")
