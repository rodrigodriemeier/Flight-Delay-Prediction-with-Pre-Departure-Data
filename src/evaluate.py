# evaluate.py
# Validation evaluation + final test prediction
# This script is the FINAL entry point of the project.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    recall_score,
    precision_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

from train import train_model
from features import build_features
from config import (
    MIN_RECALL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# ============================================================
# VALIDATION EVALUATION
# ============================================================

model, X_val, y_val = train_model()

proba_val = model.predict_proba(X_val)[:, 1]

# Default threshold
y_default = (proba_val >= 0.5).astype(int)

print("=== VALIDATION (threshold = 0.5) ===")
print("Recall:", recall_score(y_val, y_default))
print("Precision:", precision_score(y_val, y_default))
print("Confusion matrix:\n", confusion_matrix(y_val, y_default))

# ------------------------------------------------------------
# Threshold tuning (recall-constrained)
# ------------------------------------------------------------

thresholds = np.arange(0.01, 1.00, 0.01)
rows = []

for t in thresholds:
    y_hat = (proba_val >= t).astype(int)
    rows.append({
        "threshold": t,
        "recall": recall_score(y_val, y_hat),
        "precision": precision_score(y_val, y_hat, zero_division=0),
    })

results = pd.DataFrame(rows)
candidates = results[results["recall"] >= MIN_RECALL]

best = candidates.sort_values("precision", ascending=False).iloc[0]
BEST_THRESHOLD = best["threshold"]

print("\n=== CHOSEN THRESHOLD (validation) ===")
print(best)

# ------------------------------------------------------------
# PR Curve
# ------------------------------------------------------------

precision, recall, _ = precision_recall_curve(y_val, proba_val)
ap = average_precision_score(y_val, proba_val)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.axhline(y=y_val.mean(), linestyle="--", label="Baseline")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Validation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# FINAL TEST PREDICTION (NO LABELS)
# ============================================================

print("\n=== RUNNING FINAL TEST PREDICTION ===")

test = pd.read_csv("flight_delays_test_cleaned.csv")
test = build_features(test)

X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= BEST_THRESHOLD).astype(int)

out = test.copy()
out["pred_proba"] = test_proba
out["pred_label"] = test_pred

out.to_csv("test_predictions.csv", index=False)

print("Saved: test_predictions.csv")
print("Positive rate on test:", out["pred_label"].mean())
