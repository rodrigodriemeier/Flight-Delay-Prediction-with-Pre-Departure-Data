# Flight Delay Prediction with Pre-Departure Data

## Project Summary
This project investigates how well flight departure delays (≥15 minutes) can be predicted using only information available before takeoff.  
Multiple models and feature representations were tested, showing that performance is mainly limited by the available information rather than model choice.
Data: https://www.kaggle.com/code/kozlowladimir/flight-delay

---

## Problem Definition

- **Objective**: Binary classification of flight departure delays (delay ≥ 15 minutes).
- **Target variable**: `dep_delayed_15min` (0/1).
- **Constraint**: Use only features available before departure.
- **Primary metric**: Recall (false negatives are more costly than false positives).
- **Secondary metrics**: Precision, Precision–Recall (PR) curve, Average Precision (AP).

---

## Dataset

- Public flight delay dataset.
- Two CSV files provided:
  - `flight_delays_train.csv` (labeled)
  - `flight_delays_test.csv` (unlabeled)
- Model performance is evaluated exclusively on the labeled dataset using an internal hold-out split.
- The external test set is used only for final inference, not for metric computation.

---

## Data Cleaning

The following cleaning steps were applied once and saved into cleaned CSV files:

- Removed `"c-"` prefix from `Month`, `DayofMonth`, and `DayOfWeek`, converting them to integers.
- Converted `DepTime` from HHMM format to minutes since midnight.
- Converted `Distance` to numeric type.
- Standardized categorical strings (`UniqueCarrier`, `Origin`, `Dest`) using `strip()` and `upper()`.
- Removed duplicate rows.
- Converted target variable from `Y/N` to `1/0`.
- Performed basic sanity checks on ranges (month, weekday, distance).

Generated files:
- `flight_delays_train_cleaned.csv`
- `flight_delays_test_cleaned.csv`

---

## Feature Engineering

Only deterministic features available before departure were used.

### Time-Related Features
- **Departure time**: discretized into coarse time-of-day bins  
  (`0–6`, `6–10`, `10–14`, `14–17`, `17–20`, `20–24`)
- **Month**: encoded using cyclic representation (`sin_month`, `cos_month`)
- **Day of week**: encoded using cyclic representation (`sin_dow`, `cos_dow`)

### Distance
- Log-transformed using `log1p` to reduce heavy-tail effects and stabilize tree splits.

### Categorical Features
- `UniqueCarrier`
- `Origin`
- `Dest`
- `DayofMonth`

---

## Evaluation Protocol

- Stratified train/validation split (80% / 20%).
- All metrics (recall, precision, PR curve, AP) are computed on the validation set only.
- Threshold selection is performed using validation data under a recall constraint.
- The external test dataset has no labels and is therefore not used for evaluation.

---

## Baseline

- **DummyClassifier** (most frequent class)
- Recall: **0.0**
- Confirms strong class imbalance and the need for recall-oriented modeling.

---

## Models Tested

### Logistic Regression
- One-Hot Encoding + StandardScaler
- `class_weight="balanced"`
- Recall @ 0.5 ≈ 0.64–0.67
- AP ≈ 0.32–0.34
- Precision collapses rapidly at high recall.

**Conclusion**: Linear separation is insufficient.

---

### Random Forest
- One-Hot Encoding (no scaling)
- Slight improvement in AP (~0.36).
- Recall–precision trade-off remains poor.

**Conclusion**: Non-linearities exist, but signal is weak.

---

### Gradient Boosting (sklearn)
- Increased model expressiveness.
- AP ≈ 0.38.
- High computational cost due to high-cardinality categorical features.

**Conclusion**: Only marginal gains over simpler models.

---

### HistGradientBoostingClassifier (Final Model)
- More efficient boosting implementation.
- Category grouping (`min_frequency`, `max_categories`) used to control dimensionality.
- AP ≈ 0.39–0.40.
- Precision still collapses at high recall levels.

**Conclusion**: Model capacity is no longer the bottleneck.

---

## Threshold Tuning

- Decision thresholds scanned from 0.01 to 0.99.
- Selected threshold maximizes precision subject to recall ≥ 0.80.
- Tuning performed exclusively on the validation set.

---

## Final Inference on External Test Set

- The final trained pipeline is applied to `flight_delays_test_cleaned.csv`.
- Output probabilities and predicted labels are saved to `test_predictions.csv`.
- No recall or precision is computed on this dataset, as ground truth is unavailable.
- This step is included to simulate a deployment-style inference pipeline.

---

## Considered but Not Implemented: Geographic Seasonality

An additional idea considered during the project was to enrich the dataset with geographic information about airports (e.g., latitude and longitude) using an external airport-location dataset.  
This would allow defining seasons based on both calendar date and geographic location (hemisphere), potentially serving as a proxy for weather-related effects, which are known to be a major driver of flight delays.

This approach was not implemented for two reasons:
- It requires external data enrichment beyond the provided dataset.
- Even with correct season definitions, seasonality alone would still be a weak proxy compared to real meteorological variables (e.g., wind, snow, storms), which are not available.

Nevertheless, this remains a plausible extension for future work.

---

## Key Findings

- Increasing model complexity leads to diminishing returns.
- Precision–Recall curves consistently collapse at high recall.
- Calendar-derived seasonality does not add meaningful signal beyond cyclic time encoding.
- The dominant limitation is the lack of informative pre-departure features, such as weather or congestion data.

---

## Conclusion

Even with modern gradient boosting models, predicting flight delays using only pre-departure information remains inherently difficult.  
The primary limitation of this task is informational rather than algorithmic.

---

## Repository Structure

data/
├── flight_delays_train.csv
├── flight_delays_test.csv
├── flight_delays_train_cleaned.csv
└── flight_delays_test_cleaned.csv

results/
└── test_predictions.csv

src/
├── config.py # global configuration
├── features.py # feature engineering
├── preprocess.py # encoding pipeline
├── train.py # model training
└── evaluate.py # validation and final inference



AUTHOR Rodrigo Driemeier dos Santos EESC - University of São Paulo (USP), São Carlos, Brazil — Mechatronics Engineering École Centrale de Lille, France — Generalist Engineering rodrigodriemeier@usp.br https://www.linkedin.com/in/rodrigo-driemeier-dos-santos-a7698633b/

