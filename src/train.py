# train.py
# Trains the final model using the predefined pipeline.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from config import (
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from features import build_features
from preprocess import make_preprocessor


def train_model(data_path="flight_delays_train_cleaned.csv"):
    """
    Train the final model and return fitted pipeline
    along with validation split.
    """

    # Load cleaned data
    df = pd.read_csv(data_path)

    # Feature engineering
    df = build_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.05,
                    max_depth=6,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model, X_val, y_val


if __name__ == "__main__":
    train_model()
