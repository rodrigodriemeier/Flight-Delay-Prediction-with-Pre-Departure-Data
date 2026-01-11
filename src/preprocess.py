# preprocess.py
# Defines how features are converted into model-ready matrices.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_preprocessor():
    """
    ColumnTransformer for numeric + categorical features.
    HistGradientBoosting requires dense input.
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
