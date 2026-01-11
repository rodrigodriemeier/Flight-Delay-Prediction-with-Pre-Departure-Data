# config.py
# Central place for global configuration and design decisions.

RANDOM_STATE = 0
TEST_SIZE = 0.2

TARGET = "dep_delayed_15min"

NUMERIC_FEATURES = [
    "sin_month",
    "cos_month",
    "sin_dow",
    "cos_dow",
    "Distance"
]

CATEGORICAL_FEATURES = [
    "DepTime",
    "DayofMonth",
    "UniqueCarrier",
    "Origin",
    "Dest",
    "Season"
]

# Operational constraint: we prioritize recall
MIN_RECALL = 0.80
