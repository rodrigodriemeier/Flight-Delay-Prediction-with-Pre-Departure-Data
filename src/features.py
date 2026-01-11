# features.py
# Feature engineering logic.
# Input: cleaned dataframe
# Output: dataframe with engineered features

import numpy as np
import pandas as pd


def add_departure_time_bins(df):
    """
    Convert departure time (minutes since midnight)
    into coarse time-of-day categories.
    """
    bins = [0, 360, 600, 840, 1020, 1200, 1440]
    labels = ["0-6", "6-10", "10-14", "14-17", "17-20", "20-24"]

    df["DepTime"] = pd.cut(
        df["DepTime"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    ).astype(str)

    return df


def add_cyclic_calendar_features(df):
    """
    Encode month and day of week as cyclic variables.
    This avoids artificial discontinuities (e.g. Dec → Jan).
    """
    theta_month = 2 * np.pi * df["Month"] / 12
    df["sin_month"] = np.sin(theta_month)
    df["cos_month"] = np.cos(theta_month)

    theta_dow = 2 * np.pi * df["DayOfWeek"] / 7
    df["sin_dow"] = np.sin(theta_dow)
    df["cos_dow"] = np.cos(theta_dow)

    return df


def add_distance_transform(df):
    """
    Log-transform distance to reduce heavy tail
    and stabilize splits in tree-based models.
    """
    df["Distance"] = np.log1p(df["Distance"])
    return df


def add_season(df):
    """
    Add season labels based on calendar month.
    Assumes Northern Hemisphere (US domestic flights).
    """

    def month_to_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df["Season"] = df["Month"].apply(month_to_season)
    return df


def build_features(df):
    """
    Main feature engineering entry point.
    All transformations are deterministic and reproducible.
    """
    df = add_departure_time_bins(df)
    df = add_cyclic_calendar_features(df)
    df = add_distance_transform(df)
    df = add_season(df)
    return df
