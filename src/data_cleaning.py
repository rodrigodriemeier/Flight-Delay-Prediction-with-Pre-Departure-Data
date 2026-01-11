import pandas as pd

# Opening csvs

df_test = pd.read_csv("flight_delays_test.csv")
df_train = pd.read_csv("flight_delays_train.csv")

# DATA CLEANING

# First, let's remove the 'c-' from Month, DayofMonth and DayOfWeek columns, and turn them to 'int'

for col in ["Month", "DayofMonth", "DayOfWeek"]:
    df_test[col] = (
        df_test[col]
        .astype(str)
        .str.replace("c-", "", regex=False)
        .astype(int)
    )
    df_train[col] = (
        df_train[col]
        .astype(str)
        .str.replace("c-", "", regex=False)
        .astype(int)
    )

# We also need to convert DepTime (total minutes in the day) and Distance (to float)

print(df_train)

df_test["Distance"] = df_test["Distance"].astype("Float64")
df_train["Distance"] = df_train["Distance"].astype("Float64")

hours = df_train["DepTime"].astype("Float64")//100
minutes = df_train["DepTime"].astype("Float64")-hours*100
df_train["DepTime"] = 60*hours + minutes

hours = df_test["DepTime"].astype("Float64")//100
minutes = df_test["DepTime"].astype("Float64")-hours*100
df_test["DepTime"] = 60*hours + minutes

# Now, we standardize UniqueCarrier, Origin and Dest using strip() and upper()

for col in ["UniqueCarrier", "Origin", "Dest"]:
    df_test[col] = df_test[col].str.strip().str.upper()
    df_train[col] = df_train[col].str.strip().str.upper()

# Removing identical lines

df_test = df_test.drop_duplicates()
df_train = df_train.drop_duplicates()

# For the train csv it's ideal to convert Y/S to 1/0 (binary)

df_train["dep_delayed_15min"] = df_train["dep_delayed_15min"].str.replace("Y", "1").str.replace("N", "0").astype("Int64")

# DATA RELIABILITY

# Simple Sanity Check (This data base is well known and used, it's alredy known for its reliability - no reason to convert to pd.nan)

# Checking Month
#print(df_train[(df_train["Month"] < 0) | (df_train["Month"] > 12)]) #OUTPUT: Empty DataFrame
#print(df_test[(df_test["Month"] < 0) | (df_test["Month"] > 12)]) #OUTPUT: Empty DataFrame

# Checking DayofMonth
#print(df_train[(df_train["DayofMonth"] < 0) | (df_train["DayofMonth"] > 31)]) #OUTPUT: Empty DataFrame
#print(df_test[(df_test["DayofMonth"] < 0) | (df_test["DayofMonth"] > 31)]) #OUTPUT: Empty DataFrame

# Checking DayOfWeek
#print(df_train[(df_train["DayOfWeek"] < 1) | (df_train["DayOfWeek"] > 7)]) #OUTPUT: Empty DataFrame
#print(df_test[(df_test["DayOfWeek"] < 1) | (df_test["DayOfWeek"] > 7)]) #OUTPUT: Empty DataFrame

# Checking Distance and visualizing its max value (reasonable)
#print(df_train[df_train["Distance"] < 0]) #OUTPUT: Empty DataFrame
#print(f"Max value - Distance - Train: {df_train['Distance'].max()}") #OUTPUT: 4962 - reasonable
#print(df_test[df_test["Distance"] < 0]) #OUTPUT: Empty DataFrame
#print(f"Max value - Distance - Test: {df_test['Distance'].max()}") #OUTPUT: 4962 - reasonable

# Exporting both train and test csvs cleaned and verified

df_train.to_csv("flight_delays_train_cleaned.csv", index=False)
df_test.to_csv("flight_delays_test_cleaned.csv", index=False)

print(df_train)