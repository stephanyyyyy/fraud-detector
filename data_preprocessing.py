# data preprocessing steps before training in Spark

import pandas as pd

# read dataset csv file
df = pd.read_csv("financial_fraud_detection_dataset.csv")

# drop noninformative columns
df = df.drop(columns=["time_since_last_transaction", "spending_deviation_score", "geo_anomaly_score", "velocity_score", "receiver_account", "payment_channel", "ip_address", "device_hash", "fraud_type", "transaction_id"])

# Parse "timestamp" feature in csv and extract time-based features
df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed')

# create hour_of_day for the hour of each day (extracts hour in military time)
df["hour_of_day"] = df["timestamp"].dt.hour

# create day_of_week (1=Monday,...,7=Sunday)
df["day_of_week"] = df["timestamp"].dt.dayofweek + 1

# create is_weekend using day_of_week to determine which transactions were on Sat./Sun. and 
# label them 1 if Sat./Sun., else 0)
df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)

# 4. Save to new CSV
df.to_csv("modified_fraud_dataset_final.csv", index=False)
print("Saved new dataet.")
