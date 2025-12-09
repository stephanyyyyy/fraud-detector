from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

# Where to save the trained GBT pipeline model (inside Spark container)
MODEL_PATH = "/opt/spark/work-dir/fraud_gbt_model"

# Start Spark
spark = (
    SparkSession.builder
    .appName("Fraud_GBT_Weighted")
    .config("spark.sql.shuffle.partitions", "8")  # can adjust # of partitions for more or less parallelism
    .getOrCreate()
)

# Load data from HDFS
hdfs_path = "hdfs://namenode:8020/training_data/modified_fraud_dataset.csv"
df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(hdfs_path)
)

# Convert boolean is_fraud to numeric label 0/1
df_model = df.withColumn("label", F.col("is_fraud").cast("int"))

# Time-based features from raw timestamp
df_model = df_model.withColumn("ts", F.to_timestamp("timestamp"))
df_model = df_model.withColumn("hour_of_day", F.hour("ts"))
df_model = df_model.withColumn("day_of_week", F.dayofweek("ts"))
df_model = df_model.withColumn(
    "is_weekend",
     (F.col("day_of_week") >= 6).cast("int")
)

# Use the FULL dataset
df_used = df_model
used_rows = df_used.count()
print(f"Using ALL {used_rows} rows for weighted GBT in Spark.")

# Compute class counts on the dataset (for weighting)
fraud_count = df_used.filter(F.col("label") == 1).count()
nonfraud_count = df_used.filter(F.col("label") == 0).count()

if fraud_count == 0 or nonfraud_count == 0:
    weight_for_fraud = 1.0
    weight_for_nonfraud = 1.0
else:
    majority = max(fraud_count, nonfraud_count)
    weight_for_fraud = majority / float(fraud_count)
    weight_for_nonfraud = majority / float(nonfraud_count)

print(f"Weight for fraud (label=1):     {weight_for_fraud:.4f}")
print(f"Weight for non-fraud (label=0): {weight_for_nonfraud:.4f}")

# Add weight column based on label
df_weighted = df_used.withColumn(
    "weight",
    F.when(F.col("label") == 1, F.lit(weight_for_fraud))
     .otherwise(F.lit(weight_for_nonfraud))
)

# Categorical & numeric features
categorical_cols = [
    "transaction_type",
    "device_used",
    "location",
    "merchant_category",
]
numeric_cols = [
    "amount",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

# StringIndexer: string -> index
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep" 
    )
    for c in categorical_cols
]

# OneHotEncoder: index -> one-hot vector
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in categorical_cols],
    outputCols=[f"{c}_oh" for c in categorical_cols],
    dropLast=False,  # keep all categories
)

# Final feature columns = numeric + one-hot encoded categorical
feature_cols = numeric_cols + [f"{c}_oh" for c in categorical_cols]
print("Using features:", feature_cols)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep",
)

# Train/test split on df_weighted BEFORE assembling
train_df, test_df = df_weighted.randomSplit([0.8, 0.2], seed=42)
print(f"Train count: {train_df.count()}  Test count: {test_df.count()}")

# Gradient-Boosted Trees with class weights
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="weight",
    maxIter=10,
    maxDepth=4,
    stepSize=0.1,
    seed=42,
)

# Build a Pipeline: index -> encode -> assemble -> model
pipeline = Pipeline(
    stages=indexers + [encoder, assembler, gbt]
)

pipeline_model = pipeline.fit(train_df)
print("Weighted GBT pipeline training complete.")

# Predictions on test set
predictions = pipeline_model.transform(test_df).select("label", "prediction").cache()

preds = predictions.select(
    F.col("label").cast("int").alias("label"),
    F.col("prediction").cast("int").alias("prediction")
)

# confusion matrix
tp = preds.filter((F.col("label") == 1) & (F.col("prediction") == 1)).count()
tn = preds.filter((F.col("label") == 0) & (F.col("prediction") == 0)).count()
fp = preds.filter((F.col("label") == 0) & (F.col("prediction") == 1)).count()
fn = preds.filter((F.col("label") == 1) & (F.col("prediction") == 0)).count()

total = tp + tn + fp + fn

accuracy = (tp + tn) / total if total > 0 else 0.0
precision_fraud = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_fraud = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_fraud = (
    2 * (precision_fraud * recall_fraud) / (precision_fraud + recall_fraud)
    if (precision_fraud + recall_fraud) > 0
    else 0.0
)

print("\n=== CONFUSION MATRIX (Weighted GBT) ===")
print(f"TP (fraud predicted fraud):       {tp}")
print(f"FN (fraud predicted non-fraud):   {fn}")
print(f"FP (non-fraud predicted fraud):   {fp}")
print(f"TN (non-fraud predicted non):     {tn}")

print("\n=== METRICS (Weighted GBT) ===")
print(f"Accuracy:              {accuracy:.4f}")
print(f"Precision (fraud=1):   {precision_fraud:.4f}")
print(f"Recall (fraud=1):      {recall_fraud:.4f}")
print(f"F1-score (fraud=1):    {f1_fraud:.4f}")

# Save GBT model
print(f"Saving GBT model to: {MODEL_PATH}")
pipeline_model.write().overwrite().save(MODEL_PATH)
print("Model saved.")

spark.stop()
