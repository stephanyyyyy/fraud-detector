from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Where to save the trained Logistic Regression pipeline model (inside Spark container)
MODEL_PATH = "/opt/spark/work-dir/fraud_lr_model"

# Start Spark
# Adjust number of partitions for more or less parallelism
spark = (
    SparkSession.builder
    .appName("Fraud_LR_Weighted")
    .config("spark.sql.shuffle.partitions", "8")
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

# Compute class counts on the dataset (for weighting)
fraud_count = df_model.filter(F.col("label") == 1).count()
nonfraud_count = df_model.filter(F.col("label") == 0).count()

if fraud_count == 0 or nonfraud_count == 0:
    weight_for_fraud = 1.0
    weight_for_nonfraud = 1.0
else:
    majority = max(fraud_count, nonfraud_count)
    weight_for_fraud = majority / float(fraud_count)
    weight_for_nonfraud = majority / float(nonfraud_count)

# Print weight for each class
print(f"Weight for fraud (label=1): {weight_for_fraud:.4f}")
print(f"Weight for non-fraud (label=0): {weight_for_nonfraud:.4f}")

# Add weight column based on label
df_weighted = df_model.withColumn(
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
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

# Create StringIndexers for every categorical column to convert text → numeric index
indexers = []
for feature in categorical_cols:
    indexer = StringIndexer(
        inputCol=feature,
        outputCol=f"{feature}_idx",
        handleInvalid="keep"
    )
    indexers.append(indexer)

# Build input and output lists for OneHotEncoder
input_cols = []
for feature in categorical_cols:
    input_cols.append(f"{feature}_idx")

output_cols = []
for feature in categorical_cols:
    output_cols.append(f"{feature}_oh")

encoder = OneHotEncoder(
    inputCols=input_cols,
    outputCols=output_cols,
    dropLast=False
)

# Combine numeric and one-hot encoded categorical features for a full feature list
feature_cols = list(numeric_cols)
for feature in categorical_cols:
    feature_cols.append(f"{feature}_oh")

# Combine all features into a single "features" vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep"
)

# Train/test split (80/20)
train_df, test_df = df_weighted.randomSplit([0.8, 0.2], seed=42)

# Logistic Regression with class weights
log_reg_clf = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="weight",
    maxIter=20,
    regParam=0.0,
    elasticNetParam=0.0,
)

# Build pipeline with indexers, encoder, assembler, and 
pipeline = Pipeline(stages=indexers + [encoder, assembler, log_reg_clf])

# Train model
lr_pipeline_model = pipeline.fit(train_df)
print("Training complete (Weighted LR).")

# Predictions on test set
predictions = lr_pipeline_model.transform(test_df).select("label", "prediction").cache()

preds = predictions.select(
    F.col("label").cast("int").alias("label"),
    F.col("prediction").cast("int").alias("prediction")
)

# Confusion matrix and metrics for analysis
true_pos = preds.filter((F.col("label") == 1) & (F.col("prediction") == 1)).count()
true_neg = preds.filter((F.col("label") == 0) & (F.col("prediction") == 0)).count()
false_pos = preds.filter((F.col("label") == 0) & (F.col("prediction") == 1)).count()
false_neg = preds.filter((F.col("label") == 1) & (F.col("prediction") == 0)).count()

total = true_pos + true_neg + false_pos + false_neg

if total > 0:
    accuracy = (true_pos + true_neg) / total
else:
    accuracy = 0.0

if (true_pos + false_pos) > 0:
    precision = true_pos / (true_pos + false_pos)
else:
    precision = 0.0

if (true_pos + false_neg) > 0:
    recall = true_pos / (true_pos + false_neg)
else:
    recall = 0.0

if (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0.0

print("\n=== CONFUSION MATRIX (Weighted LR) ===")
print(f"True Positives (fraud predicted fraud):       {true_pos}")
print(f"False Negatives (fraud predicted non-fraud):  {false_neg}")
print(f"False Positives (non-fraud predicted fraud):  {false_pos}")
print(f"True Negatives (non-fraud predicted non):     {true_neg}")

print("\n=== METRICS for Weighted LR ===")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-score:    {f1_score:.4f}")

# Save LR model
lr_pipeline_model.write().overwrite().save(MODEL_PATH)
print("Model saved.")

spark.stop()
