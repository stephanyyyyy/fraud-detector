from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Where to save the Random Forest model
MODEL_PATH = "/opt/spark/work-dir/fraud_rf_unweighted_model"

# Start Spark
# can adjust number of partitions for more or less parallelism
spark = (
    SparkSession.builder
    .appName("Fraud_RF_Unweighted")
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

# Create StringIndexers for every categorical column to convert the text categories to a numeric index
indexers = []
for feature in categorical_cols:
    indexer = StringIndexer(inputCol=feature, outputCol=f"{feature}_idx", handleInvalid="keep")
    indexers.append(indexer)

# Build input and output lists for before and after encoding 
input_cols = []
for feature in categorical_cols:
    input_cols.append(f"{feature}_idx")

output_cols = []
for feature in categorical_cols:
    output_cols.append(f"{feature}_oh")

encoder = OneHotEncoder(inputCols=input_cols, outputCols=output_cols, dropLast=False)

# Combine numeric and one-hot encoded categorical features together for a full list of features
feature_cols = list(numeric_cols)
for feature in categorical_cols:
    feature_cols.append(f"{feature}_oh")

# Combine all features into a single features vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

# Train/test split (80/20) 
train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

# Random Forest classifier (unweighted)
random_forest_clf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=20,
    maxDepth=6,
    seed=42,
)

# Build pipeline with indexers, encoder, assembler, and RF
pipeline = Pipeline(stages=indexers + [encoder, assembler, random_forest_clf])

# Train model
rf_pipeline_model = pipeline.fit(train_df)
print("Training complete (Unweighted RF).")

# Predictions on test set
predictions = rf_pipeline_model.transform(test_df).select("label", "prediction").cache()

preds = predictions.select(
    F.col("label").cast("int").alias("label"),
    F.col("prediction").cast("int").alias("prediction")
)

# Calculate metrics for analysis 
# Confusion matrix
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

print("\n=== CONFUSION MATRIX (Unweighted RF) ===")
print(f"True Positives (fraud predicted fraud):       {true_pos}")
print(f"False Negatives (fraud predicted non-fraud):  {false_neg}")
print(f"False Positives (non-fraud predicted fraud):  {false_pos}")
print(f"True Negatives (non-fraud predicted non):     {true_neg}")

print("\n=== METRICS for Unweighted RF ===")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-score:    {f1_score:.4f}")

# Save RF model
rf_pipeline_model.write().overwrite().save(MODEL_PATH)
print("Model saved.")

spark.stop()
