from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# create SparkSeeesion
spark = (SparkSession.builder.appName("Fraud_LR_Weighted").config("spark.sql.shuffle.partitions", "8").getOrCreate())

# load dataset from HDFS
hdfs_path = "hdfs://namenode:8020/training_data/modified_fraud_dataset.csv"
df = (spark.read.option("header", "true").option("inferSchema", "true").csv(hdfs_path))

# convert labels to 0 and 1 (0 for nonfraud and 1 for fraud)
df_model = df.withColumn("label", F.col("is_fraud").cast("int"))

# calculate weight for each class to help with the imbalance
fraud_count = df_model.filter(F.col("label") == 1).count()
nonfraud_count = df_model.filter(F.col("label") == 0).count()

if fraud_count == 0 or nonfraud_count == 0:
    fraud_weight = 1.0
    nonfraud_weight = 1.0
else:
    majority = max(fraud_count, nonfraud_count)
    fraud_weight = majority / float(fraud_count)
    nonfraud_weight = majority / float(nonfraud_count)

print(f"Weight for fraud: {fraud_weight:.4f}")
print(f"Weight for nonfraud: {nonfraud_weight:.4f}")

# add the weight column to df based on label
df_weighted = df_model.withColumn("weight", F.when(F.col("label") == 1, F.lit(fraud_weight)).otherwise(F.lit(nonfraud_weight)))

# choose features (numeric) NEED TO CHANGE
feature_cols = ["amount", "spending_deviation_score", "velocity_score", "geo_anomaly_score",]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep",)

df_vector = assembler.transform(df_weighted).select("features", "label", "weight")

# split data (90/10) for train/test; can be adjusted as needed 
train_df, test_df = df_vector.randomSplit([0.9, 0.1], seed=42)

# train logistic regression model and get predictions on test set
lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="weight", maxIter=20,regParam=0.0, elasticNetParam=0.0,)
lr_model = lr.fit(train_df)
predictions = lr_model.transform(test_df).select("label", "prediction").cache()

# model metrics (can remove or comment out later)
preds = predictions.select(F.col("label").cast("int").alias("label"), F.col("prediction").cast("int").alias("prediction"))

# confusion matrix
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
    precision_fraud = true_pos / (true_pos + false_pos)
else:
    precision_fraud = 0.0

if (true_pos + false_neg) > 0:
    recall_fraud = true_pos / (true_pos + false_neg)
else:
    recall_fraud = 0.0

if (precision_fraud + recall_fraud) > 0:
    f1_fraud = 2 * (precision_fraud * recall_fraud) / (precision_fraud + recall_fraud)
else:
    f1_fraud = 0.0

print("\n=== CONFUSION MATRIX ===")
print(f"TP (fraud predicted fraud): {true_pos}")
print(f"FN (fraud predicted nonfraud): {false_neg}")
print(f"FP (nonfraud predicted fraud): {false_pos}")
print(f"TN (nonfraud predicted nonfraud): {true_neg}")

print("\n=== METRICS ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (fraud=1): {precision_fraud:.4f}")
print(f"Recall (fraud=1): {recall_fraud:.4f}")
print(f"F1-score (fraud=1): {f1_fraud:.4f}\n")
