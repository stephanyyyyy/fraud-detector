from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DoubleType, IntegerType, StringType
)

# where to save streamed results (saving into HDFS)
PREDICTIONS_PATH = "hdfs://namenode:8020/fraud_stream_predictions"

# Path to the GBT pipeline model trained
MODEL_PATH = "/opt/spark/work-dir/fraud_gbt_model"

# Initialize Spark Session with Scala compatibility settings
spark = (SparkSession.builder
    # .appName("Fraud_LR_Weighted")
    # .appName("Fraud_RF_Weighted")
    .appName("Fraud_GPT_Weighted")
    .config("spark.sql.streaming.schemaInference", "true")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryo.registrationRequired", "false")
    .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# Define schema for incoming JSON
schema = StructType([
    StructField("timestamp", StringType(), True),
    # numerical futures
    StructField("amount", DoubleType(), True),
    StructField("spending_deviation_score", DoubleType(), True),
    StructField("velocity_score", DoubleType(), True),
    StructField("geo_anomaly_score", DoubleType(), True),
    # categorical features
    StructField("transaction_type", StringType(), True),
    StructField("device_used", StringType(), True),
    StructField("location", StringType(), True),
    StructField("merchant_category", StringType(), True),

    # is_fraud label
    StructField("is_fraud", IntegerType(), True),
])

# Read from Kafka
df_raw = (spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "fraud-transactions")
    .option("startingOffsets", "latest")
    .option("maxOffsetsPerTrigger", "5")
    
    .load())

# Parse JSON from Kafka value (the 'value' field contains transaction JSON)
df_parsed = df_raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

def predict_batch(df, epoch_id):
    """
    Process each micro-batch of streaming data
    """
    if df.isEmpty():
        print(f"Epoch {epoch_id}: Empty batch, skipping...")
        return

    try:
        print(f"\n=== Processing Epoch {epoch_id} ===")

        # Load the trained GBT pipelime model inside the function
        from pyspark.ml import PipelineModel
        gbt_model = PipelineModel.load(MODEL_PATH)

        # Time-based features (same logic as in training in model)
        df = df.withColumn("ts", F.to_timestamp("timestamp"))
        df = df.withColumn("hour_of_day", F.hour("ts"))
        df = df.withColumn("day_of_week", F.dayofweek("ts"))
        df = df.withColumn(
            "is_weekend",
            (F.col("day_of_week") >= 6).cast("int")
        )

        # Check for null values
        null_counts = df.select([col(c).isNull().cast("int").alias(c) for c in df.columns])
        if null_counts.first():
            print("Warning: Null values detected")
            df = df.na.drop()  # drop rows with nulls

        # Let the pipeline handle: index -> one-hot -> assemble -> GBT
        predictions = gbt_model.transform(df)

        # Show predictions
        predictions.select(
            "timestamp",
            "amount",
            "spending_deviation_score",
            "velocity_score",
            "geo_anomaly_score",
            "transaction_type",
            "device_used",
            "location",
            "merchant_category",
            "is_fraud",      # ground truth, if present
            "prediction",
            "probability"
        ).show(truncate=False)
        
        # Save all predictions to HDFS for later analysis
        predictions_to_save = predictions.select(
            "timestamp",
            "amount",
            "spending_deviation_score",
            "velocity_score",
            "geo_anomaly_score",
            "transaction_type",
            "device_used",
            "location",
            "merchant_category",
            "is_fraud",
            "prediction",
            "probability"
        ).withColumn("batch_id", F.lit(int(epoch_id)))

        # Append this batch to an HDFS Parquet dataset
        predictions_to_save.write.mode("append").parquet(PREDICTIONS_PATH)

        # Calculate accuracy if ground truth exists
        if "is_fraud" in predictions.columns:
            correct = predictions.filter(col("is_fraud") == col("prediction")).count()
            total = predictions.count()
            accuracy = correct / total if total > 0 else 0
            print(f"Batch accuracy: {accuracy:.2%} ({correct}/{total})")
        else:
            print("Batch accuracy: is_fraud column not present, skipping accuracy calc.")

    except Exception as e:
        print(f"Error in epoch {epoch_id}: {str(e)}")
        import traceback
        traceback.print_exc()

# Start streaming query with error handling
try:
    query = df_parsed.writeStream \
        .foreachBatch(predict_batch) \
        .option("checkpointLocation", "/tmp/checkpoint") \
        .trigger(processingTime='5 seconds') \
        .start()

    print("Streaming query started. Waiting for data...")
    print(f"Query ID: {query.id}")
    print(f"Status: {query.status}")

    query.awaitTermination()

except KeyboardInterrupt:
    print("\nStopping query...")
    query.stop()
    print("Query stopped successfully")
except Exception as e:
    print(f"Stream failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    spark.stop()
