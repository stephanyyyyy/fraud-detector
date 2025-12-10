from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DoubleType, IntegerType, StringType
)
from pyspark.ml import PipelineModel

# where to save streamed results (saving into HDFS)
PREDICTIONS_PATH = "hdfs://namenode:8020/fraud_stream_predictions"

# Path to the GBT pipeline model trained
MODEL_PATH = "/opt/spark/work-dir/fraud_gbt_model"

# Initialize Spark Session with Scala compatibility settings
spark = (SparkSession.builder
    .appName("Fraud_GPT_Weighted")
    .config("spark.sql.streaming.schemaInference", "true")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryo.registrationRequired", "false")
    .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# Define schema for incoming JSON
schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("device_used", StringType(), True),
    StructField("location", StringType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("hour_of_day", IntegerType(), True),
    StructField("day_of_week", IntegerType(), True),
    StructField("is_weekend", IntegerType(), True),
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

# Parse JSON from Kafka value ('value' field contains transaction JSON)
df_parsed = df_raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Process each micro-batch of streaming data
def predict_batch(df, epoch_id):
    if df.isEmpty():
        print(f"Epoch {epoch_id}: Empty batch, skipping...")
        return

    try:
        print(f"\n=== Processing Epoch {epoch_id} ===")

        # Load the trained GBT pipelime model
        gbt_model = PipelineModel.load(MODEL_PATH)

        # Get predictions
        predictions = gbt_model.transform(df)

        # Show transaction and predictions
        predictions.select(
            "timestamp",
            "amount",
            "transaction_type",
            "device_used",
            "location",
            "merchant_category",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_fraud",      # true label of transaction
            "prediction",    # what the model predicted
            "probability"    # probability model calculated for each label (for 0 and 1)
        ).show(truncate=False)
        
        # Save all predictions to HDFS for later analysis
        predictions_to_save = predictions.select(
            "timestamp",
            "amount",
            "transaction_type",
            "device_used",
            "location",
            "merchant_category",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_fraud",
            "prediction",
            "probability"
        ).withColumn("batch_id", F.lit(int(epoch_id)))

        # Append batch to HDFS parquet dataset
        predictions_to_save.write.mode("append").parquet(PREDICTIONS_PATH)

        # Calculate accuracy for every batch
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

# Start streaming query
try:
    query = df_parsed.writeStream \
        .foreachBatch(predict_batch) \
        .option("checkpointLocation", "/tmp/checkpoint") \
        .trigger(processingTime='15 seconds') \
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
