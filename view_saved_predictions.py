# view saved predctions in fraud_stream_predictions in HDFS
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("hdfs://namenode:8020/fraud_stream_predictions")

df.show(truncate=False)
