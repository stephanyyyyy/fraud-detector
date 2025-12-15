## Prerequisites/Services Used
Python 3.10+
Docker & Docker Compose
Hadoop HDFS 3.2.1(NameNode + DataNode)
Apache Spark 4.0.1 (PySpark)
Apache Kafka 2.8.1
Apache Zookeeper 3.8.4


## Dependencies
kafka-python (installed inside Spark container)
numpy (installed inside Spark container)

# Setup 
Start all services:
docker compose up -d
Verify containers are running:
docker compose ps


# HDFS Setup
Enter the NameNode container and create HDFS directories.
docker exec -it namenode_zk bash
hdfs dfs -mkdir -p /training_data
hdfs dfs -mkdir -p /user/spark


Copy files into the Spark Container
docker cp fraud-detector/. spark_zk:/app


Install Python Dependencies inside Spark
docker exec -it -u root spark_zk bash
python3 -m pip install numpy kafka-python


Load Training Data into HDFS
Create an HDFS directory and upload the dataset.
hdfs dfs -put modified_fraud_dataset.csv /training_data


Train the Fraud Detection Model
docker exec -it spark_zk bash
/opt/spark/bin/spark-submit /app/gbt.py
This will generate the trained model at:
/opt/spark/work-dir/fraud_gbt_model

To run another model to see evaluation metrics run:
/opt/spark/bin/spark-submit /app/<file_name>
Create Kafka Topic
Inside the Kafka container, create the topic used for streaming:
docker exec -it kafka-broker bash
kafka-topics.sh --bootstrap-server kafka-broker:9092 --create --topic fraud-transactions --partitions 1 --replication-factor 1


IX. STEPS TO RUN THE APPLICATION
Run the Kafka producer to begin generating synthetic transactions
docker exec -it spark_zk bash
python3 /app/kafka_producer.py


Run Spark Streaming Job
In a new terminal, start the Spark Structured Streaming job:
/opt/spark/bin/spark-submit   --conf spark.jars.ivy=/tmp/ivy   --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,org.apache.spark:spark-token-provider-kafka-0-10_2.13:4.0.1   /app/fraud_streaming.py


Viewing Saved Predictions
Predictions will be printed to the terminal and saved to HDFS. Verify predictions in HDFS using:
hdfs dfs -ls /user/spark/fraud_stream_predictions

To view actual contents, inside of the Spark container, Run:
/opt/spark/bin/spark-submit /app/view_saved_predictions.py
