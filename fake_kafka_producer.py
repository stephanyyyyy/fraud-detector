from kafka import KafkaProducer
import json
import random
import time
from datetime import datetime

# Kafka broker
BROKER = "kafka:9092" # or kafka-broker:9092
TOPIC = "fraud-transactions"

producer = KafkaProducer(
    bootstrap_servers=[BROKER],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Categories for fake data
TRANSACTION_TYPES = ["withdrawal", "deposit", "transfer", "payment"]
DEVICES = ["mobile", "atm", "pos", "web"]
LOCATIONS = ["Tokyo", "Toronto", "London", "Sydney", "Berlin", "Dubai", "New York", "Singapore"]
MERCHANT_CATEGORIES = ["utilities", "online", "other", "entertainment", "travel", "grocery", "reatil", "restaurant"]

# Generate a complete fake transaction
def generate_fake_transaction():
    return {
        "timestamp": datetime.utcnow().isoformat(),  # ISO format timestamp

        "amount": round(random.uniform(1, 1000), 2),
        "spending_deviation_score": round(random.uniform(0, 10), 2),
        "velocity_score": round(random.uniform(0, 10), 2),
        "geo_anomaly_score": round(random.uniform(0, 10), 2),

        "transaction_type": random.choice(TRANSACTION_TYPES),
        "device_used": random.choice(DEVICES),
        "location": random.choice(LOCATIONS),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),

        "is_fraud": random.choices([0, 1], weights=[95, 5])[0]
    }

if __name__ == "__main__":
    while True:
        txn = generate_fake_transaction()
        producer.send(TOPIC, value=txn)
        print(f"Sent: {txn}")
        time.sleep(1)  # 1 per second
