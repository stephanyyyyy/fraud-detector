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

# Categories for categorical data
TRANSACTION_TYPES = ["withdrawal", "deposit", "transfer", "payment"]
DEVICES = ["mobile", "atm", "pos", "web"]
LOCATIONS = ["Tokyo", "Toronto", "London", "Sydney", "Berlin", "Dubai", "New York", "Singapore"]
MERCHANT_CATEGORIES = ["utilities", "online", "other", "entertainment", "travel", "grocery", "retail", "restaurant"]

# Generate a complete synthetiic transaction
def generate_fake_transaction():
    day_of_week = random.randint(1, 7)
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "amount": round(random.uniform(1, 1000), 2),
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "device_used": random.choice(DEVICES),
        "location": random.choice(LOCATIONS),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "hour_of_day": random.randint(0, 23),
        "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week >= 6 else 0,
        "is_fraud": random.choices([0, 1], weights=[95, 5])[0]  # give more weight to nonfraud transactions
    }
        

if __name__ == "__main__":
    while True:
        txn = generate_fake_transaction()
        producer.send(TOPIC, value=txn)
        print(f"Sent: {txn}")
        time.sleep(1)  # 1 transaction every second
