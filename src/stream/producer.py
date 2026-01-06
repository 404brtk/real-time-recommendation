from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def produce_user_events():
    print("Starting producing user events...")
    while True:
        users = [f"user{i}" for i in range(10)]
        actions = ["click", "add_to_cart", "purchase"]

        event = {
            "user_id": random.choice(users),
            "action": random.choice(actions),
            "timestamp": time.time(),
        }
        producer.send("user-event", event)
        print(f"Sent: {event}")
        time.sleep(2)


produce_user_events()
