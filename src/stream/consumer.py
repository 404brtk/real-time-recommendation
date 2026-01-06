from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "user-event",
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    group_id="user-event-group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
)


def consume_user_events():
    print("Starting consuming user events...")
    for message in consumer:
        event = message.value
        user_id = event["user_id"]
        action = event["action"]
        timestamp = event["timestamp"]
        print(f"Received user event for {user_id}: ({action}, at {timestamp}")


consume_user_events()
