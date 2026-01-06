import json
import time
import random
import pandas as pd
from kafka import KafkaProducer

from src.config import (
    MAPPINGS_DIR,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_EVENTS,
)
from src.logging import setup_logging

logger = setup_logging("producer.log")


def load_simulation_data():
    logger.info("Loading mapping files...")

    # load mapping: customer_id (hash) <-> user_idx (int)
    df_users = pd.read_parquet(str(MAPPINGS_DIR / "user_map.parquet"))
    # ensure customer_id is string
    df_users["customer_id"] = df_users["customer_id"].astype(str)
    # convert df to a list of records for O(1) random sampling via list indexing
    users_pool = df_users.to_dict("records")

    # load mapping: article_id (string like "0101010101") <-> item_idx (int)
    df_items = pd.read_parquet(str(MAPPINGS_DIR / "item_map.parquet"))
    df_items["article_id"] = df_items["article_id"].astype(str)
    items_pool = df_items.to_dict("records")

    logger.info(
        f"Loaded {len(users_pool)} users and {len(items_pool)} items into memory."
    )
    return users_pool, items_pool


def main():
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
        )
        logger.info(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        return

    users_pool, items_pool = load_simulation_data()

    if not users_pool or not items_pool:
        logger.error("Data pools are empty. Exiting.")
        return

    logger.info(f"Starting event stream to topic '{KAFKA_TOPIC_EVENTS}'...")
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            # randomly select user and item
            user = random.choice(users_pool)
            item = random.choice(items_pool)
            """
            # simulate event type with weighted probabilities
            # click (70%), add_to_cart (20%), purchase (10%)
            event_type = random.choices(
                ["click", "add_to_cart", "purchase"],
                weights=[0.7, 0.2, 0.1],
                k=1,
            )[0]
            """
            # TODO: update to support more event types
            event_type = "purchase"  # for now only simulate purchases

            # send BOTH the raw IDs (for analytics/frontend) and internal indices (for the model)
            event = {
                "user_id": user["customer_id"],
                "user_idx": int(user["user_idx"]),
                "item_id": item["article_id"],
                "item_idx": int(item["item_idx"]),
                "event_type": event_type,
                "timestamp": time.time(),
                # TODO: update to support variable quantities
                "quantity": 1,  # for now only simulate quantity=1 purchases
            }

            producer.send(KAFKA_TOPIC_EVENTS, value=event)

            logger.info(
                f"Sent: {event['event_type'].upper()} | "
                f"User: {event['user_idx']} -> Item: {event['item_idx']}"
            )

            time.sleep(random.uniform(0.1, 1.0))

    except KeyboardInterrupt:
        logger.info("Stopping producer by user request...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        producer.close()
        logger.info("Kafka producer closed.")


if __name__ == "__main__":
    main()
