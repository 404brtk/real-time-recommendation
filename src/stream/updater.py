import json
import redis
import numpy as np
from kafka import KafkaConsumer
from qdrant_client import QdrantClient
import time

from src.logging import setup_logging
from src.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_EVENTS,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_USER_VECTOR_PREFIX,
    REDIS_USER_HISTORY_PREFIX,
    REDIS_DEBOUNCE_PREFIX,
    DEBOUNCE_SECONDS,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    LEARNING_RATE,
    EVENT_WEIGHT_MULTIPLIERS,
)

logger = setup_logging("updater.log")


def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_user_vector(r, user_idx):
    key = f"{REDIS_USER_VECTOR_PREFIX}{user_idx}"
    vector_str = r.get(key)
    if vector_str:
        return np.array(json.loads(vector_str), dtype=np.float32)
    return None


def save_user_vector(r, user_idx, vector):
    key = f"{REDIS_USER_VECTOR_PREFIX}{user_idx}"
    r.set(key, json.dumps(vector.tolist()))


def get_item_vector(qdrant, item_idx):
    try:
        points = qdrant.retrieve(
            collection_name=QDRANT_COLLECTION_NAME,
            ids=[int(item_idx)],
            with_vectors=True,
        )
        if points:
            return np.array(points[0].vector, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error retrieving item {item_idx} from Qdrant: {e}")
    return None


def calculate_new_vector(user_vector, item_vector, weight_multiplier):
    """
    move user vector towards item vector
    """
    # cold start: if user has no vector yet, then just use item vector as a starting point
    if user_vector is None:
        return item_vector

    effective_lr = LEARNING_RATE * weight_multiplier

    # apply exponential moving average to drift user profile towards the new item
    new_vector = (user_vector * (1 - effective_lr)) + (item_vector * effective_lr)
    return new_vector


def main():
    logger.info("Starting user profile updater (online learning)...")

    r = get_redis_client()
    qdrant = get_qdrant_client()

    consumer = KafkaConsumer(
        KAFKA_TOPIC_EVENTS,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="updater_group_v1",
        auto_offset_reset="latest",
    )

    logger.info(
        f"Listening on topic '{KAFKA_TOPIC_EVENTS}' with learning rate {LEARNING_RATE}..."
    )

    try:
        for message in consumer:
            event = message.value
            user_idx = event.get("user_idx")
            item_idx = event.get("item_idx")
            event_type = event.get("event_type")

            weight_multiplier = EVENT_WEIGHT_MULTIPLIERS.get(event_type)
            if weight_multiplier is None:
                continue

            debounce_key = f"{REDIS_DEBOUNCE_PREFIX}{user_idx}:{item_idx}:{event_type}"

            if r.exists(debounce_key):
                logger.debug(f"Skipping duplicate: {user_idx}->{item_idx} ({event_type})")
                continue

            r.setex(debounce_key, DEBOUNCE_SECONDS, "1")

            logger.info(
                f"Processing event: {event_type} | User: {user_idx} -> Item: {item_idx}"
            )

            item_vector = get_item_vector(qdrant, item_idx)
            if item_vector is None:
                logger.warning(
                    f"Item {item_idx} not found in vector DB. Skipping update."
                )
                continue

            user_vector = get_user_vector(r, user_idx)

            new_user_vector = calculate_new_vector(
                user_vector, item_vector, weight_multiplier
            )

            save_user_vector(r, user_idx, new_user_vector)

            # TODO: investigate different strategies for user history, how to manage it, maybe there's a better way
            # if purchase, add item to user history
            if event_type == "purchase":
                history_key = f"{REDIS_USER_HISTORY_PREFIX}{user_idx}"
                current_time = int(time.time())
                # zset (sorted set) where score is timestamp so we can keep only e.g. last n items or items from last n days
                # kinda sliding window
                # so we don't store everything in redis (ram)
                r.zadd(history_key, {str(item_idx): current_time})

                # keep only last 100 items
                r.zremrangebyrank(history_key, 0, -101)

                # also set expire to 1 year, so inactive users' history doesn't take up space
                r.expire(history_key, 60 * 60 * 24 * 365)

            status = "UPDATED" if user_vector is not None else "CREATED (Cold Start)"
            logger.info(f"User {user_idx} vector {status} successfully.")

    except KeyboardInterrupt:
        logger.info("Stopping updater...")
    finally:
        consumer.close()
        r.close()


if __name__ == "__main__":
    main()
