import os
from pathlib import Path

# data paths
DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
MAPPINGS_DIR = DATA_DIR / "mappings"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_USER_VECTOR_PREFIX = "user:vector:"
REDIS_USER_HISTORY_PREFIX = "user:history:"
REDIS_USER_MAP_PREFIX = "user:map:"  # user_idx -> customer_id mapping
REDIS_POPULAR_KEY = "global:popular_items"
REDIS_DEBOUNCE_PREFIX = "debounce:"
DEBOUNCE_SECONDS = 300  # 5 minutes

# qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = "items"
VECTOR_SIZE = 32  # the same as in als rank

# kafka
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_EVENTS = "user_events"

# online learning
LEARNING_RATE = 0.2
EVENT_WEIGHT_MULTIPLIERS = {"purchase": 1.0, "add_to_cart": 0.5, "click": 0.1}
