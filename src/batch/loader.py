import pandas as pd
import redis
import json

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.logging import setup_logging
from src.config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_USER_VECTOR_PREFIX,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    VECTOR_SIZE,
    MODELS_DIR,
    MAPPINGS_DIR,
    RAW_DIR,
)


logger = setup_logging("loader.log")


def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def load_users_to_redis():
    logger.info("Loading Users to Redis...")
    r = get_redis_client()

    users_path = str(MODELS_DIR / "user_factors.parquet")
    logger.info(f"Reading {users_path}...")
    df_users = pd.read_parquet(users_path)

    logger.info(f"Uploading {len(df_users)} user vectors to Redis pipeline...")

    # use pipeline for batch operations
    pipe = r.pipeline()
    batch_size = 10000
    count = 0

    for _, row in df_users.iterrows():
        user_id = row["id"]
        vector = row["features"].tolist()

        key = f"{REDIS_USER_VECTOR_PREFIX}{user_id}"
        pipe.set(key, json.dumps(vector))

        count += 1
        if count % batch_size == 0:
            pipe.execute()
            print(f"Uploaded {count} users...", end="\r")

    pipe.execute()
    logger.info(f"\nSuccessfully uploaded {count} user vectors to Redis.")


def load_items_to_qdrant():
    logger.info("Loading Items to Qdrant...")
    client = get_qdrant_client()

    # load item vectors from ALS model
    df_vectors = pd.read_parquet(str(MODELS_DIR / "item_factors.parquet"))
    # load mapping from article_id to item_idx
    df_map = pd.read_parquet(str(MAPPINGS_DIR / "item_map.parquet"))
    # make sure article_id is string
    df_map["article_id"] = df_map["article_id"].astype(str)

    logger.info(f"Reading metadata from {RAW_DIR}...")
    df_meta = pd.read_parquet(str(RAW_DIR / "articles.parquet"))
    df_meta["article_id"] = df_meta["article_id"].astype(str)

    logger.info("Merging vectors with metadata...")

    # join vectors with mapping to get article_id for each vector
    df_merged = df_vectors.merge(df_map, left_on="id", right_on="item_idx")
    df_merged["article_id"] = df_merged["article_id"].astype(str)
    # join with metadata to get product details
    df_full = df_merged.merge(df_meta, on="article_id", how="left")

    # columns to store as payload in Qdrant for filtering/display
    payload_cols = [
        "article_id",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "index_group_name",
    ]
    df_full[payload_cols] = df_full[payload_cols].fillna("")

    # delete existing collection if it exists, then create a new one
    logger.info("Deleting existing Qdrant collection if exists...")
    try:
        client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
    except Exception:
        pass

    logger.info("Creating Qdrant collection...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    logger.info(f"Uploading {len(df_full)} items to Qdrant...")

    points = []
    batch_size = 5000

    for i, row in df_full.iterrows():
        point_id = int(row["item_idx"])
        vector = row["features"].tolist()

        payload = {col: row[col] for col in payload_cols}

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if len(points) >= batch_size:
            client.upload_points(collection_name=QDRANT_COLLECTION_NAME, points=points)
            print(f"Uploaded {i} items...", end="\r")
            points = []

    if points:
        client.upload_points(collection_name=QDRANT_COLLECTION_NAME, points=points)

    logger.info(
        f"\nSuccessfully uploaded items to Qdrant collection '{QDRANT_COLLECTION_NAME}'."
    )


def main():
    try:
        load_users_to_redis()
        load_items_to_qdrant()
        logger.info("Loader finished successfully!")
    except Exception as e:
        logger.error(f"Loader failed: {e}")
        raise


if __name__ == "__main__":
    main()
