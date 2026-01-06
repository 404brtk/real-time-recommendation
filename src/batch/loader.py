import json
import redis

from pyspark.sql import SparkSession
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


def create_spark_session():
    return (
        SparkSession.builder.appName("Loader")
        .config("spark.driver.memory", "8g")
        .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "50")
        .master("local[*]")
        .getOrCreate()
    )


def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def load_users_to_redis(spark):
    logger.info("Loading Users from Delta Lake to Redis...")
    r = get_redis_client()

    try:
        users_path = str(MODELS_DIR / "user_factors")
        logger.info(f"Reading {users_path}...")
        df_users = spark.read.format("delta").load(users_path)

        # collect to driver and upload in batches
        logger.info("Collecting user vectors...")
        users_data = df_users.collect()

        logger.info(f"Uploading {len(users_data)} user vectors to Redis pipeline...")

        # use pipeline for batch operations
        pipe = r.pipeline()
        batch_size = 10000
        count = 0

        for row in users_data:
            user_id = row["id"]
            vector = list(row["features"])

            key = f"{REDIS_USER_VECTOR_PREFIX}{user_id}"
            pipe.set(key, json.dumps(vector))

            count += 1
            if count % batch_size == 0:
                pipe.execute()
                print(f"Uploaded {count} users...", end="\r")

        pipe.execute()
        logger.info(f"Successfully uploaded {count} user vectors to Redis.")

    finally:
        r.close()


def load_items_to_qdrant(spark):
    logger.info("Loading Items from Delta Lake to Qdrant...")
    client = get_qdrant_client()

    # load item vectors from ALS model
    df_vectors = spark.read.format("delta").load(str(MODELS_DIR / "item_factors"))
    # load mapping from article_id to item_idx
    df_map = spark.read.format("delta").load(str(MAPPINGS_DIR / "item_map"))

    logger.info(f"Reading metadata from {RAW_DIR}...")
    df_meta = spark.read.format("delta").load(str(RAW_DIR / "articles"))

    # ensure join keys are strings
    df_map = df_map.withColumn("article_id", df_map["article_id"].cast("string"))
    df_meta = df_meta.withColumn("article_id", df_meta["article_id"].cast("string"))

    logger.info("Merging vectors with metadata...")

    # join vectors with mapping to get article_id for each vector
    df_merged = df_vectors.join(df_map, df_vectors["id"] == df_map["item_idx"])
    # join with metadata to get product details
    df_full = df_merged.join(df_meta, on="article_id", how="left")

    # columns to store as payload in Qdrant for filtering/display
    payload_cols = [
        "article_id",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "index_group_name",
    ]

    # fill nulls just in case
    for col in payload_cols:
        df_full = df_full.fillna({col: ""})

    logger.info("Collecting item data...")
    items_data = df_full.select("item_idx", "features", *payload_cols).collect()

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

    logger.info(f"Uploading {len(items_data)} items to Qdrant...")

    points = []
    batch_size = 5000

    for i, row in enumerate(items_data):
        point_id = int(row["item_idx"])
        vector = list(row["features"])

        payload = {col: row[col] for col in payload_cols}

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if len(points) >= batch_size:
            client.upload_points(collection_name=QDRANT_COLLECTION_NAME, points=points)
            print(f"Uploaded {i + 1} items...", end="\r")
            points = []

    if points:
        client.upload_points(collection_name=QDRANT_COLLECTION_NAME, points=points)

    logger.info(
        f"\nSuccessfully uploaded items to Qdrant collection '{QDRANT_COLLECTION_NAME}'."
    )


def main():
    spark = create_spark_session()
    try:
        load_users_to_redis(spark)
        load_items_to_qdrant(spark)
        logger.info("Loader finished successfully!")
    except Exception as e:
        logger.error(f"Loader failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
