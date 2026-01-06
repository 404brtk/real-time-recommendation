import json
import redis
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.config import (
    REDIS_HOST,
    REDIS_PORT,
    RAW_DIR,
    MAPPINGS_DIR,
    REDIS_POPULAR_KEY,
)
from src.logging import setup_logging

logger = setup_logging("calc_popular.log")

DAYS_LOOKBACK = 30


def create_spark_session():
    return (
        SparkSession.builder.appName("CalculatePopularItems")
        .config("spark.driver.memory", "4g")
        .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .master("local[*]")
        .getOrCreate()
    )


def get_trending_items(spark, k=100):
    logger.info("Reading RAW transactions from Delta Lake...")
    df_transactions = spark.read.format("delta").load(str(RAW_DIR / "transactions"))

    # for now we cannot really use current_date() as data is historical
    # so we'd just get empty results
    # so we find max date in data instead
    max_date_row = df_transactions.agg(F.max("t_dat").alias("max_date")).collect()[0]
    max_date = max_date_row["max_date"]

    logger.info(f"Dataset ends on: {max_date}. Filtering last {DAYS_LOOKBACK} days...")

    # transactions in the last DAYS_LOOKBACK days
    df_recent = df_transactions.filter(
        F.col("t_dat") >= F.date_sub(F.lit(max_date), DAYS_LOOKBACK)
    )

    # count distinct users per article in recent period
    # we could also just count total interactions
    # but this way we avoid bias towards items bought multiple times by same user
    trending_df = (
        df_recent.groupBy("article_id")
        .agg(F.countDistinct("customer_id").alias("score"))
        .orderBy(F.col("score").desc())
        .limit(k)
    )

    return trending_df


def map_and_enrich(spark, trending_df):
    """
    map article_id to item_idx and enrich with metadata
    """
    logger.info("Mapping IDs and enriching with metadata from Delta Lake...")

    trending_df = trending_df.withColumn(
        "article_id", F.col("article_id").cast("string")
    )
    df_map = spark.read.format("delta").load(str(MAPPINGS_DIR / "item_map"))
    df_map = df_map.withColumn("article_id", F.col("article_id").cast("string"))
    df_meta = spark.read.format("delta").load(str(RAW_DIR / "articles"))
    df_meta = df_meta.withColumn("article_id", F.col("article_id").cast("string"))

    # inner join to keep only items present in map
    trending_mapped = trending_df.join(df_map, on="article_id", how="inner")

    cols_to_select = [
        "item_idx",
        "article_id",
        "score",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
    ]

    result = trending_mapped.join(df_meta, on="article_id", how="left").select(
        *cols_to_select
    )
    return result


def save_to_redis(data_list):
    logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.set(REDIS_POPULAR_KEY, json.dumps(data_list))
        logger.info(
            f"Successfully saved {len(data_list)} trending items to '{REDIS_POPULAR_KEY}'"
        )
    except Exception as e:
        logger.error(f"Failed to write to Redis: {e}")
        raise e


def main():
    spark = create_spark_session()

    df_trending = get_trending_items(spark, k=50)

    df_enriched = map_and_enrich(spark, df_trending)

    # collect all rows from the distributed df into driver memory
    # we need to do this to convert spark df to a python list for redis storage
    # since we limited to top 50 items, this should be safe, memory-wise
    rows = df_enriched.collect()
    payload = []
    for row in rows:
        row_dict = row.asDict()
        payload.append(
            {
                "item_id": row_dict["item_idx"],
                "score": float(row_dict["score"]),
                "metadata": {
                    "article_id": row_dict["article_id"],
                    "name": row_dict["prod_name"],
                    "group": row_dict["product_group_name"],
                    "type": row_dict["product_type_name"],
                    "image_code": row_dict["article_id"][:3],
                },
                "source": "trending_now",
            }
        )
    save_to_redis(payload)
    spark.stop()


if __name__ == "__main__":
    main()
