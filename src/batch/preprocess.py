import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.logging import setup_logging
from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    MAPPINGS_DIR,
)

logger = setup_logging("preprocess.log")


def create_spark_session():
    return (
        SparkSession.builder.appName("ETL")
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


def read_raw_data(spark):
    logger.info("Reading RAW data from Delta Lake...")
    df_customers = spark.read.format("delta").load(str(RAW_DIR / "customers"))
    df_articles = spark.read.format("delta").load(str(RAW_DIR / "articles"))
    
    # load main historical transactions
    df_transactions = spark.read.format("delta").load(str(RAW_DIR / "transactions"))
    
    # try to load new streaming events if they exist
    events_path = str(RAW_DIR / "events")
    
    try:
        # check if delta table exists on disk
        if os.path.exists(events_path) and os.path.exists(os.path.join(events_path, "_delta_log")):
            logger.info(f"Found streaming events at {events_path}. Merging...")
            df_events = spark.read.format("delta").load(events_path)
            
            df_transactions = df_transactions.unionByName(df_events, allowMissingColumns=True)
            
            logger.info(f"Merged successfully. Total rows: {df_transactions.count()}")
        else:
            logger.info("No streaming events found yet. Skipping merge.")
            
    except Exception as e:
        # log warning but proceed with historical data only to not break the pipeline
        logger.warning(f"Could not load events data. Error: {e}")

    return df_customers, df_articles, df_transactions


# there is a problem with StringIndexer's metadata so we need to get around it
# and create our own id mapping
# TODO: investigate if there's a better way
def generate_ids_distributed(spark, df_input, id_col, idx_col, start_idx=0):
    rdd_ids = df_input.select(id_col).distinct().repartition(10).rdd.map(lambda r: r[0])
    rdd_zipped = rdd_ids.zipWithIndex()
    df_mapped = rdd_zipped.toDF([id_col, idx_col])
    if start_idx > 0:
        df_mapped = df_mapped.withColumn(idx_col, F.col(idx_col) + start_idx)
    return df_mapped.withColumn(idx_col, F.col(idx_col).cast("integer"))


def get_or_create_mapping(spark, df_input, id_col_name, idx_col_name, mapping_path):
    """
    stateful mapping function.
    ensures that existing ids keep their index, and only new ids get new indices.
    """
    logger.info(f"Handling mapping for: {id_col_name}")
    path_str = str(mapping_path)
    current_ids = df_input.select(id_col_name).distinct()

    is_delta_exists = False
    try:
        if os.path.exists(path_str) and os.path.exists(
            os.path.join(path_str, "_delta_log")
        ):
            is_delta_exists = True
    except Exception:
        pass

    if not is_delta_exists:
        logger.info(f"Creating NEW mapping table at {path_str}")

        df_mapped = generate_ids_distributed(
            spark, current_ids, id_col_name, idx_col_name, start_idx=0
        )

        df_mapped.write.format("delta").mode("overwrite").save(path_str)
        return df_mapped

    else:
        logger.info(f"Updating EXISTING mapping at {path_str}")
        existing_mapping = spark.read.format("delta").load(path_str)

        new_ids_only = current_ids.join(
            existing_mapping, on=id_col_name, how="left_anti"
        )

        if new_ids_only.count() == 0:
            logger.info("No new IDs found. Using existing mapping.")
            return existing_mapping

        max_idx_row = existing_mapping.agg(F.max(idx_col_name)).collect()[0][0]
        start_idx = (max_idx_row + 1) if max_idx_row is not None else 0

        logger.info(f"Found new IDs. Appending starting from index {start_idx}...")

        new_mapped = generate_ids_distributed(
            spark, new_ids_only, id_col_name, idx_col_name, start_idx=start_idx
        )

        new_mapped.write.format("delta").mode("append").save(path_str)

        return spark.read.format("delta").load(path_str)


def transform_transactions(df_transactions, df_user_map, df_item_map):
    logger.info("Transforming transactions...")

    df_with_users = df_transactions.join(
        F.broadcast(df_user_map), on="customer_id", how="inner"
    )

    df_with_users = df_with_users.withColumn(
        "article_id", F.col("article_id").cast("string")
    )
    df_final = df_with_users.join(df_item_map, on="article_id", how="inner")

    df_interactions = (
        df_final.groupBy("user_idx", "item_idx")
        .agg(F.log1p(F.sum(
            F.when(F.col("event_type") == "purchase", 1.0)
             .when(F.col("event_type") == "add_to_cart", 0.5)
             .when(F.col("event_type") == "click", 0.1)
             .otherwise(1.0)  # historical data without event_type - we can safely assume it's a purchase
        )).alias("rating"))
        .select(
            F.col("user_idx").cast("integer"),
            F.col("item_idx").cast("integer"),
            F.col("rating").cast("float"),
        )
    )
    return df_interactions


def main():
    spark = create_spark_session()

    df_customers, df_articles, df_transactions = read_raw_data(spark)

    df_user_map = get_or_create_mapping(
        spark, df_customers, "customer_id", "user_idx", MAPPINGS_DIR / "user_map"
    )
    df_articles = df_articles.withColumn(
        "article_id", F.col("article_id").cast("string")
    )

    df_item_map = get_or_create_mapping(
        spark, df_articles, "article_id", "item_idx", MAPPINGS_DIR / "item_map"
    )

    df_train_data = transform_transactions(df_transactions, df_user_map, df_item_map)

    count = df_train_data.count()
    logger.info(f"Generated {count} interaction rows.")

    df_train_data.write.format("delta").mode("overwrite").save(
        str(PROCESSED_DIR / "train_data")
    )

    logger.info("ETL Pipeline Finished Successfully.")
    spark.stop()


if __name__ == "__main__":
    main()
