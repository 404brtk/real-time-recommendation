from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
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
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .master("local[*]")
        .getOrCreate()
    )


def read_raw_data(spark):
    logger.info("Reading RAW data from Delta Lake...")
    df_customers = spark.read.format("delta").load(str(RAW_DIR / "customers"))
    df_articles = spark.read.format("delta").load(str(RAW_DIR / "articles"))
    df_transactions = spark.read.format("delta").load(str(RAW_DIR / "transactions"))
    return df_customers, df_articles, df_transactions


def build_user_mapping(df_customers, df_transactions):
    logger.info("Building mapping for User ID")
    user_indexer = StringIndexer(
        inputCol="customer_id", outputCol="user_idx", handleInvalid="keep"
    )
    user_indexer_model = user_indexer.fit(df_transactions)
    df_user_map = (
        user_indexer_model.transform(df_customers.select("customer_id"))
        .select("customer_id", "user_idx")
        .distinct()
    )
    df_user_map.write.format("delta").mode("overwrite").save(
        str(MAPPINGS_DIR / "user_map")
    )
    logger.info("User ID mapping saved to Delta Lake.")
    return user_indexer_model


def build_item_mapping(df_articles):
    logger.info("Building mapping for Item ID")
    # make sure article_id is string
    df_articles = df_articles.withColumn(
        "article_id", F.col("article_id").cast("string")
    )
    item_indexer = StringIndexer(
        inputCol="article_id", outputCol="item_idx", handleInvalid="keep"
    )
    item_indexer_model = item_indexer.fit(df_articles)
    df_item_map = (
        item_indexer_model.transform(df_articles.select("article_id"))
        .select("article_id", "item_idx")
        .distinct()
    )
    df_item_map.write.format("delta").mode("overwrite").save(
        str(MAPPINGS_DIR / "item_map")
    )
    logger.info("Item ID mapping saved to Delta Lake.")
    return item_indexer_model


def transform_transactions(df_transactions, user_indexer_model, item_indexer_model):
    logger.info("Transforming transactions data...")
    df_transactions_users = user_indexer_model.transform(df_transactions)
    df_transactions_final = item_indexer_model.transform(df_transactions_users)

    df_interactions = (
        df_transactions_final.groupBy("user_idx", "item_idx")
        .agg(F.log1p(F.count("article_id")).alias("rating"))
        .select("user_idx", "item_idx", "rating")
    )
    df_interactions = (
        df_interactions.withColumn("user_idx", F.col("user_idx").cast("integer"))
        .withColumn("item_idx", F.col("item_idx").cast("integer"))
        .withColumn("rating", F.col("rating").cast("float"))
    )
    logger.info(f"Number of interactions: {df_interactions.count()}")
    return df_interactions


def save_processed_data(df_interactions):
    df_interactions.write.format("delta").mode("overwrite").save(
        str(PROCESSED_DIR / "train_data")
    )
    logger.info(
        "Preprocessing completed successfully. Delta Lake tables ready at data/processed/"
    )


def main():
    spark = create_spark_session()

    df_customers, df_articles, df_transactions = read_raw_data(spark)

    user_indexer_model = build_user_mapping(df_customers, df_transactions)
    item_indexer_model = build_item_mapping(df_articles)

    df_interactions = transform_transactions(
        df_transactions, user_indexer_model, item_indexer_model
    )

    save_processed_data(df_interactions)

    spark.stop()


if __name__ == "__main__":
    main()
