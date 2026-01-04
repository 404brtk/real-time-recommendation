from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from src.logging import setup_logging

logger = setup_logging("preprocess.log")


def create_spark_session():
    return (
        SparkSession.builder.appName("ETL")
        .config("spark.driver.memory", "8g")
        .master("local[*]")
        .getOrCreate()
    )


def read_raw_data(spark):
    logger.info("Reading RAW data...")
    df_customers = spark.read.csv(
        "data/raw/customers.csv", header=True, inferSchema=True
    )
    df_articles = spark.read.csv("data/raw/articles.csv", header=True, inferSchema=True)
    df_transactions = spark.read.csv(
        "data/raw/transactions_train.csv", header=True, inferSchema=True
    )
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
    df_user_map.write.mode("overwrite").parquet("data/mappings/user_map.parquet")
    logger.info("User ID mapping saved.")
    return user_indexer_model


def build_item_mapping(df_articles):
    logger.info("Building mapping for Item ID")
    item_indexer = StringIndexer(
        inputCol="article_id", outputCol="item_idx", handleInvalid="keep"
    )
    item_indexer_model = item_indexer.fit(df_articles)
    df_item_map = (
        item_indexer_model.transform(df_articles.select("article_id"))
        .select("article_id", "item_idx")
        .distinct()
    )
    df_item_map.write.mode("overwrite").parquet("data/mappings/item_map.parquet")
    logger.info("Item ID mapping saved.")
    return item_indexer_model


def transform_transactions(df_transactions, user_indexer_model, item_indexer_model):
    logger.info("Transforming transactions data...")
    df_transactions_users = user_indexer_model.transform(df_transactions)
    df_transactions_final = item_indexer_model.transform(df_transactions_users)

    # ALS needs 'rating' column even though it's implicit feedback
    # we can bypass it by either setting everything to 1 or using count of interactions
    # in this case we'll set everything to count
    # or actually use log(1+count) so that extreme counts don't dominate too much
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
    df_interactions.write.mode("overwrite").parquet("data/processed/train_data.parquet")
    logger.info(
        "Preprocessing completed successfully. The data is ready at data/processed/"
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
