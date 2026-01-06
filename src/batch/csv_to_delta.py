from pyspark.sql import SparkSession
from src.config import RAW_DIR, DATA_DIR
from src.logging import setup_logging
from pyspark.sql import functions as F

logger = setup_logging("csv_to_delta.log")


def create_spark_session():
    return (
        SparkSession.builder.appName("CSV_to_Delta")
        .config("spark.driver.memory", "2g")
        .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .master("local[*]")
        .getOrCreate()
    )


def convert_csv_to_delta(spark):
    logger.info(f"Reading CSV files from {RAW_DIR}...")

    df_customers = spark.read.csv(
        str(RAW_DIR / "customers.csv"), header=True, inferSchema=True
    )

    df_articles = spark.read.csv(
        str(RAW_DIR / "articles.csv"), header=True, inferSchema=True
    )
    # ensure article_id does not lose leading zeros
    df_articles = df_articles.withColumn(
        "article_id", F.format_string("%010d", F.col("article_id").cast("int"))
    )

    df_transactions = spark.read.csv(
        str(RAW_DIR / "transactions_train.csv"), header=True, inferSchema=True
    )
    # ensure article_id does not lose leading zeros
    df_transactions = df_transactions.withColumn(
        "article_id", F.format_string("%010d", F.col("article_id").cast("int"))
    )

    logger.info("Saving as Delta Lake tables...")
    df_customers.write.format("delta").mode("overwrite").save(
        str(DATA_DIR / "raw" / "customers")
    )
    df_articles.write.format("delta").mode("overwrite").save(
        str(DATA_DIR / "raw" / "articles")
    )
    df_transactions.write.format("delta").mode("overwrite").save(
        str(DATA_DIR / "raw" / "transactions")
    )

    logger.info("CSV to Delta Lake conversion completed.")


def main():
    spark = create_spark_session()
    convert_csv_to_delta(spark)
    spark.stop()


if __name__ == "__main__":
    main()
