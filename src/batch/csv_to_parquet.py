from pyspark.sql import SparkSession
from src.config import RAW_DIR, DATA_DIR
from src.logging import setup_logging


logger = setup_logging("csv_to_parquet.log")


def convert_csv_to_parquet(spark):
    logger.info(f"Reading CSV files from {RAW_DIR}...")

    df_customers = spark.read.csv(
        str(RAW_DIR / "customers.csv"), header=True, inferSchema=True
    )
    df_articles = spark.read.csv(
        str(RAW_DIR / "articles.csv"), header=True, inferSchema=True
    )
    df_transactions = spark.read.csv(
        str(RAW_DIR / "transactions_train.csv"), header=True, inferSchema=True
    )

    logger.info("Saving as parquet...")
    df_customers.write.mode("overwrite").parquet(
        str(DATA_DIR / "raw" / "customers.parquet")
    )
    df_articles.write.mode("overwrite").parquet(
        str(DATA_DIR / "raw" / "articles.parquet")
    )
    df_transactions.write.mode("overwrite").parquet(
        str(DATA_DIR / "raw" / "transactions.parquet")
    )

    logger.info("CSV to Parquet conversion completed.")


def main():
    spark = (
        SparkSession.builder.appName("CSV_to_Parquet")
        .config("spark.driver.memory", "2g")
        .master("local[*]")
        .getOrCreate()
    )
    convert_csv_to_parquet(spark)
    spark.stop()


if __name__ == "__main__":
    main()
