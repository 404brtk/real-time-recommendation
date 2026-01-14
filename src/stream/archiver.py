from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_date, from_unixtime, lit
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)
from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_EVENTS, RAW_DIR
from src.logging import setup_logging

logger = setup_logging("archiver.log")


def create_spark_session():
    return (
        SparkSession.builder.appName("StreamArchiver")
        .config("spark.driver.memory", "2g")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0,io.delta:delta-spark_2.13:4.0.0",
        )
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .master("local[*]")
        .getOrCreate()
    )


def main():
    spark = create_spark_session()
    logger.info("Starting Delta Lake Archiver...")

    # schema must match the producer output
    kafka_schema = StructType(
        [
            StructField("user_id", StringType()),  # customer_id in batch
            StructField("user_idx", IntegerType()),
            StructField("item_id", StringType()),  # article_id in batch
            StructField("item_idx", IntegerType()),
            StructField("event_type", StringType()),
            StructField("timestamp", DoubleType()),
            StructField("quantity", IntegerType()),
        ]
    )

    df_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC_EVENTS)
        .option("startingOffsets", "latest")
        .load()
    )

    df_parsed = df_stream.select(
        from_json(col("value").cast("string"), kafka_schema).alias("data")
    ).select("data.*")

    # transform to match the transactions schema expected by preprocess.py
    # - rename user_id -> customer_id, item_id -> article_id
    # - convert unix timestamp to date format (matching t_dat in original transactions)
    df_transformed = (
        df_parsed.withColumn("customer_id", col("user_id"))
        .withColumn("article_id", col("item_id"))
        .withColumn("t_dat", to_date(from_unixtime(col("timestamp"))))
        .withColumn("price", lit(0.03))  # some average price - irrelevant in this case
        .withColumn(
            "sales_channel_id", lit(2)
        )  # it's online system so we can safely assume sales_channel_id is always 2
        .select(
            "t_dat",
            "customer_id",
            "article_id",
            "price",
            "sales_channel_id",
            "event_type",
        )
    )

    # we need checkpoint to keep track of processed events
    checkpoint_path = str(RAW_DIR / "events_checkpoint")
    # we make separate directory for events table, we could just append to transactions, since the schema is the same
    # but for safety we make a separate directory
    # TODO: potentially update to use the same directory as transactions
    output_path = str(RAW_DIR / "events")

    def log_batch(batch_df, batch_id):
        count = batch_df.count()
        if count > 0:
            # get event type breakdown
            event_counts = batch_df.groupBy("event_type").count().collect()
            breakdown = ", ".join(
                [f"{row['event_type']}: {row['count']}" for row in event_counts]
            )
            logger.info(f"Batch {batch_id}: Archived {count} events ({breakdown})")
            batch_df.write.format("delta").mode("append").save(output_path)

    query = (
        df_transformed.writeStream.foreachBatch(log_batch)
        .option("checkpointLocation", checkpoint_path)
        .start()
    )

    logger.info("Archiver ready. Listening for events...")
    logger.info(f"Output path: {output_path}")
    query.awaitTermination()


if __name__ == "__main__":
    main()
