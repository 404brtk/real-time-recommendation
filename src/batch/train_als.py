from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from src.logging import setup_logging
from src.config import MODELS_DIR, PROCESSED_DIR
import time


logger = setup_logging("train_als.log")


def create_spark_session():
    return (
        SparkSession.builder.appName("ALS_Training")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .master("local[*]")
        .getOrCreate()
    )


def train_als_model(spark):
    logger.info("Loading training data from Delta Lake...")
    df_train = spark.read.format("delta").load(str(PROCESSED_DIR / "train_data"))

    als = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        implicitPrefs=True,
        rank=32,
        regParam=0.01,
        alpha=1.0,  # weighing factor for "ratings" in implicit feedback
        maxIter=10,
        coldStartStrategy="drop",
    )

    logger.info("Starting ALS training... (this may take a while)")
    start_time = time.time()
    model = als.fit(df_train)
    duration = time.time() - start_time
    logger.info(f"Training completed in {duration:.2f} seconds.")

    return model


def save_factors(model):
    logger.info("Saving user and item factors to Delta Lake...")

    model.userFactors.write.format("delta").mode("overwrite").save(
        str(MODELS_DIR / "user_factors")
    )
    logger.info("User factors saved to data/models/user_factors")

    model.itemFactors.write.format("delta").mode("overwrite").save(
        str(MODELS_DIR / "item_factors")
    )
    logger.info("Item factors saved to data/models/item_factors")


def main():
    spark = create_spark_session()

    model = train_als_model(spark)

    save_factors(model)

    spark.stop()


if __name__ == "__main__":
    main()
