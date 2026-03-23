"""
Dieses Skript wandelt die Spark-DenseVector-Wahrscheinlichkeiten
in eine einfache Float-Spalte um, damit Pandas sie lesen kann.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType

spark = SparkSession.builder \
    .appName("Fix Predictions") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

df = spark.read.parquet("output/predictions.parquet")

extract_prob = udf(lambda prob: float(prob[1]), FloatType())
df_fixed = df.withColumn("prob_positive", extract_prob(col("probability"))) \
    .select("label", "prediction", "prob_positive", "text")

df_fixed.write.parquet("output/predictions_fixed.parquet", mode="overwrite")
print(f"Fertig: {df_fixed.count():,} Zeilen")
spark.stop()
