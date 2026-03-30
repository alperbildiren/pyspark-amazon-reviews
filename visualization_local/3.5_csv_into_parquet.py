#4.0 csv into parquet


from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CSV to Parquet Conversion") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Pfad zu deiner CSV anpassen
csv_path = "/Users/zuky/Documents/Studium/Bachelor_BusinessAnalytics/BBA_Vertiefung/BigData/cleaned_reviews.csv"   # ← hier deinen Pfad eintragen

df = spark.read.csv(
    csv_path,
    header=True,           # wichtig, weil Parquet aus Schritt 2 eine Header hat
    inferSchema=True,
    multiLine=True,
    escape='"'
)

print(f"CSV geladen: {df.count():,} Zeilen")
df.printSchema()

# Als Parquet speichern (genau wie im Original)
output_path = "/Users/zuky/Documents/Studium/Bachelor_BusinessAnalytics/BBA_Vertiefung/BigData/cleaned_reviews.parquet"

df.write.parquet(output_path, mode="overwrite")
print(f"Parquet erfolgreich gespeichert: {output_path}")



