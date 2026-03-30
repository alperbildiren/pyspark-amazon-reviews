
#4.1 Daten laden
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("AmazonReviews – Klassifikation") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

df = spark.read.parquet("/Users/zuky/Documents/Studium/Bachelor_BusinessAnalytics/BBA_Vertiefung/BigData/cleaned_reviews.parquet/tfidf_features.parquet")
print(f"Daten geladen: {df.count():,} Zeilen")
df.printSchema()

#4.2 Test split


train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Training:  {train_df.count():,} Zeilen")
print(f"Test:      {test_df.count():,} Zeilen")
print("\nLabel-Verteilung (Training):")
train_df.groupBy("label").count().orderBy("label").show()


#4.3 Logistics Regression trainieren

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="tfidf_features", labelCol="label", maxIter=20, regParam=0.1)

print("Training gestartet...")
lr_model = lr.fit(train_df)
print("Training abgeschlossen!")
print(f"Accuracy (Training):  {lr_model.summary.accuracy:.4f}")
print(f"Area under ROC:       {lr_model.summary.areaUnderROC:.4f}")


#4.4 Vorhersage auf Testdaten – GECACHT speichern

predictions = lr_model.transform(test_df)
predictions.cache()  # ← DAS ist der wichtigste Fix
predictions.count()  # einmal materialisieren

predictions.select("label", "prediction", "probability", "text").show(10, truncate=60)


#4.5 Evaluation


from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

binary_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                            metricName="areaUnderROC")
auc = binary_eval.evaluate(predictions)

multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
precision = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"})
recall = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})
f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})

print("=" * 45)
print("  EVALUATIONS-ERGEBNISSE (Testdaten)")
print("=" * 45)
print(f"  Accuracy:           {accuracy:.4f}")
print(f"  Precision (gew.):   {precision:.4f}")
print(f"  Recall (gew.):      {recall:.4f}")
print(f"  F1-Score (gew.):    {f1:.4f}")
print(f"  AUC-ROC:            {auc:.4f}")
print("=" * 45)



#4.6 Confusion Matrix

import matplotlib.pyplot as plt
import numpy as np

cm_data = predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").toPandas()
cm = np.zeros((2, 2), dtype=int)
for _, row in cm_data.iterrows():
    cm[int(row["label"]), int(row["prediction"])] = int(row["count"])

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
labels = ["Negativ (0)", "Positiv (1)"]
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel("Vorhersage"); ax.set_ylabel("Tatsächlich")
ax.set_title("Confusion Matrix – Logistic Regression")
for i in range(2):
    for j in range(2):
        color = "white" if cm[i,j] > cm.max()/2 else "black"
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=18, fontweight="bold", color=color)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.close()

print(f"True Negative:  {cm[0,0]:,}")
print(f"False Positive: {cm[0,1]:,}")
print(f"False Negative: {cm[1,0]:,}")
print(f"True Positive:  {cm[1,1]:,}")



#4.7 Beispielvorhersagen



from pyspark.sql.functions import col

print("=== KORREKTE Vorhersagen ===")
predictions.filter(col("label") == col("prediction")) \
    .select("label", "prediction", "text").show(5, truncate=80)

print("=== FALSCHE Vorhersagen ===")
predictions.filter(col("label") != col("prediction")) \
    .select("label", "prediction", "text").show(5, truncate=80)


#4.8 Ergebnisse speichern


from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

extract_prob = udf(lambda prob: float(prob[1]), FloatType())

output_path = "/Users/zuky/Documents/Studium/Bachelor_BusinessAnalytics/BBA_Vertiefung/BigData/predictions.parquet"
predictions.select(
    "label",
    "prediction",
    "text",
    extract_prob("probability").alias("prob_positive")
).write.parquet(output_path, mode="overwrite")
print(f"Vorhersagen gespeichert: {output_path}")

model_path = "/Users/zuky/Documents/Studium/Bachelor_BusinessAnalytics/BBA_Vertiefung/BigData/lr_model"
lr_model.write().overwrite().save(model_path)
print(f"Modell gespeichert: {model_path}")