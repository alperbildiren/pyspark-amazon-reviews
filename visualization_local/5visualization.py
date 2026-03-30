#5.1 Setup

from pyspark.sql import SparkSession
import os

spark = SparkSession.builder \
    .appName("AmazonReviews – Visualisierung") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

df_clean = spark.read.parquet("/Users/zuky/Documents/Studium/Bachelor_BusinessAnalytics/BBA_Vertiefung/BigData/cleaned_reviews.parquet")
df_preds = spark.read.parquet("/Users/zuky/Documents/Studium/Bachelor_BusinessAnalytics/BBA_Vertiefung/BigData/predictions.parquet")

print(f"Bereinigte Daten: {df_clean.count():,}")
print(f"Vorhersagen:      {df_preds.count():,}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'font.size': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold'})

OUTPUT = "/Users/zuky/Documents/Studium/Bachelor_BusinessAnalytics/BBA_Vertiefung/BigData"
os.makedirs(OUTPUT, exist_ok=True)


#5.2 Balancierter Datensatz

from pyspark.sql.functions import col, count as spark_count

label_dist = df_clean.groupBy("label").agg(spark_count("*").alias("count")).orderBy("label").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ["#F09595", "#5DCAA5"]

bars = axes[0].bar(["Negativ (0)", "Positiv (1)"], label_dist["count"],
                   color=colors, edgecolor=["#A32D2D", "#0F6E56"])
for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x()+bar.get_width()/2., h, f'{int(h):,}', ha='center', va='bottom')
axes[0].set_title("Sentiment-Verteilung (nach Balancierung)")
axes[0].set_ylabel("Anzahl")

axes[1].pie(label_dist["count"], labels=["Negativ","Positiv"], colors=colors,
            autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title("Anteil")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/01_sentiment_verteilung.png", dpi=150, bbox_inches='tight')
plt.close()  # ← statt plt.show()
print("✓ 01_sentiment_verteilung.png gespeichert")


#5.3 Textlängen nach Sentiment

from pyspark.sql.functions import length

df_len = df_clean.withColumn("text_length", length(col("text")))
pos_len = df_len.filter(col("label")==1).select("text_length").toPandas()
neg_len = df_len.filter(col("label")==0).select("text_length").toPandas()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(neg_len["text_length"].dropna(), bins=50, alpha=0.6, color="#F09595", edgecolor="#A32D2D", label="Negativ", range=(0,3000))
ax.hist(pos_len["text_length"].dropna(), bins=50, alpha=0.6, color="#5DCAA5", edgecolor="#0F6E56", label="Positiv", range=(0,3000))
ax.set_xlabel("Textlänge (Zeichen)"); ax.set_ylabel("Anzahl")
ax.set_title("Textlängen nach Sentiment"); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}/02_textlaenge_sentiment.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 02_textlaenge_sentiment.png gespeichert")


#5.4 Top Wörter

from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import explode

tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
df_filt = remover.transform(tokenizer.transform(df_clean))
df_words = df_filt.select("label", explode("filtered").alias("word"))

top_pos = df_words.filter(col("label")==1).groupBy("word").agg(spark_count("*").alias("count")).orderBy(col("count").desc()).limit(20).toPandas()
top_neg = df_words.filter(col("label")==0).groupBy("word").agg(spark_count("*").alias("count")).orderBy(col("count").desc()).limit(20).toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].barh(top_pos["word"][::-1], top_pos["count"][::-1], color="#5DCAA5", edgecolor="#0F6E56")
axes[0].set_title("Top 20 – POSITIV"); axes[0].set_xlabel("Häufigkeit")
axes[1].barh(top_neg["word"][::-1], top_neg["count"][::-1], color="#F09595", edgecolor="#A32D2D")
axes[1].set_title("Top 20 – NEGATIV"); axes[1].set_xlabel("Häufigkeit")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/03_top_woerter.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 03_top_woerter.png gespeichert")


#5.5 Wordcloud

from wordcloud import WordCloud

pos_freq = dict(zip(top_pos["word"], top_pos["count"]))
neg_freq = dict(zip(top_neg["word"], top_neg["count"]))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
wc_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens", max_words=80).generate_from_frequencies(pos_freq)
axes[0].imshow(wc_pos, interpolation="bilinear"); axes[0].set_title("POSITIV", fontsize=16, fontweight='bold'); axes[0].axis("off")
wc_neg = WordCloud(width=800, height=400, background_color="white", colormap="Reds", max_words=80).generate_from_frequencies(neg_freq)
axes[1].imshow(wc_neg, interpolation="bilinear"); axes[1].set_title("NEGATIV", fontsize=16, fontweight='bold'); axes[1].axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/04_wordcloud.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 04_wordcloud.png gespeichert")


#5.6 Confusion Matrix

cm_data = df_preds.groupBy("label","prediction").count().orderBy("label","prediction").toPandas()
cm = np.zeros((2,2), dtype=int)
for _, row in cm_data.iterrows():
    cm[int(row["label"]), int(row["prediction"])] = int(row["count"])
cm_pct = cm / cm.sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels_cm = ["Negativ (0)", "Positiv (1)"]

for idx, (data, fmt, title) in enumerate([(cm, "{:,}", "absolut"), (cm_pct, "{:.1f}%", "prozentual")]):
    im = axes[idx].imshow(data, cmap="Blues")
    axes[idx].set_xticks([0,1]); axes[idx].set_yticks([0,1])
    axes[idx].set_xticklabels(labels_cm); axes[idx].set_yticklabels(labels_cm)
    axes[idx].set_xlabel("Vorhersage"); axes[idx].set_ylabel("Tatsächlich")
    axes[idx].set_title(f"Confusion Matrix ({title})")
    for i in range(2):
        for j in range(2):
            color = "white" if data[i,j] > data.max()/2 else "black"
            axes[idx].text(j, i, fmt.format(data[i,j]), ha="center", va="center", fontsize=16, fontweight="bold", color=color)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/05_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 05_confusion_matrix.png gespeichert")


#5.7 ROC Kurve

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from sklearn.metrics import roc_curve, auc

extract_prob = udf(lambda prob: float(prob[1]), FloatType())
roc_pd = df_preds.withColumn("prob_positive", extract_prob(col("probability"))).select("label","prob_positive").toPandas()

fpr, tpr, _ = roc_curve(roc_pd["label"], roc_pd["prob_positive"])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, color="#378ADD", lw=2.5, label=f"Logistic Regression (AUC = {roc_auc:.4f})")
ax.plot([0,1],[0,1], color="#B4B2A9", lw=1.5, linestyle="--", label="Zufall (AUC = 0.5)")
ax.fill_between(fpr, tpr, alpha=0.15, color="#85B7EB")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC-Kurve"); ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/06_roc_kurve.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 06_roc_kurve.png gespeichert")


#5.8 Wahrscheinlichkeitsverteilung

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(roc_pd[roc_pd["label"]==0]["prob_positive"], bins=50, alpha=0.6, color="#F09595", edgecolor="#A32D2D", label="Negativ")
ax.hist(roc_pd[roc_pd["label"]==1]["prob_positive"], bins=50, alpha=0.6, color="#5DCAA5", edgecolor="#0F6E56", label="Positiv")
ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, label="Schwellenwert")
ax.set_xlabel("Wahrscheinlichkeit (positiv)"); ax.set_ylabel("Anzahl")
ax.set_title("Verteilung der Vorhersage-Wahrscheinlichkeiten"); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}/08_wahrscheinlichkeiten.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 08_wahrscheinlichkeiten.png gespeichert")


#5.9 Metriken Dashboard

total_p = cm.sum()
acc = (cm[0,0]+cm[1,1])/total_p
prec_pos = cm[1,1]/(cm[0,1]+cm[1,1]) if (cm[0,1]+cm[1,1])>0 else 0
rec_pos = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1])>0 else 0
f1_v = 2*prec_pos*rec_pos/(prec_pos+rec_pos) if (prec_pos+rec_pos)>0 else 0

metrics = {'Accuracy': acc, 'AUC-ROC': roc_auc, 'Precision (pos)': prec_pos, 'Recall (pos)': rec_pos, 'F1 (pos)': f1_v}

fig, ax = plt.subplots(figsize=(8, 4))
names = list(metrics.keys()); values = list(metrics.values())
bars = ax.barh(names[::-1], values[::-1], color="#378ADD", height=0.5)
for bar in bars:
    w = bar.get_width()
    ax.text(w+0.01, bar.get_y()+bar.get_height()/2., f'{w:.4f}', ha='left', va='center', fontsize=11, fontweight='bold')
ax.set_xlim([0,1.15]); ax.set_title("Evaluations-Metriken")
ax.axvline(x=0.5, color="#B4B2A9", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/07_metriken_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 07_metriken_dashboard.png gespeichert")


#5.10 Einzelne Bewertungen

import pandas as pd
from pyspark.sql.functions import col, udf, length
from pyspark.sql.types import FloatType

extract_prob = udf(lambda prob: float(prob[1]), FloatType())
df_with_prob = df_preds.withColumn("prob_positive", extract_prob(col("probability")))

examples_pos = df_with_prob.filter(
    (col("label") == 1) & (col("prediction") == 1) & (col("prob_positive") > 0.85)
).withColumn("tlen", length("text")).filter("tlen BETWEEN 80 AND 300") \
 .select("text", "prob_positive", "label").limit(10).toPandas()

examples_neg = df_with_prob.filter(
    (col("label") == 0) & (col("prediction") == 0) & (col("prob_positive") < 0.15)
).withColumn("tlen", length("text")).filter("tlen BETWEEN 80 AND 300") \
 .select("text", "prob_positive", "label").limit(10).toPandas()

examples_pos["sentiment"] = "Positiv"
examples_neg["sentiment"] = "Negativ"
examples = pd.concat([examples_neg, examples_pos]).reset_index(drop=True)
examples["short_text"] = examples["text"].apply(lambda x: x[:75] + "..." if len(x) > 75 else x)

fig, ax = plt.subplots(figsize=(12, 8))
colors_bar = ["#F09595" if s == "Negativ" else "#5DCAA5" for s in examples["sentiment"]]
bars = ax.barh(range(len(examples)), examples["prob_positive"], color=colors_bar, edgecolor="white")
ax.set_yticks(range(len(examples)))
ax.set_yticklabels(examples["short_text"], fontsize=9)
ax.set_xlabel("Wahrscheinlichkeit (positiv)")
ax.set_title("Einzelne Bewertungen und ihre Modell-Vorhersage")
ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1, label="Schwellenwert")
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT}/09_einzelne_bewertungen.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 09_einzelne_bewertungen.png gespeichert")


#5.11 Gespeicherte Dateien auflisten

print("=" * 55)
print("  GESPEICHERTE DATEIEN")
print("=" * 55)
for f in sorted(os.listdir(OUTPUT)):
    fp = os.path.join(OUTPUT, f)
    if os.path.isfile(fp) and f.endswith(".png"):
        print(f"  {f:45s} {os.path.getsize(fp)/(1024*1024):>6.2f} MB")
print("=" * 55)
print("Fertig!")