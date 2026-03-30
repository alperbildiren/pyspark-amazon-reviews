# Kann eine Maschine Emotionen lesen?

## Sentiment-Analyse von Amazon-Bewertungen mit PySpark

---

### Die Ausgangsfrage

Jeden Tag veröffentlichen Millionen von Menschen Bewertungen auf Amazon.
In diesen Texten stecken Emotionen: Begeisterung, Frust, Enttäuschung, Freude.
Doch kann eine Maschine diese Emotionen automatisch erkennen?

In diesem Projekt gehen wir dieser Frage nach. Wir arbeiten mit dem
**originalen Trainingsdatensatz** von 3,6 Millionen Amazon-Bewertungen,
entnehmen eine repräsentative Stichprobe und bauen eine vollständige
NLP-Pipeline mit Apache Spark — von der Rohdatenanalyse bis zur Vorhersage.

---

### Der Datensatz

| Eigenschaft | Wert |
|------------|------|
| Quelle | [Amazon Reviews – Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) |
| Originaldatei | `train.csv` (3,6 Mio. Zeilen, 1,5 GB) |
| Stichprobe | 400.000 zufällige Bewertungen |
| Spalten | Score (1=negativ, 2=positiv), Summary, Text |

**Warum eine Stichprobe?** Der vollständige Datensatz mit 3,6 Mio. Zeilen würde auf
einem einzelnen Rechner sehr lange dauern. Eine zufällige Stichprobe von 400.000 Zeilen
ist statistisch repräsentativ und übertrifft die Mindestanforderung von 100.000 um das Vierfache.

---

### Der Weg: Von Rohdaten zur Vorhersage

```
train.csv (3,6 Mio. Zeilen)
    │
    ▼  Zufällige Stichprobe
┌─────────────────────────────────┐
│  01 – Daten laden & erkunden     │  ← Wie sieht die echte Verteilung aus?
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  02 – Datenbereinigung           │  ← Nulls, Duplikate, Balancierung
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  03 – Tokenisierung & TF-IDF    │  ← Wörter werden zu Zahlen
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  04 – Sentiment-Klassifikation   │  ← Logistic Regression lernt
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  05 – Visualisierung             │  ← Was hat das Modell gelernt?
└─────────────────────────────────┘
               ▼
         Parquet-Dateien
```

---

### Technologie-Stack

| Werkzeug | Einsatz |
|----------|---------|
| PySpark 4.x | Datenverarbeitung, ML-Pipeline |
| Spark MLlib | Tokenizer, TF-IDF, Logistic Regression |
| Matplotlib | Diagramme und Visualisierungen |
| WordCloud | Wortwolken-Visualisierung |
| Streamlit + Plotly | Interaktives Dashboard |
| Parquet | Effizientes Speicherformat |

---

### Projektstruktur

```
PYSPARK_AMAZON_PROJECT/
├── data/
│   ├── train.csv                         ← Originaldaten (3,6 Mio.)
│   └── test.csv                          ← Nicht verwendet
├── Notebooks/
│   ├── 01_data_loading.ipynb             ← Kapitel 1: Daten entdecken
│   ├── 02_data_cleaning.ipynb            ← Kapitel 2: Daten bereinigen
│   ├── 03_tokenization_tfidf.ipynb       ← Kapitel 3: Wörter → Zahlen
│   ├── 04_sentiment_classification.ipynb  ← Kapitel 4: Modell trainieren
│   └── 05_visualization.ipynb            ← Kapitel 5: Ergebnisse zeigen
├── output/
│   ├── cleaned_reviews.parquet           ← Bereinigte Daten
│   ├── tfidf_features.parquet            ← TF-IDF-Vektoren
│   ├── predictions.parquet               ← Modell-Vorhersagen
│   └── *.png                             ← Visualisierungen
├── dashboard.py                          ← Streamlit-Dashboard
├── fix_parquet.py                        ← Hilfsskript
└── README.md
```

---

### Ausführung

```bash
# 1. Virtuelle Umgebung
python -m venv .venv
source .venv/bin/activate

# 2. Abhängigkeiten
pip install pyspark matplotlib wordcloud streamlit plotly scikit-learn

# 3. Notebooks der Reihe nach ausführen: 01 → 02 → 03 → 04 → 05

# 4. Dashboard starten
streamlit run dashboard.py
```

---

*Projekt erstellt im Rahmen des Kurses — Thema 008 | Slot 1/2*
