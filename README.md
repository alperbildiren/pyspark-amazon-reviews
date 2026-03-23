# Kann eine Maschine Emotionen lesen?

## Sentiment-Analyse von 400.000 Amazon-Bewertungen mit PySpark

---

### Die Ausgangsfrage

Jeden Tag veröffentlichen Millionen von Menschen Bewertungen auf Amazon. In diesen Texten
stecken Emotionen: Begeisterung über ein großartiges Buch, Frust über ein defektes Ladegerät,
Enttäuschung über einen Film. Doch kann eine Maschine diese Emotionen automatisch erkennen?

In diesem Projekt gehen wir dieser Frage nach. Wir laden 400.000 Amazon-Bewertungen,
verarbeiten sie mit Apache Spark, verwandeln Wörter in mathematische Vektoren und trainieren
ein Machine-Learning-Modell, das vorhersagt: **Ist diese Bewertung positiv oder negativ?**

---

### Der Datensatz

| Eigenschaft | Wert |
|------------|------|
| Quelle | [Amazon Reviews – Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) |
| Datei | `test.csv` (kein Header) |
| Umfang | 400.000 Bewertungen |
| Spalten | Score (1=negativ, 2=positiv), Summary, Text |
| Balance | 200.000 negativ / 200.000 positiv (50/50) |

---

### Der Weg: Von Rohdaten zur Vorhersage

```
CSV-Datei (400K Zeilen)
    │
    ▼
┌─────────────────────────────┐
│  01 – Daten laden & erkunden │  ← Wie sehen die Daten aus?
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  02 – Datenbereinigung       │  ← Nulls, Duplikate, Normalisierung
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  03 – Tokenisierung & TF-IDF│  ← Wörter werden zu Zahlen
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  04 – Sentiment-Klassifikation│ ← Logistic Regression lernt
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  05 – Visualisierung         │  ← Was hat das Modell gelernt?
└─────────────────────────────┘
              ▼
        Parquet-Dateien
   (bereinigt, TF-IDF, Vorhersagen)
```

---

### Die Ergebnisse

| Metrik | Wert | Bedeutung |
|--------|------|-----------|
| Accuracy | **84,7%** | 5 von 6 Bewertungen werden korrekt erkannt |
| AUC-ROC | **0,92** | Das Modell trennt positiv/negativ deutlich besser als Zufall |
| F1-Score | **84,7%** | Precision und Recall sind ausgewogen |

Das Modell hat gelernt, dass Wörter wie *great*, *love* und *best* auf positive Bewertungen
hinweisen, während *money*, *bought* und *product* häufiger in negativen Bewertungen vorkommen.
TF-IDF sorgt dafür, dass allgegenwärtige Wörter wie *book* oder *one* heruntergewichtet werden.

---

### Was bedeutet das?

Mit einer einfachen Logistic Regression und TF-IDF-Features erreichen wir bereits **84,7% Accuracy**.
Das zeigt: Selbst ein einfaches Modell kann menschliche Emotionen in Text erkennen — nicht perfekt,
aber signifikant besser als Zufall (50%). Die ROC-Kurve mit AUC = 0,92 bestätigt, dass das Modell
die Trennung zwischen positiv und negativ robust gelernt hat.

Die verbleibenden ~15% Fehler entstehen dort, wo auch Menschen unsicher wären: bei sarkastischen,
gemischten oder neutral formulierten Bewertungen.

---

### Mögliche Verbesserungen

| Ansatz | Erwartete Verbesserung |
|--------|----------------------|
| N-Gramme (Bi-/Trigramme) | Kontextbezug: "not good" ≠ "good" |
| Word2Vec / GloVe | Semantische Ähnlichkeiten erfassen |
| Deep Learning (LSTM, BERT) | Satzstruktur und Kontext verstehen |
| Mehr Daten (Train-Set) | ~3,6 Mio. Bewertungen verfügbar |

---

### Projektstruktur

```
PYSPARK_AMAZON_PROJECT/
├── data/
│   ├── test.csv                         ← Rohdaten (400K Bewertungen)
│   └── train.csv                        ← Nicht verwendet (zu groß)
├── Notebooks/
│   ├── 01_data_loading.ipynb            ← Kapitel 1: Daten entdecken
│   ├── 02_data_cleaning.ipynb           ← Kapitel 2: Daten bereinigen
│   ├── 03_tokenization_tfidf.ipynb      ← Kapitel 3: Wörter → Zahlen
│   ├── 04_sentiment_classification.ipynb ← Kapitel 4: Modell trainieren
│   └── 05_visualization.ipynb           ← Kapitel 5: Ergebnisse zeigen
├── output/
│   ├── cleaned_reviews.parquet          ← Bereinigte Daten
│   ├── tfidf_features.parquet           ← TF-IDF-Vektoren
│   ├── predictions.parquet              ← Modell-Vorhersagen
│   ├── lr_model/                        ← Gespeichertes Modell
│   └── *.png                            ← Visualisierungen
├── src/                                 ← Hilfsfunktionen (optional)
└── README.md                            ← Diese Datei
```

---

### Technologie-Stack

| Werkzeug | Einsatz |
|----------|---------|
| PySpark 4.x | Datenverarbeitung, ML-Pipeline |
| Spark MLlib | Tokenizer, TF-IDF, Logistic Regression |
| Matplotlib | Diagramme und Visualisierungen |
| WordCloud | Wortwolken-Visualisierung |
| Parquet | Effizientes Speicherformat |
| Jupyter Notebook | Interaktive Dokumentation |

---

### Ausführung

```bash
# 1. Virtuelle Umgebung erstellen
python -m venv .venv
source .venv/bin/activate

# 2. Abhängigkeiten installieren
pip install pyspark matplotlib wordcloud

# 3. Notebooks der Reihe nach ausführen
#    01 → 02 → 03 → 04 → 05
```

---

*Projekt erstellt im Rahmen des Kurses — Thema 008 | Slot 1/2*
*Werkzeuge: PySpark, Spark MLlib, Python*
