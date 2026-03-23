"""
Amazon Reviews Sentiment-Analyse — Dashboard
=============================================
Starten mit: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# ─────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Reviews – Sentiment-Analyse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

POS_COLOR = "#2ecc71"
NEG_COLOR = "#e74c3c"
BLUE = "#3498db"
GRAY = "#95a5a6"

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_PATH, "output")


# ─────────────────────────────────────────────
# Daten laden
# ─────────────────────────────────────────────
@st.cache_data
def load_clean_data():
    path = os.path.join(OUTPUT_PATH, "cleaned_reviews.parquet")
    return pd.read_parquet(path)


@st.cache_data
def load_predictions():
    path = os.path.join(OUTPUT_PATH, "predictions.parquet")
    df = pd.read_parquet(path)

    # prob_positive extrahieren (verschiedene Formate abfangen)
    if "prob_positive" not in df.columns and "probability" in df.columns:
        try:
            sample = df["probability"].iloc[0]
            if hasattr(sample, "values"):
                df["prob_positive"] = df["probability"].apply(lambda x: float(x.values[1]))
            elif isinstance(sample, (list, np.ndarray)):
                df["prob_positive"] = df["probability"].apply(lambda x: float(x[1]))
            elif isinstance(sample, dict):
                df["prob_positive"] = df["probability"].apply(lambda x: float(x.get("values", [0, 0.5])[1]))
            else:
                df["prob_positive"] = 0.5
        except Exception:
            df["prob_positive"] = 0.5

    return df


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0;'>
        Kann eine Maschine Emotionen lesen?
    </h1>
    <p style='text-align: center; color: gray; font-size: 1.2em; margin-top: 0.2em;'>
        Sentiment-Analyse von 400.000 Amazon-Bewertungen mit PySpark
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Daten laden
try:
    df_clean = load_clean_data()
    df_preds = load_predictions()
    data_loaded = True
except Exception as e:
    st.error(f"Fehler beim Laden der Daten: {e}")
    st.info("Bitte zuerst die Notebooks 01–04 ausführen, um die Parquet-Dateien zu erzeugen.")
    data_loaded = False

if data_loaded:

    # ─────────────────────────────────────────
    # Sidebar
    # ─────────────────────────────────────────
    with st.sidebar:
        st.header("Projektübersicht")
        st.markdown(f"""
        - **Datensatz:** Amazon Reviews (Kaggle)
        - **Umfang:** {len(df_clean):,} Bewertungen
        - **Modell:** Logistic Regression
        - **Features:** TF-IDF (10.000 Dim.)
        - **Split:** 80% Train / 20% Test
        """)

        st.divider()
        st.header("Navigation")
        section = st.radio(
            "Abschnitt wählen:",
            [
                "📊 Datensatz-Übersicht",
                "📝 Textanalyse",
                "🤖 Modell-Ergebnisse",
                "🔍 Live-Explorer",
            ],
        )

    # ─────────────────────────────────────────
    # 1. DATENSATZ-ÜBERSICHT
    # ─────────────────────────────────────────
    if section == "📊 Datensatz-Übersicht":

        st.header("📊 Datensatz-Übersicht")

        total = len(df_clean)
        n_pos = len(df_clean[df_clean["label"] == 1])
        n_neg = len(df_clean[df_clean["label"] == 0])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gesamtbewertungen", f"{total:,}")
        col2.metric("Positiv", f"{n_pos:,}", f"{n_pos/total*100:.1f}%")
        col3.metric("Negativ", f"{n_neg:,}", f"{n_neg/total*100:.1f}%")
        col4.metric("Balance", "50 / 50", "Perfekt")

        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=["Negativ (0)", "Positiv (1)"],
                y=[n_neg, n_pos],
                marker_color=[NEG_COLOR, POS_COLOR],
                text=[f"{n_neg:,}", f"{n_pos:,}"],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Sentiment-Verteilung",
                yaxis_title="Anzahl",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            fig_pie = go.Figure(go.Pie(
                labels=["Negativ", "Positiv"],
                values=[n_neg, n_pos],
                marker=dict(colors=[NEG_COLOR, POS_COLOR]),
                hole=0.4,
                textinfo="percent+label",
            ))
            fig_pie.update_layout(
                title="Anteil positiv / negativ",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        st.subheader("Textlängen-Verteilung")
        df_clean["text_length"] = df_clean["text"].str.len()

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_clean[df_clean["label"] == 0]["text_length"],
            nbinsx=50, name="Negativ", marker_color=NEG_COLOR, opacity=0.6,
        ))
        fig_hist.add_trace(go.Histogram(
            x=df_clean[df_clean["label"] == 1]["text_length"],
            nbinsx=50, name="Positiv", marker_color=POS_COLOR, opacity=0.6,
        ))
        fig_hist.update_layout(
            barmode="overlay",
            xaxis_title="Textlänge (Zeichen)",
            yaxis_title="Anzahl",
            xaxis_range=[0, 3000],
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ─────────────────────────────────────────
    # 2. TEXTANALYSE
    # ─────────────────────────────────────────
    elif section == "📝 Textanalyse":

        st.header("📝 Textanalyse – Die Sprache der Emotionen")

        st.markdown("""
        Welche Wörter verraten die Stimmung? Nach Tokenisierung und StopWord-Entfernung
        zeigen sich klare Unterschiede zwischen positiven und negativen Bewertungen.
        """)

        from collections import Counter

        stop_words = set([
            "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
            "he", "him", "his", "she", "her", "it", "its", "they", "them",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between",
            "through", "during", "before", "after", "above", "below", "to",
            "from", "up", "down", "in", "out", "on", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "s", "t", "can",
            "will", "just", "don", "should", "now", "d", "ll", "m", "o",
            "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
            "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
            "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
        ])

        @st.cache_data
        def get_top_words(df, label, n=20):
            texts = df[df["label"] == label]["text"]
            words = []
            for text in texts:
                if isinstance(text, str):
                    words.extend([w for w in text.split() if w not in stop_words and len(w) > 2])
            counter = Counter(words)
            return pd.DataFrame(counter.most_common(n), columns=["word", "count"])

        top_pos = get_top_words(df_clean, 1, 20)
        top_neg = get_top_words(df_clean, 0, 20)

        c1, c2 = st.columns(2)

        with c1:
            fig_pos = go.Figure(go.Bar(
                y=top_pos["word"][::-1], x=top_pos["count"][::-1],
                orientation="h", marker_color=POS_COLOR,
            ))
            fig_pos.update_layout(
                title="Top 20 Wörter – POSITIV",
                xaxis_title="Häufigkeit", template="plotly_white", height=500,
            )
            st.plotly_chart(fig_pos, use_container_width=True)

        with c2:
            fig_neg = go.Figure(go.Bar(
                y=top_neg["word"][::-1], x=top_neg["count"][::-1],
                orientation="h", marker_color=NEG_COLOR,
            ))
            fig_neg.update_layout(
                title="Top 20 Wörter – NEGATIV",
                xaxis_title="Häufigkeit", template="plotly_white", height=500,
            )
            st.plotly_chart(fig_neg, use_container_width=True)

        st.divider()
        st.markdown("""
        **Erkenntnis:** Gemeinsame Wörter wie *book* und *one* erscheinen in beiden Listen —
        TF-IDF gewichtet sie herunter. Die Unterschiede sind aufschlussreich:
        - **Positiv:** *great, love, best, well* — Wörter der Begeisterung
        - **Negativ:** *money, bought, product, work* — Wörter der Enttäuschung
        """)

    # ─────────────────────────────────────────
    # 3. MODELL-ERGEBNISSE
    # ─────────────────────────────────────────
    elif section == "🤖 Modell-Ergebnisse":

        st.header("🤖 Modell-Ergebnisse – Logistic Regression")

        cm = pd.crosstab(df_preds["label"], df_preds["prediction"])
        tn = int(cm.iloc[0, 0]) if 0 in cm.columns else 0
        fp = int(cm.iloc[0, 1]) if 1 in cm.columns else 0
        fn = int(cm.iloc[1, 0]) if 0 in cm.columns else 0
        tp = int(cm.iloc[1, 1]) if 1 in cm.columns else 0

        total_preds = tn + fp + fn + tp
        accuracy = (tn + tp) / total_preds
        precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.1%}")
        col2.metric("Precision", f"{precision_pos:.1%}")
        col3.metric("Recall", f"{recall_pos:.1%}")
        col4.metric("F1-Score", f"{f1:.1%}")

        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            cm_array = np.array([[tn, fp], [fn, tp]])
            fig_cm = go.Figure(go.Heatmap(
                z=cm_array,
                x=["Negativ (0)", "Positiv (1)"],
                y=["Negativ (0)", "Positiv (1)"],
                text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
                texttemplate="%{text}",
                textfont=dict(size=18, color="white"),
                colorscale="Blues", showscale=False,
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Vorhersage", yaxis_title="Tatsächlich",
                yaxis=dict(autorange="reversed"),
                template="plotly_white", height=450,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            if "prob_positive" in df_preds.columns and df_preds["prob_positive"].nunique() > 1:
                from sklearn.metrics import roc_curve, auc

                fpr, tpr, _ = roc_curve(df_preds["label"], df_preds["prob_positive"])
                roc_auc = auc(fpr, tpr)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"LR (AUC = {roc_auc:.4f})",
                    line=dict(color=BLUE, width=2.5),
                    fill="tozeroy", fillcolor="rgba(52, 152, 219, 0.1)",
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    name="Zufall (AUC = 0.5)",
                    line=dict(color=GRAY, width=1.5, dash="dash"),
                ))
                fig_roc.update_layout(
                    title="ROC-Kurve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    template="plotly_white", height=450,
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("ROC-Kurve nicht verfügbar — Wahrscheinlichkeitsdaten fehlen.")

        st.divider()

        if "prob_positive" in df_preds.columns and df_preds["prob_positive"].nunique() > 1:
            st.subheader("Verteilung der Vorhersage-Wahrscheinlichkeiten")

            fig_prob = go.Figure()
            fig_prob.add_trace(go.Histogram(
                x=df_preds[df_preds["label"] == 0]["prob_positive"],
                nbinsx=50, name="Tatsächlich Negativ",
                marker_color=NEG_COLOR, opacity=0.6,
            ))
            fig_prob.add_trace(go.Histogram(
                x=df_preds[df_preds["label"] == 1]["prob_positive"],
                nbinsx=50, name="Tatsächlich Positiv",
                marker_color=POS_COLOR, opacity=0.6,
            ))
            fig_prob.add_vline(x=0.5, line_dash="dash", line_color="black",
                              annotation_text="Schwellenwert")
            fig_prob.update_layout(
                barmode="overlay",
                xaxis_title="Vorhergesagte Wahrscheinlichkeit (positiv)",
                yaxis_title="Anzahl",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

    # ─────────────────────────────────────────
    # 4. LIVE-EXPLORER
    # ─────────────────────────────────────────
    elif section == "🔍 Live-Explorer":

        st.header("🔍 Live-Explorer – Einzelne Vorhersagen untersuchen")

        c1, c2 = st.columns(2)
        with c1:
            filter_type = st.selectbox(
                "Anzeigen:",
                ["Alle", "Korrekte Vorhersagen", "Falsche Vorhersagen"]
            )
        with c2:
            filter_label = st.selectbox(
                "Sentiment:",
                ["Alle", "Nur Positiv", "Nur Negativ"]
            )

        df_show = df_preds.copy()

        if filter_type == "Korrekte Vorhersagen":
            df_show = df_show[df_show["label"] == df_show["prediction"]]
        elif filter_type == "Falsche Vorhersagen":
            df_show = df_show[df_show["label"] != df_show["prediction"]]

        if filter_label == "Nur Positiv":
            df_show = df_show[df_show["label"] == 1]
        elif filter_label == "Nur Negativ":
            df_show = df_show[df_show["label"] == 0]

        st.info(f"{len(df_show):,} Bewertungen gefunden")

        if len(df_show) > 0:
            sample_size = min(10, len(df_show))
            sample = df_show.sample(sample_size)

            for _, row in sample.iterrows():
                actual = "Positiv ✅" if row["label"] == 1 else "Negativ ❌"
                predicted = "Positiv" if row["prediction"] == 1.0 else "Negativ"
                correct = row["label"] == row["prediction"]
                icon = "✅" if correct else "⚠️"
                prob = row.get("prob_positive", 0.5)

                with st.container():
                    st.markdown(f"""
                    **{icon} Tatsächlich: {actual} | Vorhersage: {predicted}**
                    (Wahrscheinlichkeit positiv: {prob:.1%})
                    """)
                    st.text(str(row["text"])[:500])
                    st.divider()

            if st.button("🔄 Neue Stichprobe laden"):
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Sentiment-Analyse — PySpark + TF-IDF + Logistic Regression | Thema 008"
        "</p>",
        unsafe_allow_html=True,
    )
