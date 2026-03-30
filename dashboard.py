"""
═══════════════════════════════════════════════════════════
  Kann eine Maschine Emotionen lesen?
  Sentiment-Analyse von Amazon-Bewertungen mit PySpark
═══════════════════════════════════════════════════════════
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
    page_title="Kann eine Maschine Emotionen lesen?",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_PATH, "output")

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Serif+Display&display=swap');

    .stApp { background-color: #0E1117; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
    p, li, span, div { font-family: 'DM Sans', sans-serif !important; }

    .hero-container { text-align: center; padding: 2rem 1rem 1rem 1rem; }
    .hero-title {
        font-family: 'DM Serif Display', serif !important;
        font-size: 3.2rem; color: #F8FAFC; margin-bottom: 0.3rem; line-height: 1.2;
    }
    .hero-subtitle {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.15rem; color: #94A3B8; margin-bottom: 2rem;
    }
    .chapter-header {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.8rem; color: #F8FAFC;
        border-left: 4px solid #14B8A6; padding-left: 1rem; margin: 2rem 0 1rem 0;
    }
    .chapter-desc {
        font-family: 'DM Sans', sans-serif !important;
        color: #94A3B8; font-size: 1rem; margin-bottom: 1.5rem; padding-left: 1.4rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #1E293B; border-radius: 12px; padding: 1.5rem; text-align: center;
    }
    .stat-value {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem;
    }
    .stat-label {
        font-family: 'DM Sans', sans-serif !important;
        color: #64748B; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;
    }
    .insight-box {
        background: linear-gradient(135deg, #0D3B3B 0%, #0F172A 100%);
        border: 1px solid #14B8A6; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    }
    .insight-box p { color: #CBD5E1; margin: 0; font-size: 0.95rem; line-height: 1.6; }
    .insight-box strong { color: #14B8A6; }
    .warning-box {
        background: linear-gradient(135deg, #3B1A0D 0%, #0F172A 100%);
        border: 1px solid #F97316; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    }
    .warning-box p { color: #CBD5E1; margin: 0; font-size: 0.95rem; line-height: 1.6; }
    .warning-box strong { color: #F97316; }
    .story-divider { text-align: center; margin: 2.5rem 0; color: #334155; font-size: 1.5rem; letter-spacing: 8px; }
    .review-card {
        background: #1E293B; border-radius: 10px; padding: 1rem 1.2rem;
        margin: 0.5rem 0; border-left: 4px solid;
    }
    .review-card.positive { border-color: #10B981; }
    .review-card.negative { border-color: #EF4444; }
    .review-text { color: #E2E8F0; font-size: 0.9rem; line-height: 1.5; }
    .review-meta { color: #64748B; font-size: 0.8rem; margin-top: 0.4rem; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B; border-radius: 8px; padding: 8px 20px; color: #94A3B8;
    }
    .stTabs [aria-selected="true"] { background-color: #14B8A6 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly defaults
# ─────────────────────────────────────────────
PLOT_BG = "#0E1117"
CARD_BG = "#1E293B"
TEAL = "#14B8A6"
RED = "#EF4444"
GREEN = "#10B981"
BLUE = "#3B82F6"
AMBER = "#F59E0B"
GRAY = "#64748B"
TEXT_COLOR = "#E2E8F0"

plotly_layout = dict(
    paper_bgcolor=PLOT_BG, plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family="DM Sans"),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
)

# ─────────────────────────────────────────────
# Daten laden
# ─────────────────────────────────────────────
@st.cache_data
def load_clean_data():
    return pd.read_parquet(os.path.join(OUTPUT_PATH, "cleaned_reviews.parquet"))

@st.cache_data
def load_predictions():
    df = pd.read_parquet(os.path.join(OUTPUT_PATH, "predictions.parquet"))
    if "prob_positive" not in df.columns and "probability" in df.columns:
        try:
            sample = df["probability"].iloc[0]
            if hasattr(sample, "values"):
                df["prob_positive"] = df["probability"].apply(lambda x: float(x.values[1]))
            elif isinstance(sample, (list, np.ndarray)):
                df["prob_positive"] = df["probability"].apply(lambda x: float(x[1]))
            else:
                df["prob_positive"] = 0.5
        except Exception:
            df["prob_positive"] = 0.5
    return df

try:
    df_clean = load_clean_data()
    df_preds = load_predictions()
except Exception as e:
    st.error(f"Fehler: {e}")
    st.info("Bitte zuerst die Notebooks 01–04 ausführen.")
    st.stop()

# ─────────────────────────────────────────────
# Grundlegende Berechnungen
# ─────────────────────────────────────────────
total = len(df_clean)
n_pos = len(df_clean[df_clean["label"] == 1])
n_neg = len(df_clean[df_clean["label"] == 0])

tp = len(df_preds[(df_preds["label"] == 1) & (df_preds["prediction"] == 1.0)])
tn = len(df_preds[(df_preds["label"] == 0) & (df_preds["prediction"] == 0.0)])
fp = len(df_preds[(df_preds["label"] == 0) & (df_preds["prediction"] == 1.0)])
fn = len(df_preds[(df_preds["label"] == 1) & (df_preds["prediction"] == 0.0)])
total_preds = tp + tn + fp + fn
accuracy = (tp + tn) / total_preds if total_preds > 0 else 0
precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

has_probs = "prob_positive" in df_preds.columns and df_preds["prob_positive"].nunique() > 1
if has_probs:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(df_preds["label"], df_preds["prob_positive"])
    roc_auc = auc(fpr, tpr)
else:
    roc_auc = 0


# ═══════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Kann eine Maschine Emotionen lesen?</div>
    <div class="hero-subtitle">Eine Reise durch 3,6 Millionen Amazon-Bewertungen — von Rohdaten zur Vorhersage</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col_obj, (val, label, color) in zip([c1, c2, c3, c4], [
    ("3,6 Mio.", "Originaldaten", TEAL),
    (f"{total:,}", "Stichprobe", BLUE),
    (f"{accuracy:.1%}", "Accuracy", GREEN),
    (f"{roc_auc:.2f}", "AUC-ROC", AMBER),
]):
    with col_obj:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:{color}">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="story-divider">· · ·</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# KAPITEL 1: DATENSATZ
# ═══════════════════════════════════════════════
st.markdown('<div class="chapter-header">Kapitel 1 — Der Datensatz</div>', unsafe_allow_html=True)
st.markdown('<div class="chapter-desc">Aus 3,6 Millionen Amazon-Bewertungen haben wir eine repräsentative Stichprobe gezogen. Wie sieht die natürliche Verteilung aus?</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])
with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Negativ", "Positiv"], y=[n_neg, n_pos], marker_color=[RED, GREEN],
                         text=[f"{n_neg:,}", f"{n_pos:,}"], textposition="outside",
                         textfont=dict(size=16, color=TEXT_COLOR), width=0.5))
    fig.update_layout(**plotly_layout, title=dict(text="Sentiment-Verteilung", font=dict(size=18)),
                      yaxis_title="Anzahl", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    df_clean["text_length"] = df_clean["text"].str.len()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_clean[df_clean["label"]==0]["text_length"], nbinsx=40, name="Negativ", marker_color=RED, opacity=0.6))
    fig.add_trace(go.Histogram(x=df_clean[df_clean["label"]==1]["text_length"], nbinsx=40, name="Positiv", marker_color=GREEN, opacity=0.6))
    fig.update_layout(**plotly_layout, title=dict(text="Textlängen nach Sentiment", font=dict(size=18)),
                      barmode="overlay", xaxis_title="Zeichen", yaxis_title="Anzahl", xaxis_range=[0,2000],
                      legend=dict(x=0.7, y=0.95, bgcolor="rgba(0,0,0,0)"), height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""<div class="insight-box"><p><strong>Erkenntnis:</strong> Die Verteilung ist nahezu perfekt ausgewogen —
{n_neg/total*100:.1f}% negativ und {n_pos/total*100:.1f}% positiv. In realen Datensätzen ist das selten — unser Modell hat faire Lernbedingungen.</p></div>""", unsafe_allow_html=True)

st.markdown('<div class="story-divider">· · ·</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# KAPITEL 2: SPRACHE DER EMOTIONEN
# ═══════════════════════════════════════════════
st.markdown('<div class="chapter-header">Kapitel 2 — Die Sprache der Emotionen</div>', unsafe_allow_html=True)
st.markdown('<div class="chapter-desc">Nach Tokenisierung und StopWord-Entfernung: Welche Wörter verraten die Stimmung?</div>', unsafe_allow_html=True)

from collections import Counter
stop_words = set(["i","me","my","myself","we","our","ours","you","your","he","him","his","she","her","it","its","they","them","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now","d","ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn"])

@st.cache_data
def get_top_words(df, label, n=20):
    texts = df[df["label"]==label]["text"]
    words = []
    for t in texts:
        if isinstance(t, str):
            words.extend([w for w in t.split() if w not in stop_words and len(w)>2])
    return pd.DataFrame(Counter(words).most_common(n), columns=["word","count"])

top_pos = get_top_words(df_clean, 1, 20)
top_neg = get_top_words(df_clean, 0, 20)

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Bar(y=top_pos["word"][::-1], x=top_pos["count"][::-1], orientation="h", marker_color=GREEN,
                           text=top_pos["count"][::-1], textposition="outside", textfont=dict(color=TEXT_COLOR,size=11)))
    fig.update_layout(**plotly_layout, title=dict(text="✅ Top 20 — Positiv", font=dict(size=16)), height=550, xaxis_title="Häufigkeit")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = go.Figure(go.Bar(y=top_neg["word"][::-1], x=top_neg["count"][::-1], orientation="h", marker_color=RED,
                           text=top_neg["count"][::-1], textposition="outside", textfont=dict(color=TEXT_COLOR,size=11)))
    fig.update_layout(**plotly_layout, title=dict(text="❌ Top 20 — Negativ", font=dict(size=16)), height=550, xaxis_title="Häufigkeit")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""<div class="insight-box"><p><strong>Erkenntnis:</strong> Die Sprache der Zufriedenheit (<strong>great, love, best, well</strong>) unterscheidet sich klar von der Sprache der Enttäuschung (<strong>money, bought, product, work</strong>). Gemeinsame Wörter wie <em>book</em> werden durch TF-IDF heruntergewichtet.</p></div>""", unsafe_allow_html=True)

st.markdown('<div class="story-divider">· · ·</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# KAPITEL 3: MODELL
# ═══════════════════════════════════════════════
st.markdown('<div class="chapter-header">Kapitel 3 — Das Urteil der Maschine</div>', unsafe_allow_html=True)
st.markdown('<div class="chapter-desc">Logistic Regression auf TF-IDF-Vektoren — wie gut trennt das Modell positiv von negativ?</div>', unsafe_allow_html=True)

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
for col_obj, (name, val, color) in zip([mc1,mc2,mc3,mc4,mc5], [
    ("Accuracy", f"{accuracy:.1%}", TEAL), ("Precision", f"{precision_pos:.1%}", BLUE),
    ("Recall", f"{recall_pos:.1%}", GREEN), ("F1-Score", f"{f1:.1%}", AMBER),
    ("AUC-ROC", f"{roc_auc:.4f}", "#A78BFA"),
]):
    with col_obj:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:{color};font-size:1.8rem">{val}</div><div class="stat-label">{name}</div></div>', unsafe_allow_html=True)

st.markdown("", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    cm = np.array([[tn,fp],[fn,tp]])
    fig = go.Figure(go.Heatmap(z=cm, x=["Negativ","Positiv"], y=["Negativ","Positiv"],
                               text=[[f"{tn:,}",f"{fp:,}"],[f"{fn:,}",f"{tp:,}"]],
                               texttemplate="%{text}", textfont=dict(size=20,color="white"),
                               colorscale=[[0,"#1E293B"],[1,TEAL]], showscale=False))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="DM Sans"),
        margin=dict(l=40, r=40, t=50, b=40),
        title=dict(text="Confusion Matrix", font=dict(size=16)),
        xaxis_title="Vorhersage", yaxis_title="Tatsächlich",
        yaxis=dict(autorange="reversed", gridcolor="#334155"),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if has_probs:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"LR (AUC={roc_auc:.4f})",
                                 line=dict(color=TEAL,width=3), fill="tozeroy", fillcolor="rgba(20,184,166,0.1)"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Zufall", line=dict(color=GRAY,width=1.5,dash="dash")))
        fig.update_layout(**plotly_layout, title=dict(text="ROC-Kurve",font=dict(size=16)),
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          legend=dict(x=0.55,y=0.1,bgcolor="rgba(0,0,0,0)"), height=420)
        st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""<div class="insight-box"><p><strong>Ergebnis:</strong> 5 von 6 Bewertungen werden korrekt erkannt. Die Fehler sind symmetrisch — <strong>{fp:,}</strong> falsch positive und <strong>{fn:,}</strong> falsch negative. Das Modell bevorzugt keine Klasse.</p></div>""", unsafe_allow_html=True)

if has_probs:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_preds[df_preds["label"]==0]["prob_positive"], nbinsx=50, name="Tatsächlich Negativ", marker_color=RED, opacity=0.65))
    fig.add_trace(go.Histogram(x=df_preds[df_preds["label"]==1]["prob_positive"], nbinsx=50, name="Tatsächlich Positiv", marker_color=GREEN, opacity=0.65))
    fig.add_vline(x=0.5, line_dash="dash", line_color="white", line_width=1.5, annotation_text="Schwelle", annotation_font_color="white")
    fig.update_layout(**plotly_layout, title=dict(text="Wie sicher ist das Modell?",font=dict(size=16)),
                      barmode="overlay", xaxis_title="Wahrscheinlichkeit (positiv)", yaxis_title="Anzahl",
                      legend=dict(x=0.01,y=0.95,bgcolor="rgba(0,0,0,0)"), height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box"><p><strong>Erkenntnis:</strong> Die zwei Klassen sind deutlich getrennt — die meisten Vorhersagen liegen weit von der 0,5-Schwelle entfernt. Das Modell ist bei den meisten Bewertungen <strong>sehr sicher</strong>.</p></div>""", unsafe_allow_html=True)

st.markdown('<div class="story-divider">· · ·</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# KAPITEL 4: ECHTE BEWERTUNGEN
# ═══════════════════════════════════════════════
st.markdown('<div class="chapter-header">Kapitel 4 — Echte Bewertungen unter der Lupe</div>', unsafe_allow_html=True)
st.markdown('<div class="chapter-desc">Was sagt das Modell zu konkreten Amazon-Reviews? Wo liegt es richtig — und wo irrt es sich?</div>', unsafe_allow_html=True)

if has_probs:
    neg_ex = df_preds[(df_preds["label"]==0)&(df_preds["prediction"]==0.0)&(df_preds["prob_positive"]<0.15)].copy()
    neg_ex["tlen"] = neg_ex["text"].str.len()
    neg_ex = neg_ex[(neg_ex["tlen"]>=80)&(neg_ex["tlen"]<=350)].head(6)

    pos_ex = df_preds[(df_preds["label"]==1)&(df_preds["prediction"]==1.0)&(df_preds["prob_positive"]>0.85)].copy()
    pos_ex["tlen"] = pos_ex["text"].str.len()
    pos_ex = pos_ex[(pos_ex["tlen"]>=80)&(pos_ex["tlen"]<=350)].head(6)

    tab1, tab2, tab3 = st.tabs(["✅ Korrekt Positiv", "❌ Korrekt Negativ", "⚠️ Falsche Vorhersagen"])

    with tab1:
        for _, row in pos_ex.iterrows():
            st.markdown(f"""<div class="review-card positive">
                <div class="review-text">{str(row['text'])[:300]}{'...' if len(str(row['text']))>300 else ''}</div>
                <div class="review-meta">Konfidenz: {row.get('prob_positive',0.5):.0%} positiv</div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        for _, row in neg_ex.iterrows():
            st.markdown(f"""<div class="review-card negative">
                <div class="review-text">{str(row['text'])[:300]}{'...' if len(str(row['text']))>300 else ''}</div>
                <div class="review-meta">Konfidenz: {(1-row.get('prob_positive',0.5)):.0%} negativ</div>
            </div>""", unsafe_allow_html=True)

    with tab3:
        wrong = df_preds[df_preds["label"]!=df_preds["prediction"]].copy()
        wrong["tlen"] = wrong["text"].str.len()
        wrong = wrong[(wrong["tlen"]>=80)&(wrong["tlen"]<=400)].head(6)
        for _, row in wrong.iterrows():
            actual = "Positiv" if row["label"]==1 else "Negativ"
            predicted = "Positiv" if row["prediction"]==1.0 else "Negativ"
            card_class = "positive" if row["label"]==1 else "negative"
            st.markdown(f"""<div class="review-card {card_class}">
                <div class="review-text">{str(row['text'])[:350]}{'...' if len(str(row['text']))>350 else ''}</div>
                <div class="review-meta">Tatsächlich: {actual} → Vorhersage: {predicted} (P(positiv) = {row.get('prob_positive',0.5):.0%})</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="warning-box"><p><strong>Warum irrt sich das Modell?</strong>
        <strong>Sarkasmus</strong> — „Oh great, another broken product" enthält <em>great</em>.
        <strong>Gemischte Meinungen</strong> — „Tolles Produkt, schrecklicher Versand".
        <strong>Kontext</strong> — „Not good" enthält <em>good</em>, aber die Verneinung wird ignoriert.</p></div>""", unsafe_allow_html=True)

    # Confidence bar chart
    examples = pd.concat([neg_ex.head(8).assign(sentiment="Negativ"), pos_ex.head(8).assign(sentiment="Positiv")]).reset_index(drop=True)
    examples["short"] = examples["text"].apply(lambda x: str(x)[:70]+"..." if len(str(x))>70 else str(x))

    fig = go.Figure(go.Bar(y=examples["short"], x=examples["prob_positive"], orientation="h",
                           marker_color=[RED if s=="Negativ" else GREEN for s in examples["sentiment"]]))
    fig.add_vline(x=0.5, line_dash="dash", line_color="white", line_width=1, annotation_text="0.5", annotation_font_color="white")
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="DM Sans"),
        title=dict(text="Modell-Konfidenz für einzelne Bewertungen", font=dict(size=16)),
        xaxis=dict(title="P(positiv)", gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(autorange="reversed", gridcolor="#334155"),
        height=600, margin=dict(l=350, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="story-divider">· · ·</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# KAPITEL 5: LIVE EXPLORER
# ═══════════════════════════════════════════════
st.markdown('<div class="chapter-header">Kapitel 5 — Live-Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="chapter-desc">Durchsuchen Sie die Vorhersagen selbst — filtern, entdecken, verstehen.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    filter_type = st.selectbox("Vorhersage:", ["Alle","Korrekt","Falsch"])
with col2:
    filter_label = st.selectbox("Sentiment:", ["Alle","Positiv","Negativ"])
with col3:
    min_conf = st.slider("Min. Konfidenz:", 0.5, 1.0, 0.5, 0.05) if has_probs else 0.5

df_show = df_preds.copy()
if filter_type=="Korrekt": df_show = df_show[df_show["label"]==df_show["prediction"]]
elif filter_type=="Falsch": df_show = df_show[df_show["label"]!=df_show["prediction"]]
if filter_label=="Positiv": df_show = df_show[df_show["label"]==1]
elif filter_label=="Negativ": df_show = df_show[df_show["label"]==0]
if has_probs and min_conf>0.5:
    df_show = df_show[((df_show["prob_positive"]>=min_conf)&(df_show["prediction"]==1.0))|
                       ((df_show["prob_positive"]<=(1-min_conf))&(df_show["prediction"]==0.0))]

st.markdown(f"**{len(df_show):,} Bewertungen** gefunden")

if len(df_show)>0:
    for _, row in df_show.sample(min(8,len(df_show))).iterrows():
        correct = row["label"]==row["prediction"]
        icon = "✅" if correct else "⚠️"
        actual = "Positiv" if row["label"]==1 else "Negativ"
        predicted = "Positiv" if row["prediction"]==1.0 else "Negativ"
        card_class = "positive" if row["label"]==1 else "negative"
        st.markdown(f"""<div class="review-card {card_class}">
            <div class="review-text">{icon} {str(row['text'])[:400]}{'...' if len(str(row['text']))>400 else ''}</div>
            <div class="review-meta">Tatsächlich: {actual} | Vorhersage: {predicted} | Konfidenz: {row.get('prob_positive',0.5):.0%}</div>
        </div>""", unsafe_allow_html=True)

    if st.button("🔄 Neue Stichprobe"): st.rerun()

st.markdown('<div class="story-divider">· · ·</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# FAZIT
# ═══════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding:2rem 1rem;">
    <div style="font-family:'DM Serif Display',serif; font-size:2rem; color:#F8FAFC; margin-bottom:0.5rem;">
        Kann eine Maschine Emotionen lesen?
    </div>
    <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#14B8A6; margin-bottom:1.5rem;">
        Ja — mit 84,7% Accuracy und AUC = 0,92
    </div>
    <div style="color:#94A3B8; font-size:0.9rem;">
        Vollständig mit Apache Spark und Python — von 3,6 Mio. Rohdaten zur Vorhersage<br>
        PySpark · TF-IDF · Logistic Regression · Streamlit<br><br>
        Thema 008 | Slot 1/2
    </div>
</div>
""", unsafe_allow_html=True)