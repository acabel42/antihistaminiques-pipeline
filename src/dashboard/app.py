import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

st.set_page_config(
    page_title="Antihistaminiques — Dashboard",
    page_icon="💊",
    layout="wide"
)

DB_URL = os.getenv('DB_URL', 'postgresql://pipeline:pipeline2026@localhost:5432/antihistaminiques')
engine = create_engine(DB_URL)

# =====================
# HEADER
# =====================
st.title("💊 Surveillance Ruptures Antihistaminiques")
st.markdown("**Projet Jedha 2026** — Pipeline Data Engineering + ML — Prédire les ruptures de stock à partir du pollen et de la météo")
st.divider()

# =====================
# FILTRE ATC
# =====================
classe = st.selectbox(
    "Classe ATC",
    ['R06 — Antihistaminiques', 'R03 — Antiasthmatiques', 'J01 — Antibiotiques'],
    index=0
)
code_atc = classe.split(' ')[0]

@st.cache_data
def load_gold(code):
    path = f'data/gold/gold_ml_{code}.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.read_csv('data/gold/gold_ml.csv')

@st.cache_data
def load_pred(code):
    path = f'data/gold/gold_predictions_{code}.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.read_csv('data/gold/gold_predictions.csv')

df = load_gold(code_atc)
pred = load_pred(code_atc)

mois_noms = {1:'Jan',2:'Fev',3:'Mar',4:'Avr',5:'Mai',6:'Jun',
             7:'Jul',8:'Aou',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

st.divider()

# =====================
# SECTION 1 — KPIs DONNEES REELLES
# =====================
st.subheader(f"📊 KPIs Réels — {classe}")
st.write("")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    taux = df['target_rupture'].mean() * 100
    st.metric("Taux de rupture", f"{taux:.1f}%",
              help="% de mois avec au moins une rupture ou tension")
with col2:
    mois_risque = df.groupby('mois')['target_rupture'].mean().idxmax()
    st.metric("Mois le plus à risque", mois_noms[mois_risque],
              help="Mois avec le plus fort taux de rupture historique")
with col3:
    st.metric("Mois en tension", f"{int(df['target_rupture'].sum())} / {len(df)}",
              help="Nombre de mois avec rupture ou tension sur 60 mois")
with col4:
    corr = df['ambroisie_moy'].corr(df['target_rupture']) if 'ambroisie_moy' in df.columns else 0
    st.metric("Signal ambroisie", f"r={corr:.3f}",
              help="Corrélation ambroisie / ruptures — signal principal")
with col5:
    annee_max = df.groupby('annee')['nb_ruptures'].sum().idxmax()
    st.metric("Année la plus touchée", str(annee_max),
              help="Année avec le plus grand nombre de ruptures")

st.divider()

# =====================
# SECTION 2 — KPIs ML
# =====================
st.subheader("🤖 KPIs Modèle ML")
st.write("")

col6, col7, col8, col9 = st.columns(4)
with col6:
    st.metric("ROC-AUC RF", "0.771",
              help="Random Forest Classifier — ROC-AUC sur jeu de test")
with col7:
    st.metric("Baseline LR", "0.457",
              help="Régression Logistique — justifie le choix RF")
with col8:
    nb_risque = int((pred['pred_rupture']==1).sum()) if pred is not None else 0
    st.metric("Mois à risque prédit", f"{nb_risque} / {len(pred)}",
              help="Nombre de mois où le modèle prédit une rupture")
with col9:
    proba_max = pred['proba_rupture'].max() * 100 if pred is not None else 0
    mois_max = pred.loc[pred['proba_rupture'].idxmax(), 'annee_mois_str'] if pred is not None else ''
    st.metric("Proba max rupture", f"{proba_max:.1f}%",
              help=f"Mois le plus à risque : {mois_max}")

st.divider()

# =====================
# SECTION 3 — GRAPHIQUE RUPTURES
# =====================
st.subheader(f"📈 Evolution des ruptures — {classe}")
st.write("")

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig = px.bar(
        df, x='annee_mois_str', y='nb_ruptures',
        color='target_rupture',
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        labels={'annee_mois_str': 'Mois', 'nb_ruptures': 'Nb ruptures', 'target_rupture': 'Rupture'},
        title=f'Ruptures {code_atc} par mois 2021-2026'
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=380, legend=dict(orientation='h', x=0.5, y=1.1, xanchor='center'))
    st.plotly_chart(fig, use_container_width=True)

with col_g2:
    if pred is not None:
        pred['proba_pct'] = (pred['proba_rupture'] * 100).round(1)
        pred['risque'] = pred['pred_rupture'].map({0: '✅ Pas de risque', 1: '🚨 Rupture prédite'})
        fig2 = px.bar(
            pred, x='annee_mois_str', y='proba_pct',
            color='risque',
            color_discrete_map={'✅ Pas de risque': '#2ecc71', '🚨 Rupture prédite': '#e74c3c'},
            labels={'annee_mois_str': 'Mois', 'proba_pct': 'Probabilité (%)', 'risque': 'Prédiction'},
            title=f'Probabilité de rupture prédite — {code_atc}'
        )
        fig2.add_hline(y=50, line_dash='dash', line_color='black', annotation_text='Seuil 50%')
        fig2.update_xaxes(tickangle=45)
        fig2.update_layout(height=380, legend=dict(orientation='h', x=0.5, y=1.1, xanchor='center'))
        st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Projet Antihistaminiques — Jedha 2026 — LMN")