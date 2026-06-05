import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

st.set_page_config(
    page_title="Antihistaminiques — Dashboard",
    page_icon="💊",
    layout="wide"
)

# Connexion DB
DB_URL = os.getenv('DB_URL', 'postgresql://pipeline:pipeline2026@localhost:5432/antihistaminiques')
engine = create_engine(DB_URL)

# Titre
st.title("💊 Surveillance Ruptures Antihistaminiques")
st.markdown("**Projet Jedha 2026** — Pipeline Data Engineering + ML")
st.divider()

# Filtre classe ATC
classe = st.selectbox(
    "Classe ATC",
    ['R06 — Antihistaminiques', 'R03 — Antiasthmatiques', 'J01 — Antibiotiques'],
    index=0
)
code_atc = classe.split(' ')[0]

# Chargement données
@st.cache_data
def load_gold(code):
    path = f'data/gold/gold_ml_{code}.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.read_csv('data/gold/gold_ml.csv')

df = load_gold(code_atc)

# KPIs
st.subheader(f"KPIs — {classe}")
col1, col2, col3, col4 = st.columns(4)

with col1:
    taux = df['target_rupture'].mean() * 100
    st.metric("Taux de rupture", f"{taux:.1f}%")

with col2:
    mois_risque = df.groupby('mois')['target_rupture'].mean().idxmax()
    mois_noms = {1:'Jan',2:'Fev',3:'Mar',4:'Avr',5:'Mai',6:'Jun',
                 7:'Jul',8:'Aou',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    st.metric("Mois le plus à risque", mois_noms[mois_risque])

with col3:
    st.metric("Mois en tension", f"{df['target_rupture'].sum()} / {len(df)}")

with col4:
    st.metric("ROC-AUC Modèle", "0.771")

st.divider()

# Graphique évolution
st.subheader("Evolution des ruptures et du pollen")
fig = px.bar(
    df, x='annee_mois_str', y='nb_ruptures',
    color='target_rupture',
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
    title=f'Ruptures {code_atc} par mois 2021-2026',
    labels={'annee_mois_str': 'Mois', 'nb_ruptures': 'Nb ruptures', 'target_rupture': 'Rupture'}
)
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.caption("Projet Antihistaminiques — Jedha 2026 — Collègues 1, 2, 3")
