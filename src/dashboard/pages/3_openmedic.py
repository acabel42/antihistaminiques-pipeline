"""
pages/3_openmedic.py — Dashboard OpenMedic R06A
Consommation antihistaminiques par région, molécule, démographie
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="OpenMedic R06A", page_icon="💊", layout="wide")
st.title("💊 Consommation Antihistaminiques R06A")
st.write("Analyse des remboursements par région, molécule et démographie — France métropolitaine")

# Codes régions INSEE
REG_LABELS = {
    11: "Île-de-France", 24: "Centre-Val de Loire", 27: "Bourgogne-Franche-Comté",
    28: "Normandie", 32: "Hauts-de-France", 44: "Grand Est", 52: "Pays de la Loire",
    53: "Bretagne", 75: "Nouvelle-Aquitaine", 76: "Occitanie",
    84: "Auvergne-Rhône-Alpes", 93: "PACA", 94: "Corse"
}

POP_REGION = {
    "Île-de-France": 12271794, "Centre-Val de Loire": 2572853,
    "Bourgogne-Franche-Comté": 2811423, "Normandie": 3310317,
    "Hauts-de-France": 6001897, "Grand Est": 5581676,
    "Pays de la Loire": 3877171, "Bretagne": 3398083,
    "Nouvelle-Aquitaine": 6109551, "Occitanie": 6078468,
    "Auvergne-Rhône-Alpes": 8087100, "PACA": 5115265, "Corse": 344679
}

AGE_LABELS = {0.0: "< 20 ans", 20.0: "20-59 ans", 60.0: "60 ans et +"}
SEXE_LABELS = {1: "Homme", 2: "Femme"}

# Chargement
@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent.parent.parent.parent / "data" / "silver"
    df = pd.read_csv(base / "J0_silver_openmedic_2021_2025.csv", encoding="latin-1", low_memory=False)
    df["BOITES"] = pd.to_numeric(df["BOITES"], errors="coerce")
    df["region"] = df["BEN_REG"].map(REG_LABELS)
    df["tranche_age"] = df["age"].map(AGE_LABELS)
    df["genre"] = df["sexe"].map(SEXE_LABELS)
    df["molecule"] = df["L_ATC5"].str.strip().str.title()
    return df

df = load_data()

# Filtres sidebar
st.sidebar.header("Filtres")
annees = sorted(df["annee"].dropna().unique().astype(int).tolist())
annee_sel = st.sidebar.multiselect("Année", annees, default=annees)
exclure_corse = st.sidebar.checkbox("Exclure Corse", value=True)

# Filtrage
df_f = df[df["annee"].isin(annee_sel)]
df_f = df_f[df_f["region"].notna()]
if exclure_corse:
    df_f = df_f[df_f["region"] != "Corse"]

# KPIs
st.subheader("Chiffres clés")
col1, col2, col3 = st.columns(3)
col1.metric("Boîtes remboursées", f"{df_f['BOITES'].sum()/1e6:.1f}M")
col2.metric("Régions", df_f["region"].nunique())
col3.metric("Molécules R06A", df_f["molecule"].nunique())

st.markdown("---")

# Graphique 1 — Consommation par région normalisée
st.subheader("Consommation par région (normalisée /1000 hab.)")
if exclure_corse:
    st.caption("⚠️ Corse exclue — anomalie population touristique")

df_reg = df_f.groupby("region", as_index=False)["BOITES"].sum()
df_pop = pd.DataFrame(list(POP_REGION.items()), columns=["region", "pop"])
df_reg = df_reg.merge(df_pop, on="region", how="left")
df_reg["norm_1000"] = (df_reg["BOITES"] / df_reg["pop"] * 1000).round(1)
df_reg = df_reg.sort_values("norm_1000", ascending=True)

fig1 = px.bar(df_reg, x="norm_1000", y="region", orientation="h",
              labels={"norm_1000": "Boîtes / 1000 hab.", "region": ""},
              color="norm_1000", color_continuous_scale="Blues")
fig1.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# Graphique 2 — Top molécules
st.subheader("Top molécules R06A")
n = st.slider("Nombre de molécules", 5, 15, 10)
df_mol = df_f.groupby("molecule", as_index=False)["BOITES"].sum()
df_mol = df_mol.nlargest(n, "BOITES").sort_values("BOITES", ascending=True)

fig2 = px.bar(df_mol, x="BOITES", y="molecule", orientation="h",
              labels={"BOITES": "Boîtes remboursées", "molecule": ""},
              color="BOITES", color_continuous_scale="Reds")
fig2.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Graphique 3 — Démographie
st.subheader("Profil démographique")
col_age, col_sexe = st.columns(2)

with col_age:
    df_age = df_f[df_f["tranche_age"].notna()].groupby("tranche_age", as_index=False)["BOITES"].sum()
    fig3 = px.bar(df_age, x="tranche_age", y="BOITES",
                  labels={"BOITES": "Boîtes", "tranche_age": ""},
                  title="Par tranche d'âge", color_discrete_sequence=["#1f77b4"])
    st.plotly_chart(fig3, use_container_width=True)

with col_sexe:
    df_sexe = df_f[df_f["genre"].notna()].groupby("genre", as_index=False)["BOITES"].sum()
    fig4 = px.pie(df_sexe, values="BOITES", names="genre",
                  title="Par sexe", hole=0.4,
                  color_discrete_map={"Homme": "#1f77b4", "Femme": "#d62728"})
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Graphique 4 — Evolution temporelle
st.subheader("Evolution annuelle")
df_evol = df[df["region"].notna()]
if exclure_corse:
    df_evol = df_evol[df_evol["region"] != "Corse"]
df_evol_y = df_evol.groupby("annee", as_index=False)["BOITES"].sum()
fig5 = px.line(df_evol_y, x="annee", y="BOITES",
               labels={"BOITES": "Boîtes remboursées", "annee": "Année"},
               markers=True)
fig5.update_layout(height=280)
st.plotly_chart(fig5, use_container_width=True)

st.caption("Source : CNAM OpenMedic 2021-2025 — Classe ATC R06A")