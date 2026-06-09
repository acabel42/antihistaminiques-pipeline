"""
pages/3_openmedic.py — Dashboard OpenMedic R06A
Consommation antihistaminiques par région, molécule, démographie
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path

st.set_page_config(page_title="OpenMedic — Consommation", page_icon="💊", layout="wide")
st.title("💊 Consommation Antihistaminiques")
st.caption("Données OpenMedic — Remboursements médicaments France 2021-2025")

# =====================
# CONSTANTES
# =====================
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
AGE_LABELS  = {0.0: "< 20 ans", 20.0: "20-59 ans", 60.0: "60 ans et +"}
SEXE_LABELS = {1: "Homme", 2: "Femme"}
MOIS_NOMS   = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
               7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}

# =====================
# CHARGEMENT DONNEES
# =====================
@st.cache_data
def load_openmedic():
    base = Path(__file__).resolve().parent.parent.parent.parent / "data" / "silver"
    df = pd.read_csv(base / "J0_silver_openmedic_2021_2025.csv", encoding="latin-1", low_memory=False)
    df["BOITES"]      = pd.to_numeric(df["BOITES"], errors="coerce")
    df["region"]      = df["BEN_REG"].map(REG_LABELS)
    df["tranche_age"] = df["age"].map(AGE_LABELS)
    df["genre"]       = df["sexe"].map(SEXE_LABELS)
    df["molecule"]    = df["L_ATC5"].str.strip().str.title()
    return df

@st.cache_data
def load_gold(code):
    path = f'data/gold/gold_ml_{code}.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.read_csv('data/gold/gold_ml.csv')

@st.cache_data
def load_medicaments():
    base = Path(__file__).resolve().parent.parent.parent.parent / "data" / "silver"
    df = pd.read_csv(base / "J0_silver_medicaments.csv", encoding="latin-1", low_memory=False)
    return df[df["est_antihistaminique"] == True]

df_om  = load_openmedic()
df_med = load_medicaments()

# =====================
# SIDEBAR
# =====================
st.sidebar.title("🔧 Filtres")

classe_select = st.sidebar.selectbox(
    "Classe ATC",
    ['R06A — Antihistaminiques', 'R03 — Antiasthmatiques', 'J01 — Antibiotiques'],
    index=0
)
code_atc = classe_select.split(' ')[0]
df_gold = load_gold(code_atc)

st.sidebar.divider()

annees       = sorted(df_om["annee"].dropna().unique().astype(int).tolist())
annees_dispo = ['Toutes'] + annees
annee_sel    = st.sidebar.selectbox("Année", annees_dispo, index=0)

mois_dispo = ['Tous'] + list(MOIS_NOMS.values())
mois_sel   = st.sidebar.selectbox("Mois", mois_dispo, index=0)

st.sidebar.divider()

exclure_corse = st.sidebar.checkbox("Exclure Corse", value=True,
    help="Anomalie : petite population + forte fréquentation touristique")

st.sidebar.divider()
st.sidebar.markdown(f"**Classe : {code_atc}**")
st.sidebar.markdown("**Source :** CNAM OpenMedic")

# =====================
# FILTRAGE
# =====================
df_f = df_om.copy()
df_f = df_f[df_f["ATC3"].str.startswith(code_atc, na=False)]
if annee_sel != 'Toutes':
    df_f = df_f[df_f["annee"] == int(annee_sel)]
if mois_sel != 'Tous':
    mois_num = {v: k for k, v in MOIS_NOMS.items()}[mois_sel]
    df_f = df_f[df_f["mois"] == mois_num]
df_f = df_f[df_f["region"].notna()]
if exclure_corse:
    df_f = df_f[df_f["region"] != "Corse"]

# =====================
# KPIs
# =====================
st.subheader(f"📊 Chiffres clés — {classe_select}")
col1, col2, col3 = st.columns(3)
col1.metric("Boîtes remboursées", f"{df_f['BOITES'].sum()/1e6:.1f}M")
col2.metric("Régions", df_f["region"].nunique())
col3.metric("Molécules", df_f["molecule"].nunique())

st.divider()

# =====================
# LIGNE 1 — Région + Evolution
# =====================
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("📍 Consommation par région (/1000 hab.)")
    if exclure_corse:
        st.caption("⚠️ Corse exclue — anomalie population touristique")
    st.write("")

    df_reg = df_f.groupby("region", as_index=False)["BOITES"].sum()
    df_pop = pd.DataFrame(list(POP_REGION.items()), columns=["region", "pop"])
    df_reg = df_reg.merge(df_pop, on="region", how="left")
    df_reg["norm_1000"] = (df_reg["BOITES"] / df_reg["pop"] * 1000).round(1)
    df_reg = df_reg.sort_values("norm_1000", ascending=True)

    fig1 = px.bar(df_reg, x="norm_1000", y="region", orientation="h",
                  labels={"norm_1000": "Boîtes / 1000 hab.", "region": ""},
                  color="norm_1000", color_continuous_scale="Blues")
    fig1.update_layout(height=420, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig1, use_container_width=True)

with col_g2:
    st.subheader(f"📈 Evolution de la consommation {code_atc}")
    st.caption("Boîtes prescrites et remboursées par an (millions).")
    st.write("")

    df_evol_y = df_om[df_om["ATC3"].str.startswith(code_atc, na=False)].groupby("annee", as_index=False)["BOITES"].sum()
    df_evol_y["BOITES_M"] = (df_evol_y["BOITES"] / 1e6).round(1)

    pct = ((df_evol_y["BOITES_M"].iloc[-1] - df_evol_y["BOITES_M"].iloc[0])
           / df_evol_y["BOITES_M"].iloc[0] * 100)

    fig2 = px.bar(df_evol_y, x="annee", y="BOITES_M",
                  labels={"BOITES_M": "Millions de boîtes", "annee": ""},
                  color_discrete_sequence=["#7bafd4"],
                  text="BOITES_M")
    fig2.update_traces(texttemplate="%{text:.1f}M", textposition="outside")
    fig2.add_annotation(
        x=0.98, y=0.05, xref="paper", yref="paper",
        text=f"+{pct:.0f}% sur 5 ans",
        showarrow=False,
        font=dict(color="#2c7bb6", size=13, family="Arial Black"),
        xanchor="right",
        bgcolor="white",
        bordercolor="#2c7bb6",
        borderwidth=1,
        borderpad=4
    )
    fig2.update_layout(
        height=420,
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis_title="Millions de boîtes",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =====================
# LIGNE 2 — Profil patients
# =====================
st.subheader("👥 Profil patients")
st.caption("Répartition démographique moyenne sur l'ensemble des spécialités antihistaminiques.")
st.write("")

col_dem1, col_dem2 = st.columns(2)

with col_dem1:
    age_means = df_med[["pct_age_0_19_ans", "pct_age_20_59_ans", "pct_age_60_ans_et_plus"]].mean()
    fig3 = px.pie(
        values=age_means.values,
        names=["0–19 ans", "20–59 ans", "60 ans +"],
        title="Répartition par âge",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

with col_dem2:
    sexe_means = df_med[["pct_sexe_female", "pct_sexe_male"]].mean()
    fig4 = px.pie(
        values=sexe_means.values,
        names=["Femmes", "Hommes"],
        title="Répartition par sexe",
        hole=0.4,
        color_discrete_map={"Femmes": "#E07B8A", "Hommes": "#7B9FE0"}
    )
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# =====================
# LIGNE 3 — Top molécules
# =====================
st.subheader(f"🔬 Top molécules — {code_atc}")
st.write("")
n = st.slider("Nombre de molécules", 5, 15, 10)
df_mol = df_f.groupby("molecule", as_index=False)["BOITES"].sum()
df_mol = df_mol.nlargest(n, "BOITES").sort_values("BOITES", ascending=True)

fig5 = px.bar(df_mol, x="BOITES", y="molecule", orientation="h",
              labels={"BOITES": "Boîtes remboursées", "molecule": ""},
              color="BOITES", color_continuous_scale="Reds")
fig5.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig5, use_container_width=True)

st.divider()

# =====================
# LIGNE 4 — Heatmap région x molécule
# =====================
st.subheader("🗺️ Heatmap Région × Molécule")
st.caption("Volume de boîtes par région × molécule — en millions de boîtes")
st.write("")

n_hm     = st.slider("Top N molécules (heatmap)", 5, 10, 8, key="hm")
top_mols = df_f.groupby("molecule")["BOITES"].sum().nlargest(n_hm).index.tolist()
df_hm    = df_f[df_f["molecule"].isin(top_mols)]
pivot    = (df_hm.groupby(["region", "molecule"])["BOITES"]
            .sum().unstack(fill_value=0) / 1e6).round(1)

fig6 = px.imshow(
    pivot,
    labels=dict(x="Molécule", y="Région", color="Millions de boîtes"),
    aspect="auto",
    color_continuous_scale="YlOrRd",
    text_auto=".1f",
)
fig6.update_layout(
    height=450,
    xaxis=dict(tickangle=-30),
    coloraxis_colorbar=dict(title="Millions de boîtes"),
)
st.plotly_chart(fig6, use_container_width=True)

st.divider()

# =====================
# LIGNE 5 — Top 10 laboratoires
# =====================
st.subheader("🏭 Top 10 laboratoires antihistaminiques")
st.caption("Nombre de spécialités R06A par laboratoire pharmaceutique.")
st.write("")

top_labo = df_med["laboratoire"].value_counts().head(10).reset_index()
top_labo.columns = ["laboratoire", "count"]
top_labo["laboratoire"] = top_labo["laboratoire"].str.title()

fig7 = px.bar(top_labo, x="count", y="laboratoire", orientation="h",
              color="count", color_continuous_scale="Blues",
              labels={"count": "Nb spécialités", "laboratoire": ""})
fig7.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                   yaxis=dict(categoryorder="total ascending"))
st.plotly_chart(fig7, use_container_width=True)

st.divider()

# =====================
# LIGNE 6 — Evolution top 5 molécules
# =====================
st.subheader(f"📊 Evolution des top 5 molécules {code_atc} (2021-2025)")
st.caption("Comment les molécules dominantes évoluent d'année en année.")
st.write("")

top5_mols = df_f.groupby("molecule")["BOITES"].sum().nlargest(5).index.tolist()
df_top5 = (df_f[df_f["molecule"].isin(top5_mols)]
           .groupby(["annee", "molecule"], as_index=False)["BOITES"].sum())
df_top5["BOITES_M"] = (df_top5["BOITES"] / 1e6).round(2)

fig8 = px.line(df_top5, x="annee", y="BOITES_M", color="molecule",
               markers=True,
               labels={"BOITES_M": "Millions de boîtes", "annee": "Année", "molecule": "Molécule"})
fig8.update_layout(
    height=380,
    xaxis=dict(tickmode="linear", dtick=1),
    legend=dict(x=0, y=1.15, orientation="h"),
)
st.plotly_chart(fig8, use_container_width=True)

st.divider()

# =====================
# LIGNE 7 — Coût remboursement par molécule
# =====================
st.subheader("💶 Coût de remboursement médian par boîte")
st.caption("Molécules les plus chères à rembourser pour l'Assurance Maladie (€/boîte).")
st.write("")

df_om_cout = df_f.copy()
df_om_cout["rem_par_boite"] = df_om_cout["REM_clean"] / df_om_cout["BOITES"].replace(0, float("nan"))
ratio_mol = (df_om_cout.groupby("molecule")["rem_par_boite"]
             .median().sort_values(ascending=True).reset_index())
ratio_mol.columns = ["molecule", "rem_median"]

fig9 = px.bar(ratio_mol, x="rem_median", y="molecule", orientation="h",
              color="rem_median", color_continuous_scale="Blues",
              labels={"rem_median": "€ / boîte (médiane)", "molecule": ""},
              text=ratio_mol["rem_median"].apply(lambda x: f"{x:.2f}€"))
fig9.update_traces(textposition="outside")
fig9.update_layout(height=420, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig9, use_container_width=True)

st.divider()

# =====================
# LIGNE 8 — Concentration marché (Lorenz)
# =====================
st.subheader("📉 Concentration du marché")
st.caption("Courbe de Lorenz — quelques molécules représentent la majorité des volumes.")
st.write("")

volumes = df_f.groupby("molecule")["BOITES"].sum().sort_values().values
if len(volumes) > 1:
    cumsum = np.cumsum(volumes)
    cumsum_pct = cumsum / cumsum[-1] * 100
    n = len(cumsum_pct)
    x_pct = np.arange(1, n + 1) / n * 100
    gini = 1 - 2 * float(np.trapezoid(cumsum_pct / 100, x_pct / 100))

    col_lorenz, col_parts = st.columns(2)

    with col_lorenz:
        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(
            x=x_pct, y=cumsum_pct,
            name="Lorenz",
            line=dict(color="#e07b54", width=2.5),
            fill="tonexty", fillcolor="rgba(224,123,84,0.15)"
        ))
        fig10.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            name="Égalité parfaite",
            line=dict(color="black", dash="dash", width=1),
        ))
        fig10.add_hline(y=80, line_dash="dot", line_color="#7bafd4", opacity=0.7)
        fig10.update_layout(
            height=380,
            title=f"Courbe de Lorenz — Gini = {gini:.2f}",
            xaxis_title="% des molécules",
            yaxis_title="% cumulé du volume",
            legend=dict(x=0, y=1.1, orientation="h"),
        )
        st.plotly_chart(fig10, use_container_width=True)

    with col_parts:
        df_mol_parts = df_f.groupby("molecule", as_index=False)["BOITES"].sum()
        df_mol_parts = df_mol_parts.sort_values("BOITES", ascending=False)
        top6 = df_mol_parts.head(6).copy()
        autres = pd.DataFrame({"molecule": ["Autres"], "BOITES": [df_mol_parts.iloc[6:]["BOITES"].sum()]})
        df_pie = pd.concat([top6, autres])
        df_pie["BOITES_M"] = (df_pie["BOITES"] / 1e6).round(1)

        fig11 = px.pie(df_pie, values="BOITES_M", names="molecule",
                       title="Parts de marché — Top 6 + Autres",
                       hole=0.4,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        fig11.update_layout(height=380, showlegend=True,
                            legend=dict(x=1, y=0.5))
        st.plotly_chart(fig11, use_container_width=True)

    idx_80 = int(np.searchsorted(cumsum_pct, 80))
    n_mol_80 = n - idx_80
    st.info(f"💡 **Indice de Gini : {gini:.2f}** — Marché très concentré. "
            f"{n_mol_80} molécule{'s' if n_mol_80 > 1 else ''} sur {n} "
            f"représente{'nt' if n_mol_80 > 1 else ''} 80% des volumes remboursés. "
            f"Une rupture sur cette molécule dominante aurait un impact massif sur l'approvisionnement.")

st.divider()
st.caption(" Projet Antihistaminiques — Jedha 2026 — LMN ")