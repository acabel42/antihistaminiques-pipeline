import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
ENGINE = create_engine(os.getenv('DB_URL'))

st.set_page_config(page_title="Ruptures ANSM", page_icon="💊", layout="wide")
st.title("💊 Médicaments & Ruptures de stock ANSM")
st.markdown("Analyse des antihistaminiques R06A — 2021 à 2024")

# ── Chargement ──────────────────────────────────────────────
@st.cache_data
def load_data():
    med = pd.read_sql("SELECT * FROM medicaments", ENGINE)
    rup = pd.read_sql("SELECT * FROM ruptures", ENGINE)
    return med, rup

med, rup = load_data()
anti_med = med[med['est_antihistaminique'] == True]
anti_rup = rup[rup['est_antihistaminique'] == True]

# ── KPIs ────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Antihistaminiques", f"{len(anti_med)}")
col2.metric("Ruptures totales", f"{len(anti_rup)}")
col3.metric("Molécules touchées", f"{anti_rup['molecule'].nunique()}")
col4.metric("Pic de ruptures", "2022")

st.markdown("---")

# ── Section 1 : Laboratoires ────────────────────────────────
st.subheader("🏭 Top 10 laboratoires")
top_labo = anti_med['laboratoire'].value_counts().head(10).reset_index()
top_labo.columns = ['laboratoire', 'count']
fig1 = px.bar(top_labo, x='count', y='laboratoire', orientation='h',
              color='count', color_continuous_scale='Blues',
              title="Top 10 laboratoires — antihistaminiques")
fig1.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── Section 2 : Profil patients ─────────────────────────────
st.subheader("👥 Profil démographique des patients")
col1, col2 = st.columns(2)

age_means = anti_med[['pct_age_0_19_ans', 'pct_age_20_59_ans', 'pct_age_60_ans_et_plus']].mean()
fig2 = px.pie(
    values=age_means.values,
    names=['0–19 ans', '20–59 ans', '60 ans +'],
    title="Répartition par âge",
    color_discrete_sequence=px.colors.qualitative.Set2
)
col1.plotly_chart(fig2, use_container_width=True)

sexe_means = anti_med[['pct_sexe_female', 'pct_sexe_male']].mean()
fig3 = px.pie(
    values=sexe_means.values,
    names=['Femmes', 'Hommes'],
    title="Répartition par sexe",
    color_discrete_sequence=['#E07B8A', '#7B9FE0']
)
col2.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Section 3 : Ruptures par année ──────────────────────────
st.subheader("📅 Évolution des ruptures par année")
rup_annee = anti_rup.groupby('annee').size().reset_index(name='nb_ruptures')
fig4 = px.bar(rup_annee, x='annee', y='nb_ruptures',
              color='nb_ruptures', color_continuous_scale='Reds',
              title="Ruptures antihistaminiques par année")
fig4.update_xaxes(tickmode='linear')
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Section 4 : Saisonnalité ────────────────────────────────
st.subheader("🗓️ Saisonnalité des ruptures")
col1, col2 = st.columns(2)

saison = anti_rup['saison_allergies'].value_counts().reset_index()
saison.columns = ['saison', 'count']
saison['saison'] = saison['saison'].map({1: 'Saison allergique', 0: 'Hors saison'})
fig5 = px.pie(saison, values='count', names='saison',
              title="Ruptures saison vs hors saison",
              color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
col1.plotly_chart(fig5, use_container_width=True)

rup_mois = anti_rup.groupby(['annee', 'mois']).size().reset_index(name='nb_ruptures')
fig6 = px.density_heatmap(rup_mois, x='mois', y='annee', z='nb_ruptures',
                           color_continuous_scale='YlOrRd',
                           title="Heatmap ruptures par mois et année")
fig6.update_xaxes(tickvals=list(range(1,13)),
                  ticktext=['Jan','Fév','Mar','Avr','Mai','Jun',
                            'Jul','Aoû','Sep','Oct','Nov','Déc'])
col2.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ── Section 5 : Top molécules ───────────────────────────────
st.subheader("💊 Top molécules en rupture")
col1, col2 = st.columns(2)

top_mol = anti_rup['molecule'].value_counts().head(10).reset_index()
top_mol.columns = ['molecule', 'count']
fig7 = px.bar(top_mol, x='count', y='molecule', orientation='h',
              color='count', color_continuous_scale='Purples',
              title="Top 10 molécules en rupture")
fig7.update_layout(yaxis={'categoryorder': 'total ascending'})
col1.plotly_chart(fig7, use_container_width=True)

causes = anti_rup['cause_categorie'].value_counts().reset_index()
causes.columns = ['cause', 'count']
fig8 = px.bar(causes, x='cause', y='count',
              color='count', color_continuous_scale='Oranges',
              title="Causes de rupture")
fig8.update_xaxes(tickangle=30)
col2.plotly_chart(fig8, use_container_width=True)

st.markdown("---")

# ── Section 6 : Jointure médicaments × ruptures ─────────────
st.subheader("🔗 Médicaments les plus prescrits en rupture")
jointure = anti_med.merge(anti_rup, on='cis', how='inner')
top_rup = jointure.groupby('molecule_x')['nb_patients_ville_x'].mean().sort_values(ascending=False).head(10).reset_index()
top_rup.columns = ['molecule', 'nb_patients_moyen']
fig9 = px.bar(top_rup, x='nb_patients_moyen', y='molecule', orientation='h',
              color='nb_patients_moyen', color_continuous_scale='RdYlGn',
              title="Molécules en rupture — nombre moyen de patients")
fig9.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig9, use_container_width=True)