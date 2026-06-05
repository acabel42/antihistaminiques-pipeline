import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
ENGINE = create_engine(os.getenv('DB_URL'))

st.set_page_config(page_title="Ruptures ANSM", page_icon="💊", layout="wide")
st.title("💊 Médicaments & Ruptures de stock ANSM")
st.markdown("Analyse des ruptures de stock par classe ATC — 2021 à 2024")

# ── Chargement ──────────────────────────────────────────────
@st.cache_data
def load_data():
    med  = pd.read_sql("SELECT * FROM medicaments", ENGINE)
    rup  = pd.read_sql("SELECT * FROM ruptures", ENGINE)
    try:
        gold = pd.read_csv('data/gold/gold_ml_advanced.csv')
    except:
        gold = pd.read_csv('data/gold/gold_ml.csv')
    return med, rup, gold

med, rup, gold = load_data()

# ── Filtres sidebar ─────────────────────────────────────────
st.sidebar.header("🔍 Filtres")

classe_select = st.sidebar.selectbox(
    "Classe ATC",
    ['R06 — Antihistaminiques', 'R03 — Antiasthmatiques', 'J01 — Antibiotiques'],
    index=0
)
code_atc = classe_select.split(' ')[0]

annees_dispo = sorted(rup['annee'].dropna().unique().astype(int))
annees_select = st.sidebar.multiselect(
    "Année(s)",
    options=annees_dispo,
    default=annees_dispo
)

# ── Application des filtres ─────────────────────────────────
anti_med = med[med['code_atc'].str.startswith(code_atc, na=False)]
anti_rup = rup[
    (rup['code_atc'].str.startswith(code_atc, na=False)) &
    (rup['annee'].isin(annees_select))
]

# ── KPIs ────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Médicaments", f"{len(anti_med)}")
col2.metric("Ruptures totales", f"{len(anti_rup)}")
col3.metric("Molécules touchées", f"{anti_rup['molecule'].nunique()}")
annee_pic = anti_rup.groupby('annee').size().idxmax() if len(anti_rup) > 0 else "N/A"
col4.metric("Pic de ruptures", str(annee_pic))
taux = gold['target_rupture'].mean() * 100
col5.metric("Taux de rupture R06", f"{taux:.1f}%")

st.markdown("---")

# ── Section 1 : Ruptures par année ──────────────────────────
st.subheader("📅 Évolution des ruptures par année")
rup_annee = anti_rup.groupby('annee').size().reset_index(name='nb_ruptures')
fig1 = px.bar(rup_annee, x='annee', y='nb_ruptures',
              labels={'nb_ruptures': 'Nombre de ruptures', 'annee': 'Année'},
              color='nb_ruptures', color_continuous_scale='Reds',
              title=f"Ruptures {code_atc} par année")
fig1.update_xaxes(tickmode='linear')
fig1.update_coloraxes(colorbar_title="Nombre de ruptures")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── Section 2 : Taux de rupture mensuel ─────────────────────
st.subheader("📊 Taux de rupture mensuel R06A")
gold_sorted = gold.sort_values('annee_mois_str')
fig_taux = px.bar(gold_sorted, x='annee_mois_str', y='target_rupture',
                  labels={
                      'annee_mois_str': 'Mois',
                      'target_rupture': 'Rupture (0=Non / 1=Oui)'
                  },
                  color='target_rupture',
                  color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
                  title="Mois avec rupture ou tension R06A (2021–2026)")
fig_taux.update_xaxes(tickangle=45)
st.plotly_chart(fig_taux, use_container_width=True)

st.markdown("---")

# ── Section 3 : Saisonnalité ────────────────────────────────
st.subheader("🗓️ Saisonnalité des ruptures")
col1, col2 = st.columns(2)

saison = anti_rup['saison_allergies'].value_counts().reset_index()
saison.columns = ['saison', 'count']
saison['saison'] = saison['saison'].map({1: 'Saison allergique', 0: 'Hors saison'})
fig2 = px.pie(saison, values='count', names='saison',
              title="Ruptures saison vs hors saison",
              color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
col1.plotly_chart(fig2, use_container_width=True)

rup_mois = anti_rup.groupby(['annee', 'mois']).size().reset_index(name='nb_ruptures')
fig3 = px.density_heatmap(rup_mois, x='mois', y='annee', z='nb_ruptures',
                           labels={
                               'nb_ruptures': 'Nombre de ruptures',
                               'mois': 'Mois',
                               'annee': 'Année'
                           },
                           color_continuous_scale='YlOrRd',
                           title="Heatmap ruptures par mois et année")
fig3.update_xaxes(tickvals=list(range(1,13)),
                  ticktext=['Jan','Fév','Mar','Avr','Mai','Jun',
                            'Jul','Aoû','Sep','Oct','Nov','Déc'])
fig3.update_coloraxes(colorbar_title="Nombre de ruptures")
col2.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Section 4 : Top molécules ───────────────────────────────
st.subheader("💊 Top molécules en rupture")
col1, col2 = st.columns(2)

top_mol = anti_rup['molecule'].value_counts().head(10).reset_index()
top_mol.columns = ['molecule', 'count']
fig4 = px.bar(top_mol, x='count', y='molecule', orientation='h',
              labels={'count': 'Nombre de ruptures', 'molecule': 'Molécule'},
              color='count', color_continuous_scale='Purples',
              title=f"Top 10 molécules {code_atc} en rupture")
fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
fig4.update_coloraxes(colorbar_title="Nombre de ruptures")
col1.plotly_chart(fig4, use_container_width=True)

causes = anti_rup['cause_categorie'].value_counts().reset_index()
causes.columns = ['cause', 'count']
fig5 = px.bar(causes, x='cause', y='count',
              labels={'count': 'Nombre de ruptures', 'cause': 'Cause'},
              color='count', color_continuous_scale='Oranges',
              title="Causes de rupture")
fig5.update_xaxes(tickangle=30)
fig5.update_coloraxes(colorbar_title="Nombre de ruptures")
col2.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ── Section 5 : Jointure médicaments × ruptures ─────────────
st.subheader("🔗 Médicaments les plus prescrits en rupture")
jointure = anti_med.merge(anti_rup, on='cis', how='inner')

if len(jointure) > 0:
    top_rup = jointure.groupby('molecule_x')['nb_patients_ville_x'].mean().sort_values(ascending=False).head(10).reset_index()
    top_rup.columns = ['molecule', 'nb_patients_moyen']
    fig6 = px.bar(top_rup, x='nb_patients_moyen', y='molecule', orientation='h',
                  labels={
                      'nb_patients_moyen': 'Nombre moyen de patients',
                      'molecule': 'Molécule'
                  },
                  color='nb_patients_moyen', color_continuous_scale='RdYlGn',
                  title=f"Molécules {code_atc} en rupture — nombre moyen de patients")
    fig6.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig6.update_coloraxes(colorbar_title="Nombre moyen de patients")
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # ── Scatter plot ─────────────────────────────────────────
    st.subheader("🎯 Molécules à double risque — Prescription vs Ruptures")

    scatter_data = jointure.groupby('molecule_x').agg(
        nb_ruptures=('cis', 'count'),
        nb_patients_moyen=('nb_patients_ville_x', 'mean')
    ).reset_index()
    scatter_data.columns = ['molecule', 'nb_ruptures', 'nb_patients_moyen']

    # Tronquer les noms longs
    scatter_data['molecule_court'] = scatter_data['molecule'].str[:20] + \
        scatter_data['molecule'].apply(lambda x: '...' if len(x) > 20 else '')

    fig_scatter = px.scatter(
        scatter_data,
        x='nb_patients_moyen',
        y='nb_ruptures',
        text='molecule_court',
        hover_name='molecule',
        size='nb_ruptures',
        color='nb_ruptures',
        color_continuous_scale='Reds',
        labels={
            'nb_patients_moyen': 'Nombre moyen de patients',
            'nb_ruptures': 'Nombre de ruptures',
        },
        title="Molécules à double risque — Volume de prescription vs Nombre de ruptures"
    )
    fig_scatter.update_traces(textposition='top center', textfont_size=10)
    fig_scatter.update_coloraxes(colorbar_title="Nb ruptures")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("💡 Les molécules en haut à droite sont les plus critiques — très prescrites ET souvent en rupture")

else:
    st.info("Pas de données disponibles pour cette classe ATC.")

st.divider()
st.caption("Projet Antihistaminiques — Jedha 2026 — LMN")