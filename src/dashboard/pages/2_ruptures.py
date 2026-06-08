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
st.markdown("Analyse des ruptures de stock par classe ATC — 2021 à 2026")

# ── Chargement ──────────────────────────────────────────────
@st.cache_data
def load_data():
    med = pd.read_sql("SELECT * FROM medicaments", ENGINE)
    rup = pd.read_sql("SELECT * FROM ruptures", ENGINE)

    # Ruptures 2025-2026
    try:
        rup_2026_raw = pd.read_csv('data/silver/J0_silver_ruptures_ansm_2026.csv')
        rup_2026 = rup_2026_raw[rup_2026_raw['annee_debut'] >= 2025].copy()
        rup_2026['annee']            = rup_2026['annee_debut']
        rup_2026['mois']             = rup_2026['mois_debut']
        rup_2026['molecule']         = rup_2026['dci_norm']
        rup_2026['code_atc']         = rup_2026['ATC4']
        rup_2026['saison_allergies'] = rup_2026['mois_debut'].isin([3, 4, 5, 6]).astype(int)
        cause_map = {
            'Rupture de stock':            'Rupture',
            "Tension d'approvisionnement": 'Tension approvisionnement',
            'Remise à disposition':        'Autre',
            'Arrêt de commercialisation':  'Arrêt commercialisation'
        }
        rup_2026['cause_categorie'] = rup_2026['statut'].map(cause_map).fillna('Autre')
        rup_2026['cis'] = None
        cols = ['annee', 'mois', 'molecule', 'code_atc',
                'saison_allergies', 'cause_categorie', 'cis']
        rup_all = pd.concat([rup[cols], rup_2026[cols]], ignore_index=True)
    except Exception:
        rup_all = rup.copy()

    # Gold par classe ATC
    gold_files = {
        'R06': 'data/gold/gold_ml_R06.csv',
        'R03': 'data/gold/gold_ml_R03.csv',
        'J01': 'data/gold/gold_ml_J01.csv',
    }
    gold_dict = {}
    for atc, path in gold_files.items():
        try:
            gold_dict[atc] = pd.read_csv(path)
        except Exception:
            gold_dict[atc] = None

    # Fallback gold global
    try:
        gold_global = pd.read_csv('data/gold/gold_ml_advanced.csv')
    except Exception:
        gold_global = pd.read_csv('data/gold/gold_ml.csv')

    return med, rup, rup_all, gold_dict, gold_global


med, rup, rup_all, gold_dict, gold_global = load_data()

# ── Filtres sidebar ─────────────────────────────────────────
st.sidebar.header("🔍 Filtres")

classe_select = st.sidebar.selectbox(
    "Classe ATC",
    ['R06 — Antihistaminiques', 'R03 — Antiasthmatiques', 'J01 — Antibiotiques'],
    index=0
)
code_atc = classe_select.split(' ')[0]

annees_dispo = sorted(rup_all['annee'].dropna().unique().astype(int))
annees_select = st.sidebar.multiselect(
    "Année(s)",
    options=annees_dispo,
    default=annees_dispo
)

# ── Application des filtres ─────────────────────────────────
anti_med = med[med['code_atc'].str.startswith(code_atc, na=False)]

anti_rup = rup_all[
    (rup_all['code_atc'].str.startswith(code_atc, na=False)) &
    (rup_all['annee'].isin(annees_select))
]

# Jointure sur 2021-2024 uniquement (cis requis)
anti_rup_join = rup[
    (rup['code_atc'].str.startswith(code_atc, na=False)) &
    (rup['annee'].isin(annees_select))
]

# Gold pour la classe sélectionnée
gold = gold_dict.get(code_atc) if gold_dict.get(code_atc) is not None else gold_global

# ── KPIs ────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Médicaments", f"{len(anti_med)}")
col2.metric("Ruptures totales", f"{len(anti_rup)}")
col3.metric("Molécules touchées", f"{anti_rup['molecule'].nunique()}")
annee_pic = anti_rup.groupby('annee').size().idxmax() if len(anti_rup) > 0 else "N/A"
col4.metric("Pic de ruptures", str(annee_pic))
taux = gold['target_rupture'].mean() * 100 if gold is not None else 0
col5.metric(f"Taux de rupture {code_atc}", f"{taux:.1f}%")

st.markdown("---")

# ── Section 1 : Ruptures par année ──────────────────────────
st.subheader("📅 Évolution des ruptures par année (2021–2026)")

rup_annee = anti_rup.groupby('annee').size().reset_index(name='nb_ruptures')
fig1 = px.bar(
    rup_annee, x='annee', y='nb_ruptures',
    labels={'nb_ruptures': 'Nombre de ruptures', 'annee': 'Année'},
    color='nb_ruptures', color_continuous_scale='Reds',
    title=f"Ruptures {code_atc} par année (2021–2026)"
)
fig1.update_xaxes(tickmode='linear')
fig1.update_coloraxes(colorbar_title="Nombre de ruptures")

if len(rup_annee) > 0 and rup_annee['annee'].max() == 2026:
    val_2026 = rup_annee[rup_annee['annee'] == 2026]['nb_ruptures'].values[0]
    fig1.add_annotation(
        x=2026, y=val_2026,
        text="* jan–juin 2026", showarrow=False,
        yshift=12, font=dict(size=11, color='gray')
    )

st.plotly_chart(fig1, width='stretch')

st.markdown("---")

# ── Section 2 : Taux de rupture mensuel ─────────────────────
st.subheader(f"📊 Taux de rupture mensuel {code_atc}")

if gold is not None and 'annee_mois_str' in gold.columns:
    gold_sorted = gold.sort_values('annee_mois_str')
    fig_taux = px.bar(
        gold_sorted, x='annee_mois_str', y='target_rupture',
        labels={
            'annee_mois_str': 'Mois',
            'target_rupture': 'Rupture (0=Non / 1=Oui)'
        },
        color='target_rupture',
        color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
        title=f"Mois avec rupture ou tension {code_atc} (2021–2026)"
    )
    fig_taux.update_xaxes(tickangle=45)
    st.plotly_chart(fig_taux, width='stretch')
else:
    st.info(f"Données gold non disponibles pour {code_atc}.")

st.markdown("---")

# ── Section 3 : Saisonnalité ────────────────────────────────
st.subheader("🗓️ Saisonnalité des ruptures")
col1, col2 = st.columns(2)

saison = anti_rup['saison_allergies'].value_counts().reset_index()
saison.columns = ['saison', 'count']
saison['saison'] = saison['saison'].map({1: 'Saison allergique', 0: 'Hors saison'})
fig2 = px.pie(
    saison, values='count', names='saison',
    title="Ruptures saison vs hors saison (2021–2026)",
    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
)
col1.plotly_chart(fig2, width='stretch')

rup_mois = anti_rup.groupby(['annee', 'mois']).size().reset_index(name='nb_ruptures')

# Construire un pivot complet avec toutes les années et tous les mois
mois_labels = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
annees_dispo_heatmap = sorted(rup_mois['annee'].unique())
pivot = rup_mois.pivot(index='annee', columns='mois', values='nb_ruptures')
pivot = pivot.reindex(index=annees_dispo_heatmap, columns=range(1, 13)).fillna(0)
pivot.columns = mois_labels
pivot.index = [str(a) for a in pivot.index]

fig3 = px.imshow(
    pivot,
    color_continuous_scale='YlOrRd',
    labels={'color': 'Nb ruptures', 'x': 'Mois', 'y': 'Année'},
    title="Heatmap ruptures par mois et année (2021–2026)",
    text_auto=True,
    aspect='auto'
)
fig3.update_coloraxes(colorbar_title="Nombre de ruptures")
col2.plotly_chart(fig3, width='stretch')

st.markdown("---")

# ── Section 4 : Top molécules ───────────────────────────────
st.subheader("💊 Top molécules en rupture")
col1, col2 = st.columns(2)

top_mol = anti_rup['molecule'].value_counts().head(10).reset_index()
top_mol.columns = ['molecule', 'count']
fig4 = px.bar(
    top_mol, x='count', y='molecule', orientation='h',
    labels={'count': 'Nombre de ruptures', 'molecule': 'Molécule'},
    color='count', color_continuous_scale='Purples',
    title=f"Top 10 molécules {code_atc} en rupture (2021–2026)"
)
fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
fig4.update_coloraxes(colorbar_title="Nombre de ruptures")
col1.plotly_chart(fig4, width='stretch')

causes = anti_rup['cause_categorie'].value_counts().reset_index()
causes.columns = ['cause', 'count']
fig5 = px.bar(
    causes, x='cause', y='count',
    labels={'count': 'Nombre de ruptures', 'cause': 'Cause'},
    color='count', color_continuous_scale='Oranges',
    title="Causes de rupture (2021–2026)"
)
fig5.update_xaxes(tickangle=30)
fig5.update_coloraxes(colorbar_title="Nombre de ruptures")
col2.plotly_chart(fig5, width='stretch')

st.markdown("---")

# ── Section 5 : Jointure médicaments × ruptures ─────────────
st.subheader("🔗 Médicaments les plus prescrits en rupture")
jointure = anti_med.merge(anti_rup_join, on='cis', how='inner')

if len(jointure) > 0:
    top_rup = (jointure.groupby('molecule_x')['nb_patients_ville_x']
               .mean().sort_values(ascending=False).head(10).reset_index())
    top_rup.columns = ['molecule', 'nb_patients_moyen']
    fig6 = px.bar(
        top_rup, x='nb_patients_moyen', y='molecule', orientation='h',
        labels={'nb_patients_moyen': 'Nombre moyen de patients', 'molecule': 'Molécule'},
        color='nb_patients_moyen', color_continuous_scale='RdYlGn',
        title=f"Molécules {code_atc} en rupture — nombre moyen de patients (2021-2024)"
    )
    fig6.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig6.update_coloraxes(colorbar_title="Nombre moyen de patients")
    st.plotly_chart(fig6, width='stretch')

    st.caption("ℹ️ Jointure sur CIS disponible pour 2021-2024 uniquement")
    st.markdown("---")

    # ── Scatter double risque ────────────────────────────────
    st.subheader("🎯 Molécules à double risque — Prescription vs Ruptures")

    scatter_data = jointure.groupby('molecule_x').agg(
        nb_ruptures=('cis', 'count'),
        nb_patients_moyen=('nb_patients_ville_x', 'mean')
    ).reset_index()
    scatter_data.columns = ['molecule', 'nb_ruptures', 'nb_patients_moyen']
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
    st.plotly_chart(fig_scatter, width='stretch')
    st.caption("💡 Les molécules en haut à droite sont les plus critiques — très prescrites ET souvent en rupture")

else:
    st.info("Pas de données disponibles pour cette classe ATC.")

st.divider()
st.caption("Projet Antihistaminiques — Jedha 2026 — LMN")
