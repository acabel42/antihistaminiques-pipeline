import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Pollen & Prédictions", page_icon="🌿", layout="wide")
st.title("🌿 Analyse Pollen & Prédictions ML")

# =====================
# CHARGEMENT DONNEES
# =====================
@st.cache_data
def load_gold_atc(code):
    path = f'data/gold/gold_ml_{code}.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.read_csv('data/gold/gold_ml.csv')

@st.cache_data
def load_daily():
    return pd.read_csv('data/gold/pollen_meteo_features.csv', parse_dates=['date'])

# =====================
# MAPPING GLOBAL
# =====================
taxon_cols = {
    'Graminées': 'gram_moy',
    'Bouleau':   'bouleau_moy',
    'Ambroisie': 'ambroisie_moy',
    'Aulne':     'aulne_moy',
    'Armoise':   'armoise_moy',
    'Olivier':   'olivier_moy'
}
taxon_cols_daily = {
    'Graminées': 'graminees',
    'Bouleau':   'bouleau',
    'Ambroisie': 'ambroisie',
    'Aulne':     'aulne',
    'Armoise':   'armoise',
    'Olivier':   'olivier'
}
colors = {
    'gram_moy':     '#2ecc71',
    'bouleau_moy':  '#3498db',
    'ambroisie_moy':'#e67e22',
    'aulne_moy':    '#9b59b6',
    'armoise_moy':  '#e74c3c',
    'olivier_moy':  '#f1c40f'
}
fill_colors = {
    'gram_moy':     'rgba(46, 204, 113, 0.1)',
    'bouleau_moy':  'rgba(52, 152, 219, 0.1)',
    'ambroisie_moy':'rgba(230, 126, 34, 0.1)',
    'aulne_moy':    'rgba(155, 89, 182, 0.1)',
    'armoise_moy':  'rgba(231, 76, 60, 0.1)',
    'olivier_moy':  'rgba(241, 196, 15, 0.1)'
}
labels = {
    'gram_moy':     'Graminées',
    'bouleau_moy':  'Bouleau',
    'ambroisie_moy':'Ambroisie',
    'aulne_moy':    'Aulne',
    'armoise_moy':  'Armoise',
    'olivier_moy':  'Olivier'
}
mois_noms = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
             7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}
mois_noms_court = {1:'Jan',2:'Fev',3:'Mar',4:'Avr',5:'Mai',6:'Jun',
                   7:'Jul',8:'Aou',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
year_colors = {2021:'#95a5a6', 2022:'#7f8c8d', 2023:'#3498db',
               2024:'#e74c3c', 2025:'#2ecc71', 2026:'#9b59b6'}

ALL_TAXONS = list(taxon_cols.keys())

# =====================
# SIDEBAR
# =====================
st.sidebar.title("🔧 Filtres")

classe_select = st.sidebar.selectbox(
    "Classe ATC",
    ['R06 — Antihistaminiques', 'R03 — Antiasthmatiques', 'J01 — Antibiotiques'],
    index=0
)
code_atc = classe_select.split(' ')[0]
df = load_gold_atc(code_atc)

st.sidebar.divider()

annees_dispo = ['Toutes'] + sorted(df['annee'].unique().tolist())
annee_select = st.sidebar.selectbox("Année", annees_dispo, index=0)

if annee_select == 'Toutes':
    df_filtered = df.copy()
else:
    df_filtered = df[df['annee'] == annee_select]

mois_dispo = ['Tous'] + [mois_noms[m] for m in sorted(df_filtered['mois'].unique().tolist())]
mois_select = st.sidebar.selectbox("Mois", mois_dispo, index=0)

if mois_select != 'Tous':
    mois_num = {v:k for k,v in mois_noms.items()}[mois_select]
    df_filtered = df_filtered[df_filtered['mois'] == mois_num]

st.sidebar.divider()

taxon_select = st.sidebar.selectbox(
    "Taxon de pollen",
    ['Tous'] + ALL_TAXONS,
    index=0
)

mode_tous = taxon_select == 'Tous'

if mode_tous:
    cols_actifs = [c for c in taxon_cols.values() if c in df_filtered.columns]
    col_selected = 'gram_moy'
    col_daily = 'graminees'
    color_selected = colors['gram_moy']
else:
    col_selected = taxon_cols[taxon_select]
    col_daily = taxon_cols_daily[taxon_select]
    color_selected = colors[col_selected]
    cols_actifs = [col_selected]

if col_selected not in df_filtered.columns:
    st.error(f"Colonne {col_selected} non disponible — relancez features_pollen.py et build_gold.py")
    st.stop()

st.sidebar.divider()
st.sidebar.markdown(f"**{len(df_filtered)} mois sélectionnés**")
st.sidebar.markdown(f"**Classe : {code_atc}**")
st.sidebar.markdown(f"**Taxon : {taxon_select}**")
st.sidebar.markdown("**Légende :**")
st.sidebar.markdown("🟢 Graminées | 🔵 Bouleau | 🟠 Ambroisie")
st.sidebar.markdown("🟣 Aulne | 🔴 Armoise | 🟡 Olivier")
st.sidebar.markdown("⭐ Rupture")

st.divider()

# =====================
# SECTION 1 — KPIs
# =====================
st.subheader(f"📊 KPIs — {classe_select} — {taxon_select}")
st.write("")

col1, col2, col3, col4 = st.columns(4)
with col1:
    taux = df_filtered['target_rupture'].mean() * 100
    st.metric("Taux de rupture", f"{taux:.1f}%")
with col2:
    if mode_tous:
        max_val = max(df_filtered[c].max() for c in cols_actifs)
        max_tax = labels[max(cols_actifs, key=lambda c: df_filtered[c].max())]
        st.metric("Concentration max", f"{max_val:.1f} g/m³", help=f"Taxon : {max_tax}")
    else:
        st.metric(f"{taxon_select} max", f"{df_filtered[col_selected].max():.1f} g/m³")
with col3:
    st.metric("Mois en tension", f"{int(df_filtered['target_rupture'].sum())} / {len(df_filtered)}")
with col4:
    if len(df_filtered) > 0:
        mois_pic = df_filtered.groupby('mois')[col_selected].mean().idxmax()
        st.metric("Mois pic", mois_noms_court[mois_pic])
    else:
        st.metric("Mois pic", "N/A")

st.divider()

# =====================
# LIGNE 1 — Evolution + Heatmap
# =====================
col_g1, col_g2 = st.columns(2)

with col_g1:
    titre = "📈 Evolution — Tous les pollens" if mode_tous else f"📈 Evolution — {taxon_select}"
    st.subheader(titre)
    st.caption("Sources : RNSA 2021-2022 (capteurs physiques) — CAMS 2023-2026 (modèle satellite).")
    st.write("")

    fig1 = go.Figure()
    if mode_tous:
        for col in cols_actifs:
            fig1.add_trace(go.Scatter(
                x=df_filtered['annee_mois_str'],
                y=df_filtered[col],
                name=labels[col],
                line=dict(color=colors[col], width=2),
                stackgroup='one'
            ))
    else:
        fig1.add_trace(go.Scatter(
            x=df_filtered['annee_mois_str'],
            y=df_filtered[col_selected],
            name=taxon_select,
            line=dict(color=color_selected, width=2),
            fill='tozeroy',
            fillcolor=fill_colors[col_selected]
        ))
    fig1.add_vrect(x0='2021-01', x1='2022-12', fillcolor='lightblue', opacity=0.1,
                   annotation_text='RNSA', annotation_position='top left')
    fig1.add_vrect(x0='2023-01', x1='2026-06', fillcolor='lightyellow', opacity=0.1,
                   annotation_text='CAMS', annotation_position='top left')
    fig1.update_layout(
        xaxis_title='Mois', yaxis_title='Concentration (grains/m³)',
        xaxis=dict(tickangle=45), legend=dict(x=0, y=1.1, orientation='h'), height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_g2:
    heatmap_col = 'gram_moy' if mode_tous else col_selected
    heatmap_label = 'Graminées' if mode_tous else taxon_select
    st.subheader(f"🗓️ Heatmap — {heatmap_label}")
    st.caption(f"Concentration moyenne de {heatmap_label} par mois et par année.")
    st.write("")

    pivot = df.pivot_table(values=heatmap_col, index='annee', columns='mois', aggfunc='mean').round(2)
    pivot.columns = [mois_noms_court[m] for m in pivot.columns]
    fig2 = px.imshow(pivot, color_continuous_scale='YlOrRd',
                     labels=dict(x='Mois', y='Année', color='Grains/m³'),
                     aspect='auto', text_auto=True)
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =====================
# LIGNE 2 — Corrélations + Timeline étoiles
# =====================
col_g3, col_g4 = st.columns(2)

with col_g3:
    st.subheader(f"📊 Corrélations pollens vs Ruptures {code_atc}")
    st.caption("Corrélation de Pearson entre chaque taxon et les ruptures. Rouge = signal le plus fort.")
    st.write("")

    taxon_corr_cols = {
        'Graminées': 'gram_moy', 'Bouleau': 'bouleau_moy',
        'Ambroisie': 'ambroisie_moy', 'Aulne': 'aulne_moy',
        'Armoise': 'armoise_moy', 'Olivier': 'olivier_moy'
    }
    corr_data = []
    for nom, col in taxon_corr_cols.items():
        if col in df.columns:
            corr = df[col].corr(df['target_rupture'])
            corr_data.append({'Taxon': nom, 'Corrélation': round(corr, 3)})

    df_corr = pd.DataFrame(corr_data).sort_values('Corrélation', ascending=True)
    df_corr['couleur'] = df_corr['Corrélation'].apply(
        lambda x: '#e74c3c' if x == df_corr['Corrélation'].max() else '#3498db'
    )

    fig3 = go.Figure(go.Bar(
        x=df_corr['Corrélation'], y=df_corr['Taxon'], orientation='h',
        marker_color=df_corr['couleur'],
        text=df_corr['Corrélation'].apply(lambda x: f'{x:.3f}'),
        textposition='outside'
    ))
    fig3.add_vline(x=0, line_color='black', line_width=1)
    fig3.update_layout(
        xaxis_title='Corrélation de Pearson', yaxis_title='',
        xaxis=dict(range=[-0.5, 0.6]), height=400
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_g4:
    titre4 = "💊 Tous les pollens vs Ruptures — timeline" if mode_tous else f"💊 {taxon_select} vs Ruptures — timeline"
    st.subheader(titre4)
    st.caption("Courbes = concentration pollen. Étoiles rouges = mois avec rupture.")
    st.write("")

    df_filtered = df_filtered.copy()
    df_filtered['rupture_label'] = df_filtered['target_rupture'].apply(
        lambda x: 'Rupture' if x == 1 else 'Pas de rupture'
    )
    df_ruptures = df_filtered[df_filtered['target_rupture'] == 1]

    fig4 = go.Figure()
    for col in cols_actifs:
        fig4.add_trace(go.Scatter(
            x=df_filtered['annee_mois_str'],
            y=df_filtered[col],
            name=labels[col],
            line=dict(color=colors[col], width=2),
            fill='tozeroy' if not mode_tous else None,
            fillcolor=fill_colors[col] if not mode_tous else None
        ))

    # Étoiles ruptures sur le taxon principal
    y_etoiles = df_ruptures[col_selected] if col_selected in df_ruptures.columns else df_ruptures['gram_moy']
    fig4.add_trace(go.Scatter(
        x=df_ruptures['annee_mois_str'],
        y=y_etoiles,
        name=f'Rupture {code_atc}',
        mode='markers',
        marker=dict(symbol='star', size=16, color='#e74c3c',
                    line=dict(color='#c0392b', width=1))
    ))

    fig4.update_layout(
        xaxis=dict(title='Mois', tickangle=45),
        yaxis_title='Concentration (grains/m³)',
        legend=dict(x=0, y=1.1, orientation='h'),
        height=400
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# =====================
# LIGNE 3 — Analyse comparative journalière
# =====================
st.subheader(f"🔍 Analyse comparative — {taxon_select} par année (données journalières)")
st.write("")

annees_comp = st.multiselect(
    "Sélectionner les années à comparer",
    [2021, 2022, 2023, 2024, 2025],
    default=[2023, 2024, 2025]
)

df_daily = load_daily()

if annees_comp:
    df_daily_comp = df_daily[df_daily['annee'].isin(annees_comp)].copy()
    df_daily_comp['jour_annee'] = df_daily_comp['date'].dt.dayofyear

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        label_g = "Tous les pollens" if mode_tous else taxon_select
        st.caption(f"{label_g} par jour de l'année")
        st.write("")
        fig_g = go.Figure()
        for annee in annees_comp:
            df_y = df_daily_comp[df_daily_comp['annee']==annee].sort_values('jour_annee')
            if mode_tous:
                for t, cd in taxon_cols_daily.items():
                    if cd in df_y.columns:
                        fig_g.add_trace(go.Scatter(
                            x=df_y['jour_annee'], y=df_y[cd],
                            name=f'{t} {annee}',
                            line=dict(color=colors[taxon_cols[t]], width=1.5),
                            mode='lines', showlegend=(annee == annees_comp[0])
                        ))
            else:
                if col_daily in df_y.columns:
                    fig_g.add_trace(go.Scatter(
                        x=df_y['jour_annee'], y=df_y[col_daily],
                        name=str(annee),
                        line=dict(color=year_colors.get(annee, '#333'), width=1.5),
                        mode='lines'
                    ))
        fig_g.update_layout(
            title=f'{label_g} par jour',
            xaxis_title="Jour de l'année", yaxis_title='Grains/m³',
            legend=dict(x=0.7, y=1), height=380
        )
        if 2024 in annees_comp:
            fig_g.add_annotation(
                x=160, y=5,
                text="⚠️ 2024 Anomalie",
                showarrow=True, arrowhead=2,
                bgcolor='#fff3cd', bordercolor='#e67e22'
            )
        st.plotly_chart(fig_g, use_container_width=True)

    with col_a2:
        st.caption("Température par jour de l'année")
        st.write("")
        fig_t = go.Figure()
        for annee in annees_comp:
            df_y = df_daily_comp[df_daily_comp['annee']==annee].sort_values('jour_annee')
            fig_t.add_trace(go.Scatter(
                x=df_y['jour_annee'], y=df_y['temp_moy'],
                name=str(annee),
                line=dict(color=year_colors.get(annee, '#333'), width=1.5),
                mode='lines'
            ))
        fig_t.update_layout(
            title=f'Température par jour',
            xaxis_title="Jour de l'année", yaxis_title='Température (°C)',
            legend=dict(x=0.8, y=1), height=380
        )
        st.plotly_chart(fig_t, use_container_width=True)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.caption("Précipitations par jour de l'année")
        st.write("")
        fig_p = go.Figure()
        for annee in annees_comp:
            df_y = df_daily_comp[df_daily_comp['annee']==annee].sort_values('jour_annee')
            fig_p.add_trace(go.Scatter(
                x=df_y['jour_annee'], y=df_y['precip'],
                name=str(annee),
                line=dict(color=year_colors.get(annee, '#333'), width=1),
                mode='lines'
            ))
        fig_p.update_layout(
            title='Précipitations par jour',
            xaxis_title="Jour de l'année", yaxis_title='Précipitations (mm)',
            legend=dict(x=0.8, y=1), height=380
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with col_b2:
        label_s = "Tous les pollens" if mode_tous else taxon_select
        st.caption(f"Distribution {label_s} par jour")
        st.write("")
        fig_s = go.Figure()
        for annee in annees_comp:
            df_y = df_daily_comp[df_daily_comp['annee']==annee].sort_values('jour_annee')
            if mode_tous:
                for t, cd in taxon_cols_daily.items():
                    if cd in df_y.columns:
                        fig_s.add_trace(go.Scatter(
                            x=df_y['jour_annee'], y=df_y[cd],
                            name=f'{t} {annee}',
                            mode='markers',
                            marker=dict(color=colors[taxon_cols[t]], size=3, opacity=0.5),
                            showlegend=(annee == annees_comp[0])
                        ))
            else:
                if col_daily in df_y.columns:
                    fig_s.add_trace(go.Scatter(
                        x=df_y['jour_annee'], y=df_y[col_daily],
                        name=str(annee), mode='markers',
                        marker=dict(color=year_colors.get(annee, '#333'), size=4, opacity=0.7)
                    ))
        fig_s.update_layout(
            title=f'Distribution {label_s}',
            xaxis_title="Jour de l'année", yaxis_title='Grains/m³',
            legend=dict(x=0.7, y=1), height=380
        )
        st.plotly_chart(fig_s, use_container_width=True)

    if 2024 in annees_comp:
        st.warning("⚠️ **2024 — Anomalie thermique** : concentrations très faibles au printemps. Température printanière 12.5°C vs 16.4°C en 2023 — pollinisation réduite de 66%.")

else:
    st.info("Sélectionne au moins une année pour afficher les graphiques.")

st.divider()
st.caption("Projet Antihistaminiques — Jedha 2026 — Collègue 3")