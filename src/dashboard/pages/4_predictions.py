import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Prédictions ML", page_icon="🤖", layout="wide")
st.title("🤖 Prédictions ML — Risque de rupture")
st.divider()

# =====================
# CHARGEMENT DONNEES
# =====================
@st.cache_data
def load_data(code):
    pred_path = 'data/gold/gold_predictions.csv'
    gold_path = f'data/gold/gold_ml_{code}.csv'
    if not os.path.exists(gold_path):
        gold_path = 'data/gold/gold_ml.csv'
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path)
    else:
        st.error("Fichier gold_predictions.csv non trouvé — lancez predict.py")
        st.stop()
    gold = pd.read_csv(gold_path)
    return pred, gold

# =====================
# SIDEBAR
# =====================
st.sidebar.title("🔧 Filtres")

# Filtre classe ATC — EN PREMIER
classe_select = st.sidebar.selectbox(
    "Classe ATC",
    ['R06 — Antihistaminiques', 'R03 — Antiasthmatiques', 'J01 — Antibiotiques'],
    index=0
)
code_atc = classe_select.split(' ')[0]

pred, gold = load_data(code_atc)

# Fusion prédictions + données réelles
df = pred.merge(gold[['annee_mois_str','mois','annee','nb_ruptures',
                        'target_rupture','gram_moy','temp_moy','ambroisie_moy']],
                on='annee_mois_str', how='left')

df['proba_pct'] = (df['proba_rupture'] * 100).round(1)
df['risque_label'] = df['pred_rupture'].apply(
    lambda x: '🚨 Rupture prédite' if x == 1 else '✅ Pas de risque'
)

mois_noms_court = {1:'Jan',2:'Fev',3:'Mar',4:'Avr',5:'Mai',6:'Jun',
                   7:'Jul',8:'Aou',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

st.sidebar.divider()

# Filtre année
annees_dispo = ['Toutes'] + sorted(df['annee'].unique().tolist())
annee_select = st.sidebar.selectbox("Année", annees_dispo, index=0)

if annee_select == 'Toutes':
    df_f = df.copy()
else:
    df_f = df[df['annee'] == annee_select]

# Filtre mois pour la jauge
mois_select = st.sidebar.selectbox(
    "Mois (pour la jauge)",
    [mois_noms_court[m] for m in sorted(df_f['mois'].unique().tolist())],
    index=0
)
mois_num = {v:k for k,v in mois_noms_court.items()}[mois_select]
df_mois = df_f[df_f['mois'] == mois_num]

st.sidebar.divider()
st.sidebar.markdown(f"**Classe : {code_atc}**")
st.sidebar.markdown(f"**{(df_f['pred_rupture']==1).sum()} mois à risque** sur {len(df_f)}")
st.sidebar.markdown("**Modèle :** RF Classifier")
st.sidebar.markdown("**ROC-AUC :** 0.771")
st.sidebar.markdown("**Seuil :** probabilité > 50%")
st.sidebar.markdown("**Légende :**")
st.sidebar.markdown("🟢 Vrai négatif | 🔴 Faux positif")
st.sidebar.markdown("🟢 Vrai positif | 🟠 Faux négatif")

st.divider()

# =====================
# SECTION 1 — KPIs
# =====================
st.subheader(f"📊 KPIs Prédictions — {classe_select}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    nb_risque = (df_f['pred_rupture']==1).sum()
    st.metric("Mois à risque prédit", f"{nb_risque} / {len(df_f)}",
              help="Nombre de mois où le modèle prédit une rupture")
with col2:
    proba_max = df_f['proba_pct'].max()
    mois_max = df_f.loc[df_f['proba_pct'].idxmax(), 'annee_mois_str']
    st.metric("Probabilité max", f"{proba_max:.1f}%",
              help=f"Mois le plus à risque : {mois_max}")
with col3:
    st.metric("Précision modèle", "67%",
              help="Precision du RF Classifier sur le jeu de test")
with col4:
    st.metric("ROC-AUC", "0.771",
              help="Aire sous la courbe ROC — 1.0 = parfait, 0.5 = hasard")

st.divider()

# =====================
# SECTION 2 — Jauge + Timeline
# =====================
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader(f"🎯 Jauge de risque — {mois_select}")
    st.caption("Probabilité moyenne de rupture pour le mois sélectionné. Rouge = risque élevé, vert = risque faible. Seuil de décision = 50%.")

    if len(df_mois) > 0:
        proba_mois = df_mois['proba_rupture'].mean()
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba_mois * 100,
            delta={'reference': 50, 'valueformat': '.1f'},
            number={'suffix': '%', 'valueformat': '.1f'},
            title={'text': f"Risque rupture {code_atc} — {mois_select}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#e74c3c' if proba_mois > 0.5 else '#2ecc71'},
                'steps': [
                    {'range': [0, 30], 'color': '#d5f5e3'},
                    {'range': [30, 50], 'color': '#fdebd0'},
                    {'range': [50, 100], 'color': '#fadbd8'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        if proba_mois > 0.5:
            st.error(f"🚨 Risque de rupture élevé en {mois_select} — probabilité : {proba_mois*100:.1f}%")
        else:
            st.success(f"✅ Risque faible en {mois_select} — probabilité : {proba_mois*100:.1f}%")
    else:
        st.warning("Pas de données pour ce mois.")

with col_g2:
    st.subheader("📈 Timeline des probabilités")
    st.caption("Probabilité de rupture par mois. La ligne noire à 50% est le seuil de décision. Au-dessus = rupture prédite.")

    fig_timeline = px.bar(
        df_f, x='annee_mois_str', y='proba_pct',
        color='risque_label',
        color_discrete_map={
            '✅ Pas de risque': '#2ecc71',
            '🚨 Rupture prédite': '#e74c3c'
        },
        labels={
            'annee_mois_str': 'Mois',
            'proba_pct': 'Probabilité (%)',
            'risque_label': 'Prédiction'
        }
    )
    fig_timeline.add_hline(
        y=50, line_dash='dash', line_color='black',
        annotation_text='Seuil 50%', annotation_position='top right'
    )
    fig_timeline.update_xaxes(tickangle=45)
    fig_timeline.update_layout(
        legend=dict(x=0, y=1.1, orientation='h'),
        height=350
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

st.divider()

# =====================
# SECTION 3 — Réel vs Prédit
# =====================
col_g3, col_g4 = st.columns(2)

with col_g3:
    st.subheader("🔄 Réel vs Prédit — Graminées")
    st.caption("Courbe verte = concentration réelle. Courbe orange = prédiction du modèle pour le mois suivant. Plus les courbes sont proches, meilleur est le modèle.")

    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=df_f['annee_mois_str'],
        y=df_f['gram_moy'],
        name='Réel (g/m³)',
        line=dict(color='#2ecc71', width=2)
    ))
    fig_rv.add_trace(go.Scatter(
        x=df_f['annee_mois_str'],
        y=df_f['pred_gram_next'],
        name='Prédit mois suivant',
        line=dict(color='#e67e22', width=2, dash='dot')
    ))
    fig_rv.update_layout(
        xaxis=dict(title='Mois', tickangle=45),
        yaxis_title='Graminées (g/m³)',
        legend=dict(x=0, y=1.1, orientation='h'),
        height=380
    )
    st.plotly_chart(fig_rv, use_container_width=True)

with col_g4:
    st.subheader("🎯 Ruptures réelles vs prédites")
    st.caption("Vert foncé = bien prédit (rupture). Bleu = bien prédit (pas de rupture). Rouge = faux positif. Orange = faux négatif (rupture manquée).")

    df_f = df_f.copy()
    df_f['statut'] = df_f.apply(lambda r:
        'Vrai positif' if r['pred_rupture']==1 and r['target_rupture']==1
        else 'Vrai négatif' if r['pred_rupture']==0 and r['target_rupture']==0
        else 'Faux positif' if r['pred_rupture']==1 and r['target_rupture']==0
        else 'Faux négatif', axis=1
    )

    fig_conf = px.bar(
        df_f, x='annee_mois_str', y='proba_pct',
        color='statut',
        color_discrete_map={
            'Vrai positif':  '#2ecc71',
            'Vrai négatif':  '#85c1e9',
            'Faux positif':  '#e74c3c',
            'Faux négatif':  '#e67e22'
        },
        labels={
            'annee_mois_str': 'Mois',
            'proba_pct': 'Probabilité (%)',
            'statut': 'Statut prédiction'
        }
    )
    fig_conf.add_hline(y=50, line_dash='dash', line_color='black',
                       annotation_text='Seuil 50%')
    fig_conf.update_xaxes(tickangle=45)
    fig_conf.update_layout(
        legend=dict(x=0, y=1.1, orientation='h'),
        height=380
    )
    st.plotly_chart(fig_conf, use_container_width=True)

st.divider()

# =====================
# SECTION 4 — Tableau alertes
# =====================
st.subheader(f"🚨 Tableau des alertes — Mois à risque {code_atc}")
st.caption("Liste des mois où le modèle prédit une rupture, triés par probabilité décroissante.")

df_alertes = df_f[df_f['pred_rupture']==1].sort_values('proba_pct', ascending=False)

if len(df_alertes) > 0:
    st.dataframe(
        df_alertes[['annee_mois_str','proba_pct','pred_gram_next',
                    'gram_moy','nb_ruptures','target_rupture']].rename(columns={
            'annee_mois_str': 'Mois',
            'proba_pct': 'Probabilité (%)',
            'pred_gram_next': 'Gram prédit mois suivant',
            'gram_moy': 'Gram réel',
            'nb_ruptures': 'Nb ruptures réelles',
            'target_rupture': 'Rupture confirmée'
        }),
        use_container_width=True
    )
else:
    st.success("✅ Aucun mois à risque sur la période sélectionnée.")

st.divider()
st.caption("Projet Antihistaminiques — Jedha 2026 — LMN")