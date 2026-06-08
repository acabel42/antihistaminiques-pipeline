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

@st.cache_data
def load_data(code):
    pred_path = f'data/gold/gold_predictions_{code}.csv'
    if not os.path.exists(pred_path):
        pred_path = f'data/gold/gold_predictions_{code}.csv'
    if not os.path.exists(pred_path):
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

st.sidebar.title("🔧 Filtres")

classe_select = st.sidebar.selectbox(
    "Classe ATC",
    ['R06 — Antihistaminiques', 'R03 — Antiasthmatiques', 'J01 — Antibiotiques'],
    index=0
)
code_atc = classe_select.split(' ')[0]

pred, gold = load_data(code_atc)

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

annees_dispo = ['Toutes'] + sorted(df['annee'].unique().tolist())
annee_select = st.sidebar.selectbox("Année", annees_dispo, index=0)

if annee_select == 'Toutes':
    df_f = df.copy()
else:
    df_f = df[df['annee'] == annee_select]

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
roc_map = {'R06': '0.738', 'R03': '0.933', 'J01': '0.981'}
st.sidebar.markdown(f"**ROC-AUC CV :** {roc_map.get(code_atc, '0.771')}")
st.sidebar.markdown("**Seuil :** probabilité > 50%")
st.sidebar.markdown("**Légende :**")
st.sidebar.markdown("🟢 Vrai négatif | 🔴 Faux positif")
st.sidebar.markdown("🟢 Vrai positif | 🟠 Faux négatif")

st.divider()

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
    _prec = {"R06": "56%", "R03": "83%", "J01": "89%"}
    st.metric("Précision modèle", _prec.get(code_atc, "67%"),
              help="Accuracy du RF Classifier sur le jeu de test")
with col4:
    roc_map = {'R06': '0.738', 'R03': '0.933', 'J01': '0.981'}
    st.metric("ROC-AUC CV", roc_map.get(code_atc, '0.771'),
              help="Aire sous la courbe ROC (cross-validation) — 1.0 = parfait, 0.5 = hasard")

st.divider()

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

st.subheader("📊 Variables les plus importantes pour prédire les ruptures")
st.caption("Plus la barre est longue, plus la variable influence la prédiction du modèle.")

import joblib

try:
    clf_path = f'models/{code_atc}/rf_classifier.joblib'
    if not os.path.exists(clf_path):
        clf_path = 'models/rf_classifier.joblib'
    clf = joblib.load(clf_path)
    features_disponibles = list(clf.feature_names_in_)
    imp = pd.DataFrame({
        'feature': features_disponibles,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=True)

    fig_imp = px.bar(imp, x='importance', y='feature', orientation='h',
                     labels={'importance': 'Importance', 'feature': 'Variable'},
                     color='importance',
                     color_continuous_scale='Blues',
                     title=f"Feature Importance — RF Classifier {code_atc}")
    fig_imp.update_coloraxes(colorbar_title="Importance")
    st.plotly_chart(fig_imp, use_container_width=True)
    _cap = {"R06": "ruptures_lag1 et les pollens sont les signaux les plus prédictifs pour les antihistaminiques", "R03": "ruptures_lag1 et les indicateurs grippaux sont les plus prédictifs pour les antiasthmatiques", "J01": "ruptures_lag1 et la diarrhee sont les plus prédictifs pour les antibiotiques"}
    st.caption(f"💡 {_cap.get(code_atc, chr(32))}")

except Exception as e:
    st.warning(f"Modèle non disponible : {e}")

st.divider()

# =====================
# SECTION KPIs AVANCÉS
# =====================
st.subheader("🔬 KPIs Avancés")
st.divider()

col_k1, col_k2, col_k3 = st.columns(3)

# ── KPI 1 : Ratio ruptures / boites par année ──────────────────────────────
with col_k1:
    st.markdown("**🎯 Taux de détection des ruptures par année**")
    st.caption("Parmi les mois avec une vraie rupture, combien le modèle en a-t-il détectés ? Un taux élevé = peu de ruptures manquées.")

    detect_df = df.groupby('annee').apply(
        lambda g: pd.Series({
            'vrais_positifs': ((g['pred_rupture']==1) & (g['target_rupture']==1)).sum(),
            'total_ruptures': (g['target_rupture']==1).sum()
        })
    ).reset_index()

    # Recall = vrais positifs / total ruptures réelles
    detect_df['taux_detection'] = (
        detect_df['vrais_positifs'] / detect_df['total_ruptures'].replace(0, 1) * 100
    ).round(1)

    fig_detect = px.bar(
        detect_df, x='annee', y='taux_detection',
        labels={'annee': 'Année', 'taux_detection': 'Taux de détection (%)'},
        color='taux_detection',
        color_continuous_scale='Greens',
        text='taux_detection'
    )
    fig_detect.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_detect.update_coloraxes(showscale=False)
    fig_detect.update_layout(height=300, yaxis=dict(range=[0, 110]))
    st.plotly_chart(fig_detect, use_container_width=True)

# ── KPI 2 : Trend annuel des probabilités de rupture ──────────────────────
with col_k2:
    st.markdown("**📈 Trend annuel — probabilité de rupture moyenne**")
    st.caption("Évolution de la probabilité moyenne de rupture par année. Une tendance à la hausse indique une dégradation de la situation des stocks.")

    trend_df = df.groupby('annee')['proba_pct'].mean().reset_index()
    trend_df.columns = ['annee', 'proba_moy']

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_df['annee'],
        y=trend_df['proba_moy'],
        mode='lines+markers',
        line=dict(color='#2471a3', width=2),
        marker=dict(size=8)
    ))
    # Ligne de référence à 50%
    fig_trend.add_hline(y=50, line_dash='dash', line_color='red',
                        annotation_text='Seuil 50%')
    fig_trend.update_layout(
        xaxis_title='Année',
        yaxis_title='Probabilité moyenne (%)',
        height=300
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ── KPI 3 : Précision / Recall par mois ───────────────────────────────────
with col_k3:
    st.markdown("**🎯 Précision & Recall par mois**")
    st.caption("Précision = quand le modèle dit rupture, a-t-il raison ? Recall = parmi les vraies ruptures, combien a-t-il détecté ?")

    # Calcul précision et recall mois par mois sur toutes les années
    def calc_precision_recall(group):
        tp = ((group['pred_rupture'] == 1) & (group['target_rupture'] == 1)).sum()
        fp = ((group['pred_rupture'] == 1) & (group['target_rupture'] == 0)).sum()
        fn = ((group['pred_rupture'] == 0) & (group['target_rupture'] == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        return pd.Series({'precision': round(precision, 2), 'recall': round(recall, 2)})

    pr_df = df.groupby('mois').apply(calc_precision_recall).reset_index()
    pr_df['mois_label'] = pr_df['mois'].map(mois_noms_court)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=pr_df['mois_label'], y=pr_df['precision'],
        mode='lines+markers', name='Précision',
        line=dict(color='#2ecc71', width=2)
    ))
    fig_pr.add_trace(go.Scatter(
        x=pr_df['mois_label'], y=pr_df['recall'],
        mode='lines+markers', name='Recall',
        line=dict(color='#e74c3c', width=2)
    ))
    fig_pr.update_layout(
        xaxis_title='Mois',
        yaxis_title='Score (0-1)',
        yaxis=dict(range=[0, 1.1]),
        legend=dict(x=0, y=1.1, orientation='h'),
        height=300
    )
    st.plotly_chart(fig_pr, use_container_width=True)

st.divider()

# =====================
# SECTION 6 — Courbe ROC
# =====================
st.subheader("📈 Courbe ROC — Performance du modèle")
st.caption("La courbe ROC montre le compromis entre vrais positifs et faux positifs pour tous les seuils. Plus la courbe est proche du coin supérieur gauche, meilleur est le modèle. La diagonale pointillée = prédiction aléatoire.")

from sklearn.metrics import roc_curve, auc

try:
    clf = joblib.load(f'models/{code_atc}/rf_classifier.joblib')

    features_disponibles = [
        'gram_moy', 'gram_max', 'gram_roll7', 'gram_roll30', 'nb_jours_pic',
        'bouleau_moy', 'ambroisie_moy', 'nb_jours_pic_bouleau',
        'temp_moy', 'temp_max', 'temp_roll30',
        'precip', 'wind', 'mois', 'saison_allergies', 'source_encoded',
        'ruptures_lag1', 'gram_lag_mois', 'cumul_thermique'
    ]

   # Pour R06 on utilise gold_ml_advanced qui a toutes les features
    # Pour R03 et J01 on utilise leur gold respectif
    if code_atc == 'R06':
        gold_adv = pd.read_csv('data/gold/gold_ml_advanced.csv')
    else:
        gold_adv = pd.read_csv(f'data/gold/gold_ml_{code_atc}.csv')

    # On garde seulement les features que le modèle ET le fichier ont en commun
    features_model = list(clf.feature_names_in_)
    features_disponibles = [f for f in features_model if f in gold_adv.columns]

    df_roc = gold_adv.dropna(subset=features_disponibles + ['target_rupture'])
    X_roc = df_roc[features_disponibles]
    y_roc = df_roc['target_rupture']

    # predict_proba donne la probabilité pour chaque seuil possible
    y_prob_roc = clf.predict_proba(X_roc)[:, 1]
    fpr, tpr, _ = roc_curve(y_roc, y_prob_roc)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()

    # Courbe ROC du modèle
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'RF Classifier (AUC = {roc_auc:.3f})',
        line=dict(color='#2471a3', width=2)
    ))

    # Diagonale = modèle aléatoire (référence)
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Aléatoire (AUC = 0.5)',
        line=dict(color='gray', width=1, dash='dash')
    ))

    fig_roc.update_layout(
        xaxis_title='Taux de faux positifs',
        yaxis_title='Taux de vrais positifs',
        legend=dict(x=0.6, y=0.1),
        height=400
    )

    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption(f"💡 AUC = {roc_auc:.3f} — le modèle est {round(roc_auc*100 - 50)}% meilleur qu'une prédiction aléatoire")

except Exception as e:
    st.warning(f"Courbe ROC non disponible : {e}")

st.divider()

st.subheader("🔮 Prédiction en direct — Appel API")
st.caption("Renseignez les conditions du mois à analyser. Le dashboard interroge l'API FastAPI qui charge le modèle et retourne une prédiction en temps réel.")

import requests

API_URL = "http://127.0.0.1:8000"

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("**🌿 Pollen**")
    gram_moy       = st.slider("Graminées moy (g/m³)", 0.0, 100.0, 10.0)
    gram_max       = st.slider("Graminées max (g/m³)", 0.0, 200.0, 30.0)
    ambroisie_moy  = st.slider("Ambroisie moy (g/m³)", 0.0, 50.0, 3.0)
    bouleau_moy    = st.slider("Bouleau moy (g/m³)",   0.0, 50.0, 5.0)
    nb_jours_pic   = st.slider("Jours pic graminées",  0, 31, 3)

with col_f2:
    st.markdown("**🌡️ Météo**")
    temp_moy    = st.slider("Température moy (°C)", -5.0, 40.0, 18.0)
    temp_max    = st.slider("Température max (°C)", -5.0, 45.0, 25.0)
    precip      = st.slider("Précipitations (mm)",   0.0, 200.0, 40.0)
    wind        = st.slider("Vent moy (km/h)",       0.0, 100.0, 12.0)

with col_f3:
    st.markdown("**📅 Contexte**")
    mois_api           = st.slider("Mois", 1, 12, 5)
    saison_allergies   = st.selectbox("Saison allergies", [0, 1], index=1)
    ruptures_lag1      = st.selectbox("Rupture mois précédent", [0.0, 1.0], index=0)

gram_roll7         = gram_moy * 0.85
gram_roll30        = gram_moy * 0.70
gram_lag_mois      = gram_moy * 0.60
cumul_thermique    = temp_moy * mois_api * 3.5
nb_jours_pic_bouleau = max(0, int(bouleau_moy / 5))
temp_roll30        = temp_moy - 1.5
source_encoded     = 0.5

if st.button("🚀 Lancer la prédiction", type="primary"):
    payload = {
        "gram_moy": gram_moy, "gram_max": gram_max,
        "gram_roll7": gram_roll7, "gram_roll30": gram_roll30,
        "nb_jours_pic": nb_jours_pic, "bouleau_moy": bouleau_moy,
        "ambroisie_moy": ambroisie_moy, "nb_jours_pic_bouleau": nb_jours_pic_bouleau,
        "temp_moy": temp_moy, "temp_max": temp_max, "temp_roll30": temp_roll30,
        "precip": precip, "wind": wind, "mois": mois_api,
        "saison_allergies": saison_allergies, "source_encoded": source_encoded,
        "ruptures_lag1": ruptures_lag1, "gram_lag_mois": gram_lag_mois,
        "cumul_thermique": cumul_thermique
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        if result["rupture_predite"] == 1:
            st.error(f"🚨 {result['interpretation']} — probabilité : {result['probabilite_rupture']*100:.1f}%")
        else:
            st.success(f"✅ {result['interpretation']} — probabilité : {result['probabilite_rupture']*100:.1f}%")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ L'API FastAPI ne tourne pas. Lancez : uvicorn src.api.main:app --reload")
    except Exception as e:
        st.error(f"Erreur : {e}")

st.caption("Projet Antihistaminiques — Jedha 2026 — LMN")

