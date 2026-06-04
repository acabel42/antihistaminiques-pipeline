# 🌿 Antihistaminiques Pipeline

Projet Jedha Data Analysis Bootcamp — Prédiction des ruptures de stock d'antihistaminiques (ATC R06A) en France à partir de données pollen, météo et consommation médicaments.

---

## 🎯 Objectif

Anticiper les pics de demande en antihistaminiques R06A pour détecter proactivement les risques de rupture de stock, en croisant :
- les données de **consommation médicaments** (OpenMedic / CNAM)
- les données **polliniques** (CAMS Copernicus)
- les données **météorologiques** (13 régions françaises)
- les **signalements de ruptures** (ANSM / BDPM)

---

## 🏗️ Architecture

```
Bronze (raw)  →  Silver (nettoyé)  →  Gold (ML-ready)  →  Modèle  →  Dashboard
```

---

```
data/
├── bronze/
├── silver/
│   ├── J0_silver_medicaments.csv
│   ├── J0_silver_ruptures.csv
│   ├── J0_silver_bdpm.csv
│   ├── J0_silver_openmedic_2021_2025.csv
│   └── silver_meteo_2023_2026.csv
└── gold/
    ├── gold_ml.csv                  # 60 × 25
    └── gold_ml_advanced.csv         # 60 × 31

models/
├── lr_baseline.joblib
├── rf_classifier.joblib
└── rf_regressor.joblib

src/
├── transformations/
│   ├── build_gold.py
│   └── features_advanced.py
├── ml/
│   └── train_model.py
├── analysis/
│   └── kpis.py
└── api/
    └── main.py

pages/
├── 1_pollen.py
├── 2_ruptures.py
└── 3_openmedic.py

app.py
```

## ⚙️ Stack technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| ML | scikit-learn (Random Forest) |
| Dashboard | Streamlit + Plotly |
| API | FastAPI |
| Données spatiales | geopandas |
| Base de données | PostgreSQL (port 5432) |
| Containerisation | Docker |
| Versioning | Git / GitHub |

---

## 🚀 Lancement

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Exécuter le pipeline complet

```powershell
# Étape 1 — Construire la table Gold de base
python src/transformations/build_gold.py

# Étape 2 — Enrichir avec les features avancées
python src/transformations/features_advanced.py

# Étape 3 — Entraîner les modèles
python src/ml/train_model.py
```

### 3. Lancer le dashboard

```powershell
streamlit run app.py
```

---

## 🤖 Modèles ML

Deux modèles entraînés sur `gold_ml_advanced.csv` (60 lignes × 31 colonnes) :

### Classificateur — Risque de rupture
- **Algorithme** : Random Forest Classifier
- **Target** : `target_rupture` = `(nb_ruptures + nb_risques) > 0`
- **ROC-AUC** : 0.771
- **F1 CV** : 0.520 ± 0.176
- **Accuracy** : 0.67

### Régresseur — Intensité pollinique
- **Algorithme** : Random Forest Regressor
- **Target** : concentration pollinique gramineées (`gram_moy`)
- **R²** : 0.513
- **RMSE** : 6.300 grains/m³
- **R² CV** : 0.494 ± 0.257

### Features utilisées (19, sans data leakage)
`gram_moy`, `gram_max`, `gram_roll7`, `gram_roll30`, `nb_jours_pic`,
`bouleau_moy`, `ambroisie_moy`, `nb_jours_pic_bouleau`,
`temp_moy`, `temp_max`, `temp_roll30`, `precip`, `wind`,
`mois`, `saison_allergies`, `source_encoded`,
`ruptures_lag1`, `gram_lag_mois`, `cumul_thermique`

> ⚠️ `nb_ruptures` et `nb_risques` intentionnellement exclus (data leakage direct sur la target).

---

## 📦 Sources de données

| Dataset | Source | Période | Granularité |
|---|---|---|---|
| OpenMedic | CNAM | 2021–2025 | Région × molécule × année |
| Ruptures de stock | ANSM | — | National |
| BDPM | ANSM | — | Médicament |
| Météo | Données françaises | 2023–2026 | Région × mois |
| Pollen CAMS | Copernicus | 2023–2026 | Grille spatiale (local uniquement) |

> ℹ️ Les données CAMS (pollen) ne sont pas versionnées dans ce repo en raison de leur taille. Elles sont utilisées uniquement en local lors de la construction de la Gold table.

---

## 👥 Équipe

| | Responsabilités |
|---|---|
| **Collègue 1** | ERD/OLAP, run_pipeline.py, pages/2_ruptures.py |
| **Collègue 2 (Léo)** | OpenMedic Silver/Gold, EDA, pages/3_openmedic.py, README, dictionnaire données |
| **Collègue 3** | Pollen/météo, pipeline ML, features_advanced.py, pages/1_pollen.py |