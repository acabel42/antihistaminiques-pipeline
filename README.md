# 🌿 Antihistaminiques Pipeline

Projet Jedha Data Analysis Bootcamp — Prédiction des ruptures de stock d'antihistaminiques en France à partir de données pollen, météo, épidémiologie et consommation médicaments.

---

## 🎯 Objectif

Anticiper les pics de demande en antihistaminiques R06A pour détecter proactivement les risques de rupture de stock, en croisant :
- les données de **consommation médicaments** (OpenMedic / CNAM)
- les données **polliniques** (RNSA / CAMS Copernicus)
- les données **météorologiques** (13 régions françaises — Open-Meteo)
- les **signalements de ruptures** (ANSM 2021–2026)
- les données **épidémiologiques** (Sentinelles INSERM — grippe, IRA 2021–2026)

---

## 🏗️ Architecture

```
Sources → Bronze (raw) → Silver (nettoyé) → Gold (ML-ready) → ML → Dashboard / API
```

```
data/
├── bronze/
│   └── sources_bronze.txt
├── silver/
│   ├── J0_silver_medicaments.csv
│   ├── J0_silver_ruptures.csv
│   ├── J0_silver_ruptures_ansm_2026.csv    # ruptures 2025-2026
│   ├── J0_silver_bdpm.csv
│   ├── J0_silver_openmedic_2021_2025.csv
│   ├── J0_silver_meteo_2023_2026.csv
│   ├── J0_silver_pollen_2021_2026.csv
│   └── J0_silver_sentinelles.csv           # Sentinelles INSERM 2021-2026
└── gold/
    ├── gold_ml.csv                          # 60 × 25
    ├── gold_ml_R06.csv                      # 60 × 32 — antihistaminiques
    ├── gold_ml_R03.csv                      # 60 × 32 — antiasthmatiques
    ├── gold_ml_J01.csv                      # 60 × 32 — antibiotiques
    ├── gold_ml_advanced.csv                 # 60 × 38 — features enrichies
    ├── gold_predictions.csv
    ├── gold_predictions_R06.csv
    ├── gold_predictions_R03.csv
    └── gold_predictions_J01.csv

models/
├── lr_baseline.joblib
├── rf_baseline.joblib
├── rf_classifier.joblib
├── rf_regressor.joblib
├── R06/
│   ├── lr_baseline.joblib
│   ├── rf_classifier.joblib
│   └── rf_regressor.joblib
├── R03/
│   ├── lr_baseline.joblib
│   ├── rf_classifier.joblib
│   └── rf_regressor.joblib
└── J01/
    ├── lr_baseline.joblib
    ├── rf_classifier.joblib
    └── rf_regressor.joblib

src/
├── ingestion/
│   ├── ingest_meteo.py
│   ├── ingest_openmedic.py
│   └── ingest_sentinelles.py
├── cleaning/
│   ├── clean_medicaments_ruptures.py
│   ├── clean_openmedic.py
│   └── clean_pollen_meteo.py
├── transformations/
│   ├── build_gold.py
│   ├── features_advanced.py
│   ├── features_medicaments.py
│   ├── features_openmedic.py
│   ├── features_pollen.py
│   └── load_olap.py
├── ml/
│   ├── train_model.py
│   └── predict.py
├── analysis/
│   └── kpis.py
├── api/
│   └── main.py
└── dashboard/
    ├── app.py
    └── pages/
        ├── 1_pollen.py
        ├── 2_ruptures.py
        ├── 3_openmedic.py
        └── 4_predictions.py

notebooks/
├── eda_medicaments_ruptures.ipynb
├── eda_gold.ipynb
├── eda_pollen_complet.ipynb
└── eda_openmedic_bdpm.ipynb

docs/
├── schema_olap.png
├── schema_oltp.png
└── dictionnaire_donnees.md

.env                    # DB_URL Neon (jamais commité)
run_pipeline.py         # Orchestration complète avec logs
requirements.txt
```

---

## ⚙️ Stack technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| ML | scikit-learn 1.6.1 — Random Forest, Logistic Regression, SMOTE, GridSearchCV |
| Interprétabilité | SHAP |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + uvicorn |
| Base de données | Neon (PostgreSQL cloud) |
| ORM | SQLAlchemy + python-dotenv |
| Versioning | Git / GitHub |
| Déploiement | Hugging Face Spaces |

---

## 🚀 Lancement

### 1. Configurer les variables d'environnement

Créer un fichier `.env` à la racine :
```
DB_URL=postgresql://...  # URL de connexion Neon
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Exécuter le pipeline complet (recommandé)

```powershell
python run_pipeline.py
```

### Ou étape par étape

```powershell
python src/transformations/build_gold.py
python src/transformations/features_advanced.py
python src/ml/train_model.py
python src/ml/predict.py
```

### 4. Lancer le dashboard

```powershell
streamlit run src/dashboard/app.py
```

### 5. Lancer l'API

```powershell
python -m uvicorn src.api.main:app --reload
```

Docs disponibles sur : http://localhost:8000/docs

---

## 🤖 Modèles ML

Trois modèles entraînés par classe ATC (R06, R03, J01) sur 60 mois (jan 2021 – déc 2025).

### Classificateur — Détection mois à risque de rupture

| Classe | Algorithme | ROC-AUC | Méthode |
|---|---|---|---|
| R06 | Random Forest | 0.593 | SMOTE + GridSearchCV |
| R03 | Random Forest | — | SMOTE + GridSearchCV |
| J01 | Random Forest | — | SMOTE + GridSearchCV |
| Baseline | Logistic Regression | 0.679 | — |

### Régresseur — Prédiction concentration graminées mois suivant

| Modèle | R² | RMSE |
|---|---|---|
| Random Forest | 0.510 | 13.021 grains/m³ |

### Features classifier (19)

`gram_moy`, `gram_max`, `gram_roll7`, `gram_roll30`, `nb_jours_pic`,
`bouleau_moy`, `ambroisie_moy`, `nb_jours_pic_bouleau`,
`temp_moy`, `temp_max`, `temp_roll30`, `precip`, `wind`,
`mois`, `saison_allergies`, `source_encoded`,
`ruptures_lag1`, `gram_lag_mois`, `cumul_thermique`

### Features regressor (8)

`gram_moy`, `gram_max`, `gram_roll7`, `nb_jours_pic`,
`temp_moy`, `precip`, `mois`, `saison_allergies`

> ⚠️ `nb_ruptures` et `nb_risques` exclus intentionnellement (data leakage direct sur la target).

> ℹ️ Les scores ML sont limités par la taille du dataset (60 mois). SMOTE a permis d'équilibrer les classes mais a légèrement dégradé le ROC-AUC par rapport à la baseline sans rééchantillonnage.

---

## 🗄️ Base de données

### OLTP (6 tables)

`medicaments` · `ruptures` · `bdpm` · `openmedic` · `pollen` · `meteo`

### OLAP (étoile)

`fact_ruptures` · `dim_medicament` · `dim_date` · `dim_region` · `dim_pollen`

### Tables Gold

`medicaments_gold` · `gold_predictions` · `gold_predictions_R06` · `gold_predictions_R03` · `gold_predictions_J01`

---

## 📦 Sources de données

| Dataset | Source | Période | Granularité |
|---|---|---|---|
| Ruptures de stock | ANSM CADA | 2021–2024 | National |
| Ruptures de stock | ANSM 2025-2026 | 2025–2026 | National |
| Médicaments | ANSM CADA | — | Médicament |
| BDPM | ANSM | — | Médicament |
| OpenMedic | CNAM | 2021–2025 | Région × molécule × année |
| Météo | Open-Meteo | 2023–2026 | 13 régions × mois |
| Pollen | RNSA / ATMO / CAMS | 2021–2026 | National / régional |
| Épidémiologie | Sentinelles INSERM | 2021–2026 | National × mois |

> ℹ️ Le réseau RNSA a été liquidé en mars 2025 — pas de données pollen stations 2025 depuis cette source.

---

## 🔑 Variables d'environnement

| Variable | Description |
|---|---|
| `DB_URL` | URL de connexion PostgreSQL Neon |

Sur Hugging Face Spaces : Settings → Repository secrets → `DB_URL`

---