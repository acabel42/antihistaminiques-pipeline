from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# On charge le modèle une seule fois au démarrage de l'API
# Pas besoin de le recharger à chaque requête
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(ROOT, 'models', 'R06', 'rf_classifier.joblib')
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Antihistaminiques Pipeline API",
    description="Prédit le risque de rupture de stock d'antihistaminiques R06A",
    version="1.0"
)

# Les features que le modèle attend — dans le même ordre que FEATURES_CLF
FEATURES = [
    'gram_moy', 'gram_max', 'gram_roll7', 'gram_roll30', 'nb_jours_pic',
    'bouleau_moy', 'ambroisie_moy', 'nb_jours_pic_bouleau',
    'temp_moy', 'temp_max', 'temp_roll30',
    'precip', 'wind',
    'mois', 'saison_allergies', 'source_encoded',
    'ruptures_lag1', 'gram_lag_mois', 'cumul_thermique'
]

# Pydantic définit la structure attendue dans le body de la requête
# Si un champ manque ou a le mauvais type, FastAPI renvoie une erreur claire
class PredictionInput(BaseModel):
    gram_moy: float
    gram_max: float
    gram_roll7: float
    gram_roll30: float
    nb_jours_pic: int
    bouleau_moy: float
    ambroisie_moy: float
    nb_jours_pic_bouleau: int
    temp_moy: float
    temp_max: float
    temp_roll30: float
    precip: float
    wind: float
    mois: int
    saison_allergies: int
    source_encoded: float
    ruptures_lag1: float
    gram_lag_mois: float
    cumul_thermique: float


@app.get("/")
def home():
    # Route de base pour vérifier que l'API tourne
    return {"message": "API Antihistaminiques opérationnelle", "version": "1.0"}


@app.get("/health")
def health():
    # Route de santé — utile pour les déploiements (HF Spaces, Docker)
    return {"status": "ok", "model": "rf_classifier.joblib"}


@app.post("/predict")
def predict(data: PredictionInput):
    # On convertit l'input en DataFrame avec les colonnes dans le bon ordre
    df = pd.DataFrame([data.dict()])[FEATURES]

    # Prédiction binaire (0 ou 1) + probabilité
    prediction = int(model.predict(df)[0])
    probabilite = float(model.predict_proba(df)[0][1])

    # On renvoie un résultat lisible
    return {
        "rupture_predite": prediction,
        "probabilite_rupture": round(probabilite, 3),
        "interpretation": "Risque de rupture détecté" if prediction == 1 else "Pas de risque détecté"
    }