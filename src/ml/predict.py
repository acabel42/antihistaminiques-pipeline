import pandas as pd
import joblib
import os
import logging
from sqlalchemy import create_engine
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()
ENGINE = create_engine(os.getenv('DB_URL', 'postgresql://pipeline:pipeline2026@localhost:5432/antihistaminiques'))

FEATURES_CLF = [
    'gram_moy', 'gram_max', 'gram_roll7', 'gram_roll30', 'nb_jours_pic',
    'bouleau_moy', 'ambroisie_moy', 'nb_jours_pic_bouleau',
    'temp_moy', 'temp_max', 'temp_roll30',
    'precip', 'wind',
    'mois', 'saison_allergies', 'source_encoded',
    'ruptures_lag1', 'gram_lag_mois', 'cumul_thermique'
]

FEATURES_REG = [
    'gram_moy', 'gram_max', 'gram_roll7', 'nb_jours_pic',
    'temp_moy', 'precip', 'mois', 'saison_allergies'
]

def predict():
    logging.info("Generation des predictions...")

    gold_path = 'data/gold/gold_ml_advanced.csv' if os.path.exists(
        'data/gold/gold_ml_advanced.csv') else 'data/gold/gold_ml.csv'
    df = pd.read_csv(gold_path)
    logging.info(f"  Gold charge : {df.shape} depuis {gold_path}")

    clf = joblib.load('models/rf_classifier.joblib')
    reg = joblib.load('models/rf_regressor.joblib')
    logging.info("  Modeles charges : rf_classifier + rf_regressor")

    fc = [f for f in FEATURES_CLF if f in df.columns]
    fr = [f for f in FEATURES_REG if f in df.columns]

    df['pred_rupture']   = clf.predict(df[fc].fillna(0))
    df['proba_rupture']  = clf.predict_proba(df[fc].fillna(0))[:, 1]
    df['pred_gram_next'] = reg.predict(df[fr].fillna(0))
    df['generated_at']   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    out = df[['annee_mois_str', 'pred_rupture', 'proba_rupture',
              'pred_gram_next', 'generated_at']]

    out.to_sql('gold_predictions', ENGINE, if_exists='replace', index=False)
    out.to_csv('data/gold/gold_predictions.csv', index=False)

    logging.info(f"  Predictions sauvegardees : {len(out)} mois")
    logging.info(f"  Mois a risque : {(out['pred_rupture']==1).sum()}")
    logging.info(f"  Table gold_predictions dans PostgreSQL")

    return out

if __name__ == '__main__':
    predict()