import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
ENGINE = create_engine(os.getenv('DB_URL'))

def clean_and_load():
    # ── Médicaments ──────────────────────────────────────────
    df_med = pd.read_csv('data/silver/J0_silver_medicaments.csv')
    df_med = df_med.dropna(how='all')
    df_med = df_med.drop_duplicates()
    df_med['nb_patients_ville'] = df_med['nb_patients_ville'].fillna(0)
    df_med.to_sql('medicaments', ENGINE, if_exists='replace', index=False)
    print(f'Médicaments chargés : {len(df_med)} lignes')

    # ── Ruptures ─────────────────────────────────────────────
    df_rup = pd.read_csv('data/silver/J0_silver_ruptures.csv')
    df_rup = df_rup.dropna(how='all')
    df_rup = df_rup.drop_duplicates()
    df_rup['date_evenement'] = pd.to_datetime(df_rup['date_evenement'], errors='coerce')
    df_rup.to_sql('ruptures', ENGINE, if_exists='replace', index=False)
    print(f'Ruptures chargées : {len(df_rup)} lignes')

if __name__ == '__main__':
    clean_and_load()
