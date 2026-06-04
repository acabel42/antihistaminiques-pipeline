import pandas as pd
from sqlalchemy import create_engine
import os

from dotenv import load_dotenv
load_dotenv()
ENGINE = create_engine(os.getenv('DB_URL'))

def clean_pollen():
    print("Nettoyage pollen...")
    df = pd.read_csv('data/silver/J0_silver_cams_pollen_2023_2026.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.drop_duplicates()
    cols = ['graminees_conc','bouleau_conc','aulne_conc',
            'ambroisie_conc','armoise_conc','olivier_conc']
    for col in cols:
        df[col] = df[col].clip(lower=0)
    df['longitude'] = df['longitude'].apply(lambda x: x-360 if x > 180 else x)
    print(f"  Shape : {df.shape} | Nulls : {df.isnull().sum().sum()}")
    # Table pollen trop volumineuse pour Neon Free (512MB) — CSV uniquement
    # df.to_sql('pollen', ENGINE, if_exists='replace', index=False, chunksize=10000)
    print(f"  Pollen conserve en CSV (13.7M lignes > limite Neon 512MB)")
    return df

def clean_meteo():
    print("Nettoyage meteo...")
    df = pd.read_csv('data/silver/J0_silver_meteo_2023_2026.csv')
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df = df.drop_duplicates()
    df = df.rename(columns={'time':'date'})
    print(f"  Shape : {df.shape} | Nulls : {df.isnull().sum().sum()}")
    df.to_sql('meteo', ENGINE, if_exists='replace', index=False)
    print(f"  Charge dans PostgreSQL : table meteo")
    return df

if __name__ == '__main__':
    os.chdir('/Users/nellyta/Jedha')
    clean_pollen()
    clean_meteo()
    print("Done !")