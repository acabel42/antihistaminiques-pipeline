import pandas as pd
import numpy as np
import os

os.makedirs('data/gold', exist_ok=True)

def build_features_pollen():
    print("Feature engineering pollen...")

    pollen = pd.read_csv('data/silver/J0_silver_pollen_2021_2026.csv')
    meteo  = pd.read_csv('data/silver/J0_silver_meteo_openmeteo_2021_2026.csv')

    pollen['date'] = pd.to_datetime(pollen['date'])
    meteo['time']  = pd.to_datetime(meteo['time'], format='mixed')
    meteo['time']  = meteo['time'].dt.normalize()

    # Agréger pollen par date
    pollen_daily = pollen.groupby('date').agg(
        graminees=('graminees_conc','mean'),
        bouleau=('bouleau_conc','mean'),
        aulne=('aulne_conc','mean'),
        ambroisie=('ambroisie_conc','mean'),
        armoise=('armoise_conc','mean'),
        olivier=('olivier_conc','mean')
    ).reset_index()

    # Lags
    for lag in [3, 7, 14]:
        pollen_daily[f'gram_lag{lag}'] = pollen_daily['graminees'].shift(lag)

    # Rolling means
    for window in [7, 14, 30]:
        pollen_daily[f'gram_roll{window}'] = pollen_daily['graminees'].rolling(window, min_periods=1).mean()

    # Flag pic pollen
    pollen_daily['flag_pic_pollen'] = (pollen_daily['graminees'] > 20).astype(int)

    # Variables temporelles
    pollen_daily['mois']             = pollen_daily['date'].dt.month
    pollen_daily['annee']            = pollen_daily['date'].dt.year
    pollen_daily['jour_annee']       = pollen_daily['date'].dt.dayofyear
    pollen_daily['saison_allergies'] = pollen_daily['mois'].apply(lambda m: 1 if m in [4,5,6,7] else 0)

    # Météo nationale Open-Meteo 2021-2026
    meteo_daily = meteo.groupby('time').agg(
        temp_moy=('temperature_2m_mean','mean'),
        temp_max=('temperature_2m_max','mean'),
        precip=('precipitation_sum','mean'),
        wind=('wind_speed_10m_max','mean')
    ).reset_index().rename(columns={'time':'date'})

    # Fusion
    df = pollen_daily.merge(meteo_daily, on='date', how='left')
    df = df.dropna(subset=['gram_lag7'])

    df.to_csv('data/gold/pollen_meteo_features.csv', index=False)
    print(f"  pollen_meteo_features : {df.shape}")
    print(f"  Periode : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Nulls : {df.isnull().sum().sum()}")
    return df

if __name__ == '__main__':
    os.chdir('/Users/nellyta/Jedha')
    build_features_pollen()