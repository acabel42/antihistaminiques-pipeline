import pandas as pd
import numpy as np
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('data/gold', exist_ok=True)

def build_features_pollen():
    print("Feature engineering pollen...")

    pollen = pd.read_csv('data/silver/J0_silver_pollen_2021_2026.csv')
    meteo  = pd.read_csv('data/silver/J0_silver_meteo_openmeteo_2021_2026.csv')

    pollen['date'] = pd.to_datetime(pollen['date'])
    meteo['time']  = pd.to_datetime(meteo['time'], format='mixed')
    meteo['time']  = meteo['time'].dt.normalize()

    # Agregation pollen par date en gardant la source
    pollen_daily = pollen.groupby(['date','source']).agg(
        graminees=('graminees_conc','mean'),
        bouleau=('bouleau_conc','mean'),
        aulne=('aulne_conc','mean'),
        ambroisie=('ambroisie_conc','mean'),
        armoise=('armoise_conc','mean'),
        olivier=('olivier_conc','mean')
    ).reset_index()

    # Source encoded — correction biais RNSA/CAMS
    pollen_daily['source_encoded'] = (pollen_daily['source'] == 'CAMS').astype(int)

    # Lags graminees
    for lag in [3, 7, 14]:
        pollen_daily[f'gram_lag{lag}'] = pollen_daily['graminees'].shift(lag)

    # Rolling means graminees
    for window in [7, 14, 30]:
        pollen_daily[f'gram_roll{window}'] = pollen_daily['graminees'].rolling(window, min_periods=1).mean()

    # Lags autres taxons
    pollen_daily['bouleau_lag7']   = pollen_daily['bouleau'].shift(7)
    pollen_daily['ambroisie_lag7'] = pollen_daily['ambroisie'].shift(7)
    pollen_daily['aulne_lag7']     = pollen_daily['aulne'].shift(7)
    pollen_daily['armoise_lag7']   = pollen_daily['armoise'].shift(7)
    pollen_daily['olivier_lag7']   = pollen_daily['olivier'].shift(7)

    # Flags pics pollen
    pollen_daily['flag_pic_pollen']    = (pollen_daily['graminees'] > 20).astype(int)
    pollen_daily['flag_pic_bouleau']   = (pollen_daily['bouleau'] > 15).astype(int)
    pollen_daily['flag_pic_ambroisie'] = (pollen_daily['ambroisie'] > 5).astype(int)
    pollen_daily['flag_pic_armoise']   = (pollen_daily['armoise'] > 5).astype(int)
    pollen_daily['flag_pic_olivier']   = (pollen_daily['olivier'] > 3).astype(int)
    pollen_daily['flag_pic_aulne']     = (pollen_daily['aulne'] > 10).astype(int)

    # Variables temporelles
    pollen_daily['mois']             = pollen_daily['date'].dt.month
    pollen_daily['annee']            = pollen_daily['date'].dt.year
    pollen_daily['jour_annee']       = pollen_daily['date'].dt.dayofyear
    pollen_daily['semaine']          = pollen_daily['date'].dt.isocalendar().week.astype(int)
    pollen_daily['saison_allergies'] = pollen_daily['mois'].apply(lambda m: 1 if m in [4,5,6,7] else 0)

    # Meteo nationale Open-Meteo 2021-2026
    meteo_daily = meteo.groupby('time').agg(
        temp_moy=('temperature_2m_mean','mean'),
        temp_max=('temperature_2m_max','mean'),
        temp_min=('temperature_2m_min','mean'),
        precip=('precipitation_sum','mean'),
        wind=('wind_speed_10m_max','mean')
    ).reset_index().rename(columns={'time':'date'})

    # Lags meteo
    meteo_daily['temp_lag7']   = meteo_daily['temp_moy'].shift(7)
    meteo_daily['temp_lag14']  = meteo_daily['temp_moy'].shift(14)
    meteo_daily['precip_lag7'] = meteo_daily['precip'].shift(7)

    # Rolling meteo
    meteo_daily['temp_roll7']  = meteo_daily['temp_moy'].rolling(7, min_periods=1).mean()
    meteo_daily['temp_roll30'] = meteo_daily['temp_moy'].rolling(30, min_periods=1).mean()

    # Fusion
    df = pollen_daily.merge(meteo_daily, on='date', how='left')
    df = df.dropna(subset=['gram_lag7'])

    df.to_csv('data/gold/pollen_meteo_features.csv', index=False)
    print(f"  pollen_meteo_features : {df.shape}")
    print(f"  Periode : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Nulls : {df.isnull().sum().sum()}")
    return df

if __name__ == '__main__':
    build_features_pollen()