import pandas as pd
import numpy as np
import os

def build_features_advanced():
    df = pd.read_csv('data/gold/gold_ml.csv')
    df['annee_mois_dt'] = pd.to_datetime(df['annee_mois_str'])
    df = df.sort_values('annee_mois_dt').reset_index(drop=True)

    # Cumul thermique — somme temperatures depuis janvier
    df['cumul_thermique'] = df['temp_moy'] * 30

    # Lags graminees
    df['gram_lag_mois']  = df['gram_moy'].shift(1)
    df['gram_lag_2mois'] = df['gram_moy'].shift(2)

    # Rolling mean 3 mois
    df['gram_roll3m'] = df['gram_moy'].rolling(3, min_periods=1).mean()

    # Ratio pic saison
    df['ratio_pic_saison'] = df['nb_jours_pic'] / 30

    # Tendance ruptures
    df['ruptures_lag1'] = df['nb_ruptures'].shift(1).fillna(0)

    df = df.dropna(subset=['gram_lag_mois'])
    df.to_csv('data/gold/gold_ml.csv', index=False)
    print(f'Gold enrichi : {df.shape}')
    return df

if __name__ == '__main__':
    os.chdir('C:/Users/kongm/Desktop/PROJETS_FULLSTACK_PHARMA/antihistaminiques-pipeline')
    build_features_advanced()
