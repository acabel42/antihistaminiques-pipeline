import pandas as pd
import os

os.makedirs('data/silver', exist_ok=True)

# Charger RNSA 2021-2022
rnsa = pd.read_csv('data/silver/J0_silver_rnsa_2021_2022.csv')
rnsa['date'] = pd.to_datetime(rnsa['date'])

# Charger CAMS 2023-2026
cams = pd.read_csv('data/silver/J0_silver_cams_pollen_2023_2026.csv', low_memory=False)
cams['date'] = pd.to_datetime(cams['date'])

# Agréger CAMS par date (moyenne nationale)
cams_daily = cams.groupby('date').agg(
    graminees_conc=('graminees_conc', 'mean'),
    bouleau_conc=('bouleau_conc',     'mean'),
    aulne_conc=('aulne_conc',         'mean'),
    ambroisie_conc=('ambroisie_conc', 'mean'),
    armoise_conc=('armoise_conc',     'mean'),
    olivier_conc=('olivier_conc',     'mean'),
).reset_index()
cams_daily['autres_conc'] = 0.0
cams_daily['source']      = 'CAMS'
cams_daily['annee']       = cams_daily['date'].dt.year

# Aligner les colonnes
cols = ['date','graminees_conc','bouleau_conc','aulne_conc',
        'ambroisie_conc','armoise_conc','olivier_conc',
        'autres_conc','source','annee']

rnsa  = rnsa[cols]
cams_daily = cams_daily[cols]

# Fusionner
combined = pd.concat([rnsa, cams_daily], ignore_index=True)
combined  = combined.sort_values('date').reset_index(drop=True)

print(f"Shape finale : {combined.shape}")
print(f"Période      : {combined['date'].min().date()} -> {combined['date'].max().date()}")
print(f"Sources      : {combined['source'].value_counts().to_dict()}")
print(combined.head(3).to_string())

combined.to_csv('data/silver/J0_silver_pollen_2021_2026.csv', index=False)
print("\nSauvegarde : J0_silver_pollen_2021_2026.csv")