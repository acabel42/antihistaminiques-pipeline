import pandas as pd
import os
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('data/silver', exist_ok=True)

DATE = datetime.now().strftime('%Y%m%d')

rnsa = pd.read_csv('data/silver/J0_silver_rnsa_pollen.csv')

rnsa_2122 = rnsa[rnsa['annee'].isin([2021, 2022])].copy()

mapping = {
    'GRAMINEE': 'graminees',
    'BETULA':   'bouleau',
    'ALNUS':    'aulne',
    'AMBROSIA': 'ambroisie',
    'ARTEMISI': 'armoise',
    'OLEA':     'olivier',
    'CUPRESSA': 'autres',
    'FRAXINUS': 'autres',
    'PLATANUS': 'autres',
    'CORYLUS':  'autres',
    'ULMUS':    'autres',
}

rnsa_2122['taxon_harmonise'] = rnsa_2122['taxon'].map(mapping)

rnsa_agg = rnsa_2122.groupby(['date', 'taxon_harmonise']).agg(
    concentration=('concentration_totale', 'mean')
).reset_index()

rnsa_pivot = rnsa_agg.pivot_table(
    index='date',
    columns='taxon_harmonise',
    values='concentration',
    aggfunc='mean'
).reset_index()

rnsa_pivot.columns.name = None
rnsa_pivot.columns = ['date'] + [f'{c}_conc' for c in rnsa_pivot.columns[1:]]
rnsa_pivot['date']      = pd.to_datetime(rnsa_pivot['date'])
rnsa_pivot['source']    = 'RNSA'
rnsa_pivot['annee']     = rnsa_pivot['date'].dt.year
rnsa_pivot['loaded_at'] = DATE

print(f"Shape : {rnsa_pivot.shape}")
print(f"Colonnes : {rnsa_pivot.columns.tolist()}")
print(f"Periode : {rnsa_pivot['date'].min().date()} -> {rnsa_pivot['date'].max().date()}")

# Sauvegarde horodatee + fichier courant
rnsa_pivot.to_csv(f'data/silver/J0_silver_rnsa_2021_2022_{DATE}.csv', index=False)
rnsa_pivot.to_csv('data/silver/J0_silver_rnsa_2021_2022.csv', index=False)
print(f"Sauvegarde : J0_silver_rnsa_2021_2022.csv — Date : {DATE}")