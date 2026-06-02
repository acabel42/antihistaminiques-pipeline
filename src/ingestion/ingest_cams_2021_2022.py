import pandas as pd

# Chargement
rnsa = pd.read_csv('/Users/nellyta/Jedha/data/silver/J0_silver_rnsa_pollen.csv')

# Filtrer 2021-2022
rnsa_2122 = rnsa[rnsa['annee'].isin([2021, 2022])].copy()

# Mapping taxons RNSA -> CAMS
mapping = {
    'GRAMINEE': 'graminees',
    'BETULA':   'bouleau',
    'ALNUS':    'aulne',
    'AMBROSIA': 'ambroisie',
    'ARTEMISI': 'armoise',
    'OLEA':     'olivier',
    # Tous les autres -> autres
    'CUPRESSA': 'autres',
    'FRAXINUS': 'autres',
    'PLATANUS': 'autres',
    'CORYLUS':  'autres',
    'ULMUS':    'autres',
}

rnsa_2122['taxon_harmonise'] = rnsa_2122['taxon'].map(mapping)

# Agréger par date + taxon harmonisé
rnsa_agg = rnsa_2122.groupby(['date', 'taxon_harmonise']).agg(
    concentration=('concentration_totale', 'mean')
).reset_index()

# Pivoter pour avoir une colonne par taxon
rnsa_pivot = rnsa_agg.pivot_table(
    index='date',
    columns='taxon_harmonise',
    values='concentration',
    aggfunc='mean'
).reset_index()
rnsa_pivot.columns.name = None
rnsa_pivot.columns = ['date'] + [f'{c}_conc' for c in rnsa_pivot.columns[1:]]
rnsa_pivot['date']   = pd.to_datetime(rnsa_pivot['date'])
rnsa_pivot['source'] = 'RNSA'
rnsa_pivot['annee']  = rnsa_pivot['date'].dt.year

print(f"Shape : {rnsa_pivot.shape}")
print(f"Colonnes : {rnsa_pivot.columns.tolist()}")
print(f"Période : {rnsa_pivot['date'].min().date()} -> {rnsa_pivot['date'].max().date()}")
print(rnsa_pivot.head(3).to_string())

# Sauvegarder
rnsa_pivot.to_csv('/Users/nellyta/Jedha/data/silver/J0_silver_rnsa_2021_2022.csv', index=False)
print("\nSauvegardé : J0_silver_rnsa_2021_2022.csv")