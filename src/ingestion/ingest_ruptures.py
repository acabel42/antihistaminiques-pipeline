import pandas as pd
import os
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('data/silver', exist_ok=True)

DATE = datetime.now().strftime('%Y%m%d')

def categorise_cause(cause):
    c = str(cause).lower()
    if 'augmentation' in c or 'volume de vente' in c:
        return 'Augmentation demande'
    elif 'capacite de production' in c:
        return 'Capacite production'
    elif 'matiere premiere' in c or 'article de conditionnement' in c:
        return 'Approvisionnement matiere'
    elif 'transport' in c or 'logistique' in c:
        return 'Logistique'
    else:
        return 'Autre'

ruptures = pd.read_excel(
    'data/raw/rupture-cada26v21-data-ansm-historique-ruptures-2026.xlsx',
    sheet_name='historiqueRuptures')
produits = pd.read_excel(
    'data/raw/260302-cada26v21-data-ansm-produits-2026.xlsx',
    sheet_name='produits')
patients = pd.read_excel(
    'data/raw/260302-cada26v21-data-ansm-produits-2026.xlsx',
    sheet_name='nbPatients')

for df in [ruptures, produits, patients]:
    df['cis'] = df['cis'].astype(str).str.strip()

pat = patients[['cis','Estimation du nombre de patients traités en ville']]
pat = pat.rename(columns={'Estimation du nombre de patients traités en ville':'nb_patients_ville'})

df = ruptures.merge(produits[['cis','atc','atc_name','laboratoire']], on='cis', how='left')
df = df.merge(pat, on='cis', how='left')
df['est_antihistaminique'] = df['atc'].str.startswith('R06A', na=False)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['annee'] = df['date'].dt.year
df['mois']  = df['date'].dt.month
df['trimestre'] = df['date'].dt.quarter
df['saison_allergies'] = df['mois'].apply(lambda m: 1 if m in [3,4,5,6] else 0)
df['cause_categorie'] = df['cause'].apply(categorise_cause)
df['loaded_at'] = DATE

# Sauvegarde horodatee + fichier courant
df.to_csv(f'data/silver/J0_silver_ruptures_{DATE}.csv', index=False)
df.to_csv('data/silver/J0_silver_ruptures.csv', index=False)
print(f'Shape : {df.shape} — R06A : {df.est_antihistaminique.sum()} — Date : {DATE}')