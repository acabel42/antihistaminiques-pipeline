import pandas as pd
import os
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('data/silver', exist_ok=True)

DATE = datetime.now().strftime('%Y%m%d')

def clean_montant(s):
    return pd.to_numeric(
        s.astype(str)
         .str.replace('.', '', regex=False)
         .str.replace(',', '.', regex=False),
        errors='coerce')

fichiers = {
    2021: 'data/raw/OPEN_MEDIC_2021.CSV',
    2022: 'data/raw/OPEN_MEDIC_2022.CSV',
    2023: 'data/raw/OPEN_MEDIC_2023.CSV',
    2024: 'data/raw/OPEN_MEDIC_2024.CSV',
    2025: 'data/raw/OPEN_MEDIC_2025.CSV',
}

all_chunks = []
for annee, path in fichiers.items():
    print(f'Traitement {annee}...')
    chunks = []
    for chunk in pd.read_csv(path, sep=';', encoding='latin-1', chunksize=100000):
        r = chunk[chunk['ATC4'].astype(str).str.startswith(('R06', 'R03', 'J01'), na=False)]
        if len(r) > 0:
            chunks.append(r)
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        df['annee']     = annee
        df['REM_clean'] = clean_montant(df['REM'])
        df['BSE_clean'] = clean_montant(df['BSE'])
        df['loaded_at'] = DATE
        all_chunks.append(df)
        print(f'  {annee} : {len(df):,} lignes R06')

combined = pd.concat(all_chunks, ignore_index=True)

# Sauvegarde horodatee + fichier courant
combined.to_csv(f'data/silver/J0_silver_openmedic_2021_2025_{DATE}.csv', index=False)
combined.to_csv('data/silver/J0_silver_openmedic_2021_2025.csv', index=False)
print(f'Shape finale : {combined.shape} — Date : {DATE}')