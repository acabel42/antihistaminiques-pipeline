import pandas as pd
import os
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('data/silver', exist_ok=True)

DATE = datetime.now().strftime('%Y%m%d')

mitm = pd.read_csv(
    'data/raw/CIS_MITM.txt',
    sep='\t', header=None, encoding='latin-1',
    names=['cis','atc','denomination','lien'])
compo = pd.read_csv(
    'data/raw/CIS_COMPO_bdpm.txt',
    sep='\t', header=None, encoding='latin-1',
    names=['cis','forme','code_substance','substance',
           'dosage','ref_dosage','nature','num_liaison'])

for df in [mitm, compo]:
    df['cis'] = df['cis'].astype(str).str.strip()

compo_sa = compo[compo['nature']=='SA'][['cis','substance','dosage']]
compo_sa = compo_sa.drop_duplicates(subset='cis', keep='first')

df = mitm.merge(compo_sa, on='cis', how='left')
df['est_antihistaminique'] = df['atc'].str.startswith('R06A', na=False)
df['loaded_at'] = DATE

# Sauvegarde horodatee + fichier courant
df.to_csv(f'data/silver/J0_silver_bdpm_{DATE}.csv', index=False)
df.to_csv('data/silver/J0_silver_bdpm.csv', index=False)
print(f'Shape : {df.shape} — R06A : {df.est_antihistaminique.sum()} — Date : {DATE}')