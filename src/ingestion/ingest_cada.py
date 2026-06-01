import pandas as pd
import os

os.makedirs('data/silver', exist_ok=True)

FILE = 'data/raw/260302-cada26v21-data-ansm-produits-2026.xlsx'

produits   = pd.read_excel(FILE, sheet_name='produits')
patients   = pd.read_excel(FILE, sheet_name='nbPatients')
age        = pd.read_excel(FILE, sheet_name='nbPatients_age')
sexe       = pd.read_excel(FILE, sheet_name='nbPatients_sexe')
substances = pd.read_excel(FILE, sheet_name='produitsSubstances')

age = age.rename(columns={'Estimation de la répartition par âge des patients traités (%)': 'pct'})
age_pivot = age.pivot_table(index='cis', columns='âge', values='pct').reset_index()
age_pivot.columns.name = None
age_pivot.columns = ['cis'] + [f'pct_age_{c.replace(" ","_")}' for c in age_pivot.columns[1:]]

sexe = sexe.rename(columns={'Estimation de la répartition par sexe des patients traités (%)': 'pct'})
sexe_pivot = sexe.pivot_table(index='cis', columns='sex', values='pct').reset_index()
sexe_pivot.columns.name = None
sexe_pivot.columns = ['cis'] + [f'pct_sexe_{c}' for c in sexe_pivot.columns[1:]]

subst = substances[['cis','Substance active']].drop_duplicates(subset='cis', keep='first')
subst.columns = ['cis','substance_active']

pat = patients[['cis','Estimation du nombre de patients traités en ville']].copy()
pat.columns = ['cis','nb_patients_ville']

df = produits.rename(columns={'name':'nom_medicament','atc':'code_atc','atc_name':'molecule'})
df = df.merge(pat,        on='cis', how='left')
df = df.merge(subst,      on='cis', how='left')
df = df.merge(age_pivot,  on='cis', how='left')
df = df.merge(sexe_pivot, on='cis', how='left')
df['est_antihistaminique'] = df['code_atc'].str.startswith('R06A', na=False)

df.to_csv('data/silver/J0_silver_medicaments.csv', index=False)
print(f'Shape : {df.shape} — R06A : {df.est_antihistaminique.sum()}')