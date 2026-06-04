import pandas as pd
from sqlalchemy import create_engine
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)

ENGINE = create_engine('postgresql://pipeline:pipeline2026@localhost:5432/antihistaminiques')

def load_olap():
    logging.info("Chargement schema OLAP...")
    df = pd.read_csv('data/gold/gold_ml.csv')
    med = pd.read_csv('data/silver/J0_silver_medicaments.csv')
    logging.info(f"Gold charge : {df.shape}")
    logging.info(f"Medicaments charge : {med.shape}")

    # dim_date
    dim_date = df[['annee_mois_str','annee','mois']].copy()
    dim_date.columns = ['annee_mois_str','annee','mois']
    dim_date['trimestre'] = ((dim_date['mois']-1)//3)+1
    dim_date['saison_allergies'] = dim_date['mois'].apply(lambda m: m in [4,5,6,7])
    dim_date = dim_date.reset_index().rename(columns={'index':'id'})
    dim_date['id'] = dim_date['id'] + 1
    dim_date.to_sql('dim_date', ENGINE, if_exists='replace', index=False)
    logging.info(f"  dim_date : {dim_date.shape}")

    # dim_medicament
    dim_medicament = med[['cis','nom_medicament','laboratoire',
                           'molecule','code_atc','est_antihistaminique']].copy()
    dim_medicament.to_sql('dim_medicament', ENGINE, if_exists='replace', index=False)
    logging.info(f"  dim_medicament : {dim_medicament.shape}")

    # dim_region
    regions = [
        (1, 'Ile-de-France'), (2, 'Centre-Val-de-Loire'),
        (3, 'Bourgogne-Franche-Comte'), (4, 'Normandie'),
        (5, 'Hauts-de-France'), (6, 'Grand Est'),
        (7, 'Pays-de-la-Loire'), (8, 'Bretagne'),
        (9, 'Nouvelle-Aquitaine'), (10, 'Occitanie'),
        (11, 'Auvergne-Rhone-Alpes'), (12, 'PACA'), (13, 'Corse')
    ]
    dim_region = pd.DataFrame(regions, columns=['aasqa','region'])
    dim_region.to_sql('dim_region', ENGINE, if_exists='replace', index=False)
    logging.info(f"  dim_region : {dim_region.shape}")

    # dim_pollen
    dim_pollen = df[['annee_mois_str','gram_moy','gram_max',
                     'gram_roll7','nb_jours_pic',
                     'bouleau_moy','ambroisie_moy','source_encoded']].copy()
    dim_pollen.columns = ['annee_mois','gram_moy','gram_max',
                          'gram_roll7','nb_jours_pic',
                          'bouleau_moy','ambroisie_moy','source']
    dim_pollen = dim_pollen.reset_index().rename(columns={'index':'id'})
    dim_pollen['id'] = dim_pollen['id'] + 1
    dim_pollen.to_sql('dim_pollen', ENGINE, if_exists='replace', index=False)
    logging.info(f"  dim_pollen : {dim_pollen.shape}")

    # fact_ruptures
    fact = df[['annee_mois_str','mois','annee',
               'nb_ruptures','nb_risques',
               'target_rupture','boites_total',
               'gram_moy','temp_moy']].copy()
    fact = fact.reset_index().rename(columns={'index':'id'})
    fact['id'] = fact['id'] + 1
    fact['date_id'] = fact['id']
    fact['pollen_id'] = fact['id']
    fact['aasqa'] = None
    fact['cis'] = None
    fact['nb_patients'] = None
    fact['pred_rupture'] = fact['target_rupture']
    fact['proba_rupture'] = None
    fact = fact[['id','cis','aasqa','date_id','pollen_id',
                 'nb_ruptures','nb_patients','boites_total',
                 'gram_moy','temp_moy','pred_rupture','proba_rupture']]
    fact.to_sql('fact_ruptures', ENGINE, if_exists='replace', index=False)
    logging.info(f"  fact_ruptures : {fact.shape}")

    logging.info("OLAP charge avec succes !")
    return df

if __name__ == '__main__':
    load_olap()