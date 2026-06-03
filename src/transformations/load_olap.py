import pandas as pd
from sqlalchemy import create_engine, text
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)

ENGINE = create_engine('postgresql://pipeline:pipeline2026@localhost:5432/antihistaminiques')

def load_olap():
    logging.info("Chargement schema OLAP...")
    df = pd.read_csv('data/gold/gold_ml.csv')
    logging.info(f"Gold charge : {df.shape}")

    # dim_temps
    dim_temps = df[['annee_mois_str','annee','mois']].copy()
    dim_temps.columns = ['annee_mois','annee','mois']
    dim_temps['trimestre'] = ((dim_temps['mois']-1)//3)+1
    dim_temps['saison'] = dim_temps['mois'].apply(
        lambda m: 'Printemps' if m in [3,4,5] else
                  'Ete'        if m in [6,7,8] else
                  'Automne'    if m in [9,10,11] else 'Hiver')
    dim_temps.to_sql('dim_temps', ENGINE, if_exists='replace', index=False)
    logging.info(f"  dim_temps : {dim_temps.shape}")

    # dim_pollen
    dim_pollen = df[['annee_mois_str','gram_moy','gram_max',
                     'gram_roll7','nb_jours_pic',
                     'bouleau_moy','ambroisie_moy','source_encoded']].copy()
    dim_pollen.columns = ['annee_mois','gram_moy','gram_max',
                          'gram_roll7','nb_jours_pic',
                          'bouleau_moy','ambroisie_moy','source']
    dim_pollen.to_sql('dim_pollen', ENGINE, if_exists='replace', index=False)
    logging.info(f"  dim_pollen : {dim_pollen.shape}")

    # dim_meteo
    dim_meteo = df[['annee_mois_str','temp_moy','temp_max',
                    'temp_min','precip','wind']].copy()
    dim_meteo.columns = ['annee_mois','temp_moy','temp_max',
                         'temp_min','precip','wind']
    dim_meteo.to_sql('dim_meteo', ENGINE, if_exists='replace', index=False)
    logging.info(f"  dim_meteo : {dim_meteo.shape}")

    # fact_ruptures_mensuelles
    fact = df[['annee_mois_str','mois','annee',
               'nb_ruptures','nb_risques',
               'target_rupture','boites_total',
               'gram_moy','temp_moy']].copy()
    fact.columns = ['annee_mois','mois','annee',
                    'nb_ruptures','nb_risques',
                    'target_rupture','boites_total',
                    'gram_moy','temp_moy']
    fact.to_sql('fact_ruptures_mensuelles', ENGINE, if_exists='replace', index=False)
    logging.info(f"  fact_ruptures_mensuelles : {fact.shape}")

    logging.info("OLAP charge avec succes !")
    return df

if __name__ == '__main__':
    load_olap()