import pandas as pd
from sqlalchemy import create_engine
import os

ENGINE = create_engine('postgresql://pipeline:pipeline2026@localhost:5432/antihistaminiques')
os.makedirs('data/gold', exist_ok=True)

def build_gold():
    print("Construction table Gold...")

    pollen = pd.read_csv('data/gold/pollen_meteo_features.csv')
    rup    = pd.read_csv('data/silver/J0_silver_ruptures.csv')
    om     = pd.read_csv('data/silver/J0_silver_openmedic_2021_2025.csv', low_memory=False)

    pollen['date'] = pd.to_datetime(pollen['date'])
    rup['date_evenement'] = pd.to_datetime(rup['date_evenement'], errors='coerce')

    rup_r06 = rup[rup['code_atc'].str.startswith('R06', na=False)].copy()
    rup_r06['annee_mois_str'] = rup_r06['date_evenement'].dt.to_period('M').astype(str)
    rup_agg = rup_r06.groupby('annee_mois_str').agg(
        nb_ruptures=('classification', lambda x: (x=='rupture').sum()),
        nb_risques= ('classification', lambda x: (x=='risque').sum()),
    ).reset_index()

    # Variable cible — rupture OU tension R06
    rup_agg['target_rupture'] = ((rup_agg['nb_ruptures'] + rup_agg['nb_risques']) > 0).astype(int)

    pollen['annee_mois_str'] = pollen['date'].dt.to_period('M').astype(str)
    pollen_mois = pollen.groupby('annee_mois_str').agg(
        gram_moy=('graminees','mean'),
        gram_max=('graminees','max'),
        gram_roll7=('gram_roll7','mean'),
        nb_jours_pic=('flag_pic_pollen','sum'),
        temp_moy=('temp_moy','mean'),
        precip=('precip','mean'),
        mois=('mois','first'),
        annee=('annee','first'),
        saison_allergies=('saison_allergies','first')
    ).reset_index()

    om['BOITES'] = pd.to_numeric(om['BOITES'], errors='coerce')
    om_agg = om.groupby('annee')['BOITES'].sum().reset_index()
    om_agg.columns = ['annee','boites_total']

    gold = pollen_mois.merge(
        rup_agg[['annee_mois_str','nb_ruptures','nb_risques','target_rupture']],
        on='annee_mois_str', how='left')
    gold = gold.merge(om_agg, on='annee', how='left')
    gold['target_rupture'] = gold['target_rupture'].fillna(0).astype(int)
    gold['nb_ruptures']    = gold['nb_ruptures'].fillna(0).astype(int)
    gold['nb_risques']     = gold['nb_risques'].fillna(0).astype(int)

    gold.to_csv('data/gold/gold_ml.csv', index=False)
    gold.to_sql('gold_ml', ENGINE, if_exists='replace', index=False)

    print(f"  gold_ml : {gold.shape}")
    print(f"  Mois avec rupture/tension R06 : {gold['target_rupture'].sum()}")
    return gold

if __name__ == '__main__':
    os.chdir('/Users/nellyta/Jedha')
    build_gold()
