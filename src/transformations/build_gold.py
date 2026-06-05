import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('data/gold', exist_ok=True)

load_dotenv()
DB_URL = os.getenv('DB_URL', 'postgresql://pipeline:pipeline2026@localhost:5432/antihistaminiques')
ENGINE = create_engine(DB_URL)

def build_gold(classe_atc='R06'):
    print(f"Construction table Gold — classe ATC : {classe_atc}...")

    pollen = pd.read_csv('data/gold/pollen_meteo_features.csv')
    rup    = pd.read_csv('data/silver/J0_silver_ruptures.csv')
    om     = pd.read_csv('data/silver/J0_silver_openmedic_2021_2025.csv', low_memory=False)

    pollen['date'] = pd.to_datetime(pollen['date'])
    rup['date_evenement'] = pd.to_datetime(rup['date_evenement'], errors='coerce')

    # Ruptures filtrées par classe ATC
    rup_filtre = rup[rup['code_atc'].str.startswith(classe_atc, na=False)].copy()
    rup_filtre['annee_mois_str'] = rup_filtre['date_evenement'].dt.to_period('M').astype(str)
    rup_agg = rup_filtre.groupby('annee_mois_str').agg(
        nb_ruptures=('classification', lambda x: (x=='rupture').sum()),
        nb_risques= ('classification', lambda x: (x=='risque').sum()),
    ).reset_index()
    rup_agg['target_rupture'] = ((rup_agg['nb_ruptures'] + rup_agg['nb_risques']) > 0).astype(int)

    # Pollen par mois
    pollen['annee_mois_str'] = pollen['date'].dt.to_period('M').astype(str)
    pollen_mois = pollen.groupby('annee_mois_str').agg(
        gram_moy=('graminees','mean'),
        gram_max=('graminees','max'),
        gram_roll7=('gram_roll7','mean'),
        gram_roll30=('gram_roll30','mean'),
        nb_jours_pic=('flag_pic_pollen','sum'),
        bouleau_moy=('bouleau','mean'),
        ambroisie_moy=('ambroisie','mean'),
        aulne_moy=('aulne','mean'),
        armoise_moy=('armoise','mean'),
        olivier_moy=('olivier','mean'),
        nb_jours_pic_bouleau=('flag_pic_bouleau','sum'),
        nb_jours_pic_ambroisie=('flag_pic_ambroisie','sum'),
        nb_jours_pic_armoise=('flag_pic_armoise','sum'),
        nb_jours_pic_olivier=('flag_pic_olivier','sum'),
        nb_jours_pic_aulne=('flag_pic_aulne','sum'),
        temp_moy=('temp_moy','mean'),
        temp_max=('temp_max','mean'),
        temp_min=('temp_min','mean'),
        temp_roll30=('temp_roll30','mean'),
        precip=('precip','mean'),
        precip_lag7=('precip_lag7','mean'),
        wind=('wind','mean'),
        mois=('mois','first'),
        annee=('annee','first'),
        saison_allergies=('saison_allergies','first'),
        source_encoded=('source_encoded','mean'),
    ).reset_index()

    # Open Medic filtré par classe ATC
    om['BOITES'] = pd.to_numeric(om['BOITES'], errors='coerce')
    om_filtre = om[om['ATC4'].astype(str).str.startswith(classe_atc, na=False)]
    om_agg = om_filtre.groupby('annee')['BOITES'].sum().reset_index()
    om_agg.columns = ['annee','boites_total']

    # Jointure finale
    gold = pollen_mois.merge(
        rup_agg[['annee_mois_str','nb_ruptures','nb_risques','target_rupture']],
        on='annee_mois_str', how='left')
    gold = gold.merge(om_agg, on='annee', how='left')
    gold['target_rupture'] = gold['target_rupture'].fillna(0).astype(int)
    gold['nb_ruptures']    = gold['nb_ruptures'].fillna(0).astype(int)
    gold['nb_risques']     = gold['nb_risques'].fillna(0).astype(int)
    gold['classe_atc']     = classe_atc

    # Sauvegarde fichier specifique + fichier courant
    gold.to_csv(f'data/gold/gold_ml_{classe_atc}.csv', index=False)
    gold.to_csv('data/gold/gold_ml.csv', index=False)
    gold.to_sql(f'gold_ml_{classe_atc}', ENGINE, if_exists='replace', index=False)
    gold.to_sql('gold_ml', ENGINE, if_exists='replace', index=False)

    print(f"  gold_ml_{classe_atc} : {gold.shape}")
    print(f"  Mois avec rupture/tension {classe_atc} : {gold['target_rupture'].sum()}")
    return gold

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--classe', default='R06', help='Classe ATC (R06, R03, J01)')
    args = parser.parse_args()
    build_gold(classe_atc=args.classe)