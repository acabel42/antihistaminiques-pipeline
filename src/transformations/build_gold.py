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

SENTINELLES_FEATURES = {
    'R03': ['grippal_inc100_moy', 'grippal_inc100_max',
            'ira_inc100_moy',     'ira_inc100_max'],
    'J01': ['diarrhee_inc100_moy', 'diarrhee_inc100_max', 'varicelle_inc100_moy', 'varicelle_inc100_max'],
    'R06': [],
}

def load_ruptures(classe_atc):
    rup_cada = pd.read_csv('data/silver/J0_silver_ruptures.csv')
    rup_cada['date_evenement'] = pd.to_datetime(rup_cada['date_evenement'], errors='coerce')
    rup_cada_filtre = rup_cada[rup_cada['code_atc'].str.startswith(classe_atc, na=False)].copy()
    rup_cada_filtre['annee_mois_str'] = rup_cada_filtre['date_evenement'].dt.to_period('M').astype(str)
    rup_cada_filtre['is_rupture'] = (rup_cada_filtre['classification'] == 'rupture').astype(int)
    rup_cada_filtre['is_risque']  = (rup_cada_filtre['classification'] == 'risque').astype(int)

    ansm_path = 'data/silver/J0_silver_ruptures_ansm_2026.csv'
    rup_ansm_filtre = pd.DataFrame()
    if os.path.exists(ansm_path):
        rup_ansm = pd.read_csv(ansm_path)
        rup_ansm['date_debut'] = pd.to_datetime(rup_ansm['date_debut'], errors='coerce')
        rup_ansm_filtre = rup_ansm[rup_ansm['ATC4'].astype(str).str.startswith(classe_atc, na=False)].copy()
        rup_ansm_filtre['annee_mois_str'] = rup_ansm_filtre['date_debut'].dt.to_period('M').astype(str)
        rup_ansm_filtre['is_rupture'] = rup_ansm_filtre['statut'].str.contains('Rupture de stock', case=False, na=False).astype(int)
        rup_ansm_filtre['is_risque']  = rup_ansm_filtre['statut'].str.contains("Tension d'approvisionnement", case=False, na=False).astype(int)
        print(f"  Source ANSM 2026 chargee : {len(rup_ansm_filtre)} lignes {classe_atc}")
    else:
        print(f"  [INFO] {ansm_path} absent — source CADA uniquement")

    cols_communs = ['annee_mois_str', 'is_rupture', 'is_risque']
    rup_cada_slim = rup_cada_filtre[cols_communs].copy()
    rup_cada_slim['source'] = 'CADA'
    if not rup_ansm_filtre.empty:
        rup_ansm_slim = rup_ansm_filtre[cols_communs].copy()
        rup_ansm_slim['source'] = 'ANSM_2026'
        rup_all = pd.concat([rup_cada_slim, rup_ansm_slim], ignore_index=True)
    else:
        rup_all = rup_cada_slim

    rup_agg = rup_all.groupby('annee_mois_str').agg(
        nb_ruptures=('is_rupture', 'sum'),
        nb_risques=('is_risque', 'sum'),
    ).reset_index()
    rup_agg['target_rupture'] = ((rup_agg['nb_ruptures'] + rup_agg['nb_risques']) > 0).astype(int)
    print(f"  Ruptures fusionnees ({classe_atc}) : {len(rup_agg)} mois")
    print(f"  Mois positifs : {rup_agg['target_rupture'].sum()}")
    return rup_agg

def load_sentinelles(classe_atc):
    sent_path = 'data/silver/J0_silver_sentinelles.csv'
    features = SENTINELLES_FEATURES.get(classe_atc, [])
    if not features or not os.path.exists(sent_path):
        return None
    df_sent = pd.read_csv(sent_path)
    cols = ['annee_mois'] + [f for f in features if f in df_sent.columns]
    df_sent = df_sent[cols].rename(columns={'annee_mois': 'annee_mois_str'})
    print(f"  Sentinelles chargees ({classe_atc}) : {len(df_sent)} mois, {len(cols)-1} features")
    return df_sent

def build_gold(classe_atc='R06'):
    print(f"Construction table Gold — classe ATC : {classe_atc}...")
    pollen = pd.read_csv('data/gold/pollen_meteo_features.csv')
    om     = pd.read_csv('data/silver/J0_silver_openmedic_2021_2025.csv', low_memory=False)
    pollen['date'] = pd.to_datetime(pollen['date'])
    rup_agg = load_ruptures(classe_atc)

    pollen['annee_mois_str'] = pollen['date'].dt.to_period('M').astype(str)
    pollen_mois = pollen.groupby('annee_mois_str').agg(
        gram_moy=('graminees','mean'), gram_max=('graminees','max'),
        gram_roll7=('gram_roll7','mean'), gram_roll30=('gram_roll30','mean'),
        nb_jours_pic=('flag_pic_pollen','sum'),
        bouleau_moy=('bouleau','mean'), ambroisie_moy=('ambroisie','mean'),
        aulne_moy=('aulne','mean'), armoise_moy=('armoise','mean'), olivier_moy=('olivier','mean'),
        nb_jours_pic_bouleau=('flag_pic_bouleau','sum'),
        nb_jours_pic_ambroisie=('flag_pic_ambroisie','sum'),
        nb_jours_pic_armoise=('flag_pic_armoise','sum'),
        nb_jours_pic_olivier=('flag_pic_olivier','sum'),
        nb_jours_pic_aulne=('flag_pic_aulne','sum'),
        temp_moy=('temp_moy','mean'), temp_max=('temp_max','mean'),
        temp_min=('temp_min','mean'), temp_roll30=('temp_roll30','mean'),
        precip=('precip','mean'), precip_lag7=('precip_lag7','mean'), wind=('wind','mean'),
        mois=('mois','first'), annee=('annee','first'),
        saison_allergies=('saison_allergies','first'), source_encoded=('source_encoded','mean'),
    ).reset_index()

    om['BOITES'] = pd.to_numeric(om['BOITES'], errors='coerce')
    om_filtre = om[om['ATC4'].astype(str).str.startswith(classe_atc, na=False)]
    om_agg = om_filtre.groupby('annee')['BOITES'].sum().reset_index()
    om_agg.columns = ['annee', 'boites_total']

    gold = pollen_mois.merge(
        rup_agg[['annee_mois_str', 'nb_ruptures', 'nb_risques', 'target_rupture']],
        on='annee_mois_str', how='left')
    gold = gold.merge(om_agg, on='annee', how='left')

    df_sent = load_sentinelles(classe_atc)
    if df_sent is not None:
        gold = gold.merge(df_sent, on='annee_mois_str', how='left')
        print(f"  Features Sentinelles ajoutees : {list(df_sent.columns[1:])}")
    else:
        print(f"  Pas de features Sentinelles pour {classe_atc}")

    gold['target_rupture'] = gold['target_rupture'].fillna(0).astype(int)
    gold['nb_ruptures']    = gold['nb_ruptures'].fillna(0).astype(int)
    gold['nb_risques']     = gold['nb_risques'].fillna(0).astype(int)
    gold['classe_atc']     = classe_atc

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
    parser.add_argument('--classe', default='R06', help='Classe ATC (R06, R03, J01, ...)')
    args = parser.parse_args()
    build_gold(classe_atc=args.classe)
