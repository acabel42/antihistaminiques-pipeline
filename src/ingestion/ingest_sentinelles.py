"""
ingest_sentinelles.py
─────────────────────
Ingestion des données épidémiques Réseau Sentinelles (sentiweb.fr)
Indicateurs : Syndromes grippaux (3), IRA (25), Diarrhée aiguë (6), Varicelle (7)
Fréquence    : Hebdomadaire → agrégé en mensuel
Sortie       : data/silver/J0_silver_sentinelles.csv

Usage :
    python src/ingestion/ingest_sentinelles.py
"""

import pandas as pd
import requests
import io
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('data/silver', exist_ok=True)

# ── Indicateurs à ingérer ─────────────────────────────────────────────────────
INDICATEURS = [
    {
        'id':       'inc-3-PAY-ds2',
        'nom':      'grippal',
        'label':    'Syndromes grippaux',
        'classe':   'R03',
        'debut':    202100,   # on filtre 2021+
    },
    {
        'id':       'inc-25-PAY',
        'nom':      'ira',
        'label':    'Infections Respiratoires Aiguës',
        'classe':   'R03',
        'debut':    202100,
    },
    {
        'id':       'inc-6-PAY-ds2',
        'nom':      'diarrhee',
        'label':    'Diarrhée aiguë',
        'classe':   'J01',
        'debut':    202100,
    },
    {
        'id':       'inc-7-PAY-ds2',
        'nom':      'varicelle',
        'label':    'Varicelle',
        'classe':   'J01',
        'debut':    202100,
    },
]

BASE_URL = "https://www.sentiweb.fr/api/v1/datasets/rest/dataset"


# ── Helpers ───────────────────────────────────────────────────────────────────

def week_to_date(week_int):
    """
    Convertit une semaine au format YYYYSS (ex: 202122) en date du lundi.
    Utilise le calendrier ISO (semaine 1 = première semaine avec un jeudi).
    """
    s = str(int(week_int))
    year = int(s[:4])
    week = int(s[4:])
    # Format ISO : %G = année ISO, %V = semaine ISO, %u = jour (1=lundi)
    return pd.to_datetime(f"{year}-W{week:02d}-1", format="%G-W%V-%u",
                          errors='coerce')


def fetch_indicator(indic):
    """Télécharge un indicateur Sentinelles et retourne un DataFrame mensuel."""
    log.info(f"  Téléchargement {indic['label']} ({indic['id']})...")

    url = f"{BASE_URL}?id={indic['id']}&span=all"
    headers = {"Accept": "text/csv"}

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # La première ligne est un commentaire JSON — on la skip
    lines = resp.text.splitlines()
    csv_lines = [l for l in lines if not l.startswith('#') and not l.startswith('{')]
    csv_text = "\n".join(csv_lines)

    df = pd.read_csv(io.StringIO(csv_text))
    log.info(f"    → {len(df)} semaines brutes")

    # Nettoyage colonnes
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Filtrer 2021+
    df = df[df['week'] >= indic['debut']].copy()

    # Convertir semaine → date
    df['date'] = df['week'].apply(week_to_date)
    df = df.dropna(subset=['date'])

    # Agréger en mensuel (moyenne du taux pour 100k)
    df['annee_mois'] = df['date'].dt.to_period('M').astype(str)
    df_mois = df.groupby('annee_mois').agg(
        inc100_moy=(  'inc100', 'mean'),
        inc100_max=(  'inc100', 'max'),
        nb_semaines=( 'inc100', 'count'),
    ).reset_index()

    # Renommer avec le nom de l'indicateur
    nom = indic['nom']
    df_mois = df_mois.rename(columns={
        'inc100_moy': f'{nom}_inc100_moy',
        'inc100_max': f'{nom}_inc100_max',
        'nb_semaines': f'{nom}_nb_semaines',
    })

    log.info(f"    → {len(df_mois)} mois (2021+)")
    return df_mois


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(output_path='data/silver/J0_silver_sentinelles.csv'):
    log.info("=" * 60)
    log.info("INGEST SENTINELLES")
    log.info("=" * 60)

    # Base : tous les mois de 2021-01 à aujourd'hui
    all_months = pd.period_range('2021-01', pd.Timestamp.now().to_period('M'),
                                  freq='M').astype(str)
    df_final = pd.DataFrame({'annee_mois': all_months})

    # Ingestion de chaque indicateur
    for indic in INDICATEURS:
        try:
            df_ind = fetch_indicator(indic)
            df_final = df_final.merge(df_ind, on='annee_mois', how='left')
            log.info(f"  ✅ {indic['label']} intégré")
        except Exception as e:
            log.warning(f"  ⚠️  {indic['label']} échoué : {e}")

    # Métadonnées
    df_final['loaded_at'] = datetime.now().strftime('%Y%m%d')
    df_final['source'] = 'Sentinelles_INSERM'

    # Export
    df_final.to_csv(output_path, index=False, encoding='utf-8')

    log.info(f"\n── Aperçu ──")
    log.info(f"Shape : {df_final.shape}")
    log.info(f"\n{df_final.head(6).to_string()}")
    log.info(f"\n✅ Sauvegardé : {output_path}")

    return df_final


if __name__ == '__main__':
    run()
