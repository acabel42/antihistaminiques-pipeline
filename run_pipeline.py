import logging
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)

CLASSES = ['R06', 'R03', 'J01']

def run():
    log.info('=== PIPELINE ANTIHISTAMINIQUES - DEBUT ===')

    log.info('Etape 1 - Nettoyage medicaments + ruptures')
    try:
        from src.cleaning.clean_medicaments_ruptures import clean_and_load
        clean_and_load()
        log.info('OK - medicaments + ruptures')
    except Exception as e:
        log.error(f'ERREUR etape 1 medicaments : {e}')
        raise

    log.info('Etape 1 - Nettoyage OpenMedic + BDPM')
    try:
        from src.cleaning.clean_openmedic import clean_openmedic, clean_bdpm
        clean_openmedic()
        clean_bdpm()
        log.info('OK - OpenMedic + BDPM')
    except Exception as e:
        log.error(f'ERREUR etape 1 openmedic : {e}')
        raise

    log.info('Etape 1 - Nettoyage pollen + meteo')
    try:
        from src.cleaning.clean_pollen_meteo import clean_pollen, clean_meteo
        clean_pollen()
        clean_meteo()
        log.info('OK - pollen + meteo')
    except Exception as e:
        log.error(f'ERREUR etape 1 pollen/meteo : {e}')
        raise

    log.info('Etape 1b - Ingestion ANSM disponibilites 2026')
    try:
        from src.ingestion.ingest_ruptures_ansm_2026 import run as ingest_ansm
        ingest_ansm(
            input_path='data/raw/export_disponibilites_ansm_2026.xlsx',
            openmedic_path='data/silver/J0_silver_openmedic_2021_2025.csv',
            output_path='data/silver/J0_silver_ruptures_ansm_2026.csv'
        )
        log.info('OK - J0_silver_ruptures_ansm_2026.csv genere')
    except Exception as e:
        log.warning(f'WARN etape 1b ANSM 2026 (non bloquant) : {e}')

    log.info('Etape 1c - Ingestion donnees Sentinelles')
    try:
        from src.ingestion.ingest_sentinelles import run as ingest_sent
        ingest_sent(output_path='data/silver/J0_silver_sentinelles.csv')
        log.info('OK - J0_silver_sentinelles.csv genere')
    except Exception as e:
        log.warning(f'WARN etape 1c Sentinelles (non bloquant) : {e}')

    log.info('Etape 1d - Feature engineering pollen')
    try:
        from src.transformations.features_pollen import build_features_pollen
        build_features_pollen()
        log.info('OK - pollen_meteo_features.csv genere')
    except Exception as e:
        log.error(f'ERREUR etape 1d features_pollen : {e}')
        raise

    log.info('Etape 2 - Construction Gold par classe ATC')
    try:
        from src.transformations.build_gold import build_gold
        for classe in CLASSES:
            build_gold(classe_atc=classe)
            log.info(f'OK - gold_ml_{classe}.csv genere')
    except Exception as e:
        log.error(f'ERREUR etape 2 build_gold : {e}')
        raise

    log.info('Etape 2b - Features avancees gold_ml_advanced.csv')
    try:
        from src.transformations.features_advanced import build_features_advanced
        build_features_advanced()
        log.info('OK - gold_ml_advanced.csv genere')
    except Exception as e:
        log.error(f'ERREUR etape 2b features_advanced : {e}')
        raise

    log.info('Etape 2c - Chargement OLAP PostgreSQL')
    try:
        from src.transformations.load_olap import load_olap
        load_olap()
        log.info('OK - tables OLAP chargees')
    except Exception as e:
        log.error(f'ERREUR etape 2c load_olap : {e}')
        raise

    log.info('Etape 3 - Entrainement des modeles ML par classe')
    try:
        from src.ml.train_model_Copy import train_model
        for classe in CLASSES:
            train_model(classe_atc=classe)
            log.info(f'OK - modeles sauvegardes dans models/{classe}/')
    except Exception as e:
        log.error(f'ERREUR etape 3 train_model_Copy : {e}')
        raise

    log.info('Etape 4 - Generation predictions par classe')
    try:
        from src.ml.predict import predict
        for classe in CLASSES:
            predict(classe_atc=classe)
            log.info(f'OK - gold_predictions_{classe}.csv genere')
    except Exception as e:
        log.error(f'ERREUR etape 4 predict : {e}')
        raise

    log.info('=== PIPELINE TERMINE - logs dans pipeline.log ===')

if __name__ == '__main__':
    run()
