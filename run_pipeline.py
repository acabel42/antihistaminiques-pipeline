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


def run():
    log.info('=== PIPELINE ANTIHISTAMINIQUES - DEBUT ===')

    # ETAPE 1 - Nettoyage Silver
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

    # ETAPE 1b - Feature engineering pollen
    log.info('Etape 1b - Feature engineering pollen')
    try:
        from src.transformations.features_pollen import build_features_pollen
        build_features_pollen()
        log.info('OK - pollen_meteo_features.csv genere')
    except Exception as e:
        log.error(f'ERREUR etape 1b features_pollen : {e}')
        raise

    # ETAPE 2 - Construction Gold
    log.info('Etape 2 - Construction gold_ml.csv')
    try:
        from src.transformations.build_gold import build_gold
        build_gold()
        log.info('OK - gold_ml.csv genere')
    except Exception as e:
        log.error(f'ERREUR etape 2 build_gold : {e}')
        raise

    log.info('Etape 2 - Features avancees gold_ml_advanced.csv')
    try:
        from src.transformations.features_advanced import build_features_advanced
        build_features_advanced()
        log.info('OK - gold_ml_advanced.csv genere')
    except Exception as e:
        log.error(f'ERREUR etape 2 features_advanced : {e}')
        raise

    # ETAPE 2b - Load OLAP
    log.info('Etape 2b - Chargement OLAP PostgreSQL')
    try:
        from src.transformations.load_olap import load_olap
        load_olap()
        log.info('OK - tables OLAP chargees')
    except Exception as e:
        log.error(f'ERREUR etape 2b load_olap : {e}')
        raise

    # ETAPE 3 - Entrainement ML
    log.info('Etape 3 - Entrainement des modeles ML')
    try:
        from src.ml.train_model import train_model
        train_model()
        log.info('OK - modeles sauvegardes dans models/')
    except Exception as e:
        log.error(f'ERREUR etape 3 train_model : {e}')
        raise

    # ETAPE 4 - Predictions
    log.info('Etape 4 - Generation predictions')
    try:
        from src.ml.predict import predict
        predict()
        log.info('OK - gold_predictions genere')
    except Exception as e:
        log.error(f'ERREUR etape 4 predict : {e}')
        raise

    log.info('=== PIPELINE TERMINE - logs dans pipeline.log ===')


if __name__ == '__main__':
    run()