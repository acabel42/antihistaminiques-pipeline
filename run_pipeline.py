import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
log = logging.getLogger(__name__)

def run():
    os.chdir('/Users/nellyta/Jedha')
    log.info('=== PIPELINE ANTIHISTAMINIQUES ===')

    # ÉTAPE 1 — NETTOYAGE SILVER
    log.info('Etape 1 — Nettoyage medicaments + ruptures')
    from src.cleaning.clean_medicaments_ruptures import clean_and_load
    clean_and_load()

    log.info('Etape 2 — Nettoyage Open Medic + BDPM')
    from src.cleaning.clean_openmedic import clean_and_load as clean_openmedic
    clean_openmedic()

    log.info('Etape 3 — Nettoyage pollen + meteo')
    from src.cleaning.clean_pollen_meteo import clean_pollen, clean_meteo
    clean_pollen()
    clean_meteo()

    log.info('=== PIPELINE TERMINE ===')

if __name__ == '__main__':
    run()