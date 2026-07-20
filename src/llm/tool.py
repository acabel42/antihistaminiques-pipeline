from langchain_core.tools import tool

from src.ml.predict_model import predict_rupture


@tool
def predict_antihistamine_shortage(
    classe_atc: str,
    gram_moy: float,
    gram_max: float,
    gram_roll7: float,
    gram_roll30: float,
    nb_jours_pic: float,
    bouleau_moy: float,
    ambroisie_moy: float,
    nb_jours_pic_bouleau: float,
    temp_moy: float,
    temp_max: float,
    temp_roll30: float,
    precip: float,
    wind: float,
    mois: float,
    saison_allergies: float,
    source_encoded: float,
    ruptures_lag1: float,
    gram_lag_mois: float,
    cumul_thermique: float,
) -> dict:
    """
    Prédit le risque de rupture ou de tension
    pour une classe d'antihistaminiques.
    """

    input_data = {
        "gram_moy": gram_moy,
        "gram_max": gram_max,
        "gram_roll7": gram_roll7,
        "gram_roll30": gram_roll30,
        "nb_jours_pic": nb_jours_pic,
        "bouleau_moy": bouleau_moy,
        "ambroisie_moy": ambroisie_moy,
        "nb_jours_pic_bouleau": nb_jours_pic_bouleau,
        "temp_moy": temp_moy,
        "temp_max": temp_max,
        "temp_roll30": temp_roll30,
        "precip": precip,
        "wind": wind,
        "mois": mois,
        "saison_allergies": saison_allergies,
        "source_encoded": source_encoded,
        "ruptures_lag1": ruptures_lag1,
        "gram_lag_mois": gram_lag_mois,
        "cumul_thermique": cumul_thermique,
    }

    return predict_rupture(
        input_data=input_data,
        classe_atc=classe_atc,
    )