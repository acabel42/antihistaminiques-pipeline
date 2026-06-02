import pandas as pd
from sqlalchemy import create_engine

# ── Connexion à la base PostgreSQL ───────────────────────────────────────────
# On crée le "moteur" de connexion avec les identifiants du docker-compose.yml
# localhost:5433 car on a changé le port pour éviter le conflit
ENGINE = create_engine("postgresql://pipeline:pipeline2026@localhost:5433/antihistaminiques")

# ── Chemins vers les fichiers Silver ─────────────────────────────────────────
PATH_OPENMEDIC = "data/silver/J0_silver_openmedic_2021_2025.csv"
PATH_BDPM      = "data/silver/J0_silver_bdpm.csv"

# ── Dictionnaire de correspondance codes région → noms ───────────────────────
# BEN_REG dans OpenMedic contient des codes INSEE, pas des noms lisibles
REGIONS = {
    11: "Île-de-France",           24: "Centre-Val-de-Loire",
    27: "Bourgogne-Franche-Comté", 28: "Normandie",
    32: "Hauts-de-France",         44: "Grand Est",
    52: "Pays-de-la-Loire",        53: "Bretagne",
    75: "Nouvelle-Aquitaine",      76: "Occitanie",
    84: "Auvergne-Rhône-Alpes",    93: "PACA",
     5: "Corse",                   99: "Hors région",
     0: "Non renseigné"
}

# ── Dictionnaire tranches d'âge ───────────────────────────────────────────────
AGES = {
     0: "0-19 ans",
    20: "20-59 ans",
    60: "60 ans et +",
    99: "Non renseigné"
}

# ── Dictionnaire sexe ─────────────────────────────────────────────────────────
SEXES = {1: "Homme", 2: "Femme", 9: "Non renseigné"}


def clean_openmedic():
    """Nettoie le fichier OpenMedic et retourne un DataFrame propre."""

    print("Chargement OpenMedic...")
    df = pd.read_csv(PATH_OPENMEDIC, low_memory=False)
    print(f"  → {df.shape[0]:,} lignes chargées")

    # Suppression des colonnes inutiles
    # AGE : 77% de nulls, c'est un doublon de la colonne 'age' déjà propre
    # REM et BSE : versions texte avec des virgules, déjà nettoyées dans REM_clean et BSE_clean
    df = df.drop(columns=["AGE", "REM", "BSE"])
    print("  → Colonnes inutiles supprimées (AGE, REM, BSE)")

    # Ajout de colonnes lisibles à partir des codes
    # .map() remplace chaque code par sa valeur dans le dictionnaire
    df["region_nom"] = df["BEN_REG"].map(REGIONS)
    df["age_label"]  = df["age"].map(AGES).fillna("Non renseigné")
    df["sexe_label"] = df["sexe"].map(SEXES).fillna("Non renseigné")
    print("  → Colonnes region_nom, age_label, sexe_label ajoutées")

    # Les nulls sur 'age' sont conservés volontairement
    # Ce sont des données anonymisées non disponibles dans la source OpenMedic
    # Les imputer (remplacer par une valeur) serait faux
    print(f"  → Nulls sur age : {df['age'].isnull().sum():,} (conservés — données anonymisées)")

    print(f"  → Nettoyage terminé : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
    return df


def clean_bdpm():
    """Nettoie le fichier BDPM et retourne un DataFrame propre."""

    print("Chargement BDPM...")
    df = pd.read_csv(PATH_BDPM)
    print(f"  → {df.shape[0]:,} lignes chargées")

    # Conversion du code CIS en string
    # C'est un identifiant, pas un nombre — on ne fait pas de calculs dessus
    df["cis"] = df["cis"].astype(str)

    # Extraction des 7 premiers caractères du code ATC
    # OpenMedic utilise ATC5 sur 7 caractères (ex: R06AX27)
    # BDPM a parfois des codes plus longs — on tronque pour harmoniser
    df["ATC5"] = df["atc"].str[:7]

    # Les 1361 nulls sur dosage sont conservés
    # C'est une info optionnelle dans la BDPM, pas nécessaire pour le ML
    print(f"  → Nulls sur dosage : {df['dosage'].isnull().sum()} (conservés — info optionnelle)")

    print(f"  → Nettoyage terminé : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
    return df


def load_to_postgres(om, bdpm):
    """Charge les DataFrames nettoyés dans PostgreSQL."""

    print("Chargement dans PostgreSQL...")

    # Vérification que la connexion fonctionne avant de charger
    try:
        with ENGINE.connect() as conn:
            print("  → Connexion PostgreSQL OK ✅")
    except Exception as e:
        print(f"  → Connexion échouée : {e}")
        print("  → Vérifiez que Docker tourne : docker compose up -d postgres")
        return

    # Chargement OpenMedic
    # if_exists='replace' : si la table existe déjà, on la recrée
    # chunksize=5000 : on charge par blocs de 5000 lignes car le fichier est lourd (68 Mo)
    print("  → Chargement table openmedic...")
    om.to_sql("openmedic", ENGINE, if_exists="replace", index=False, chunksize=5000)
    print(f"  → ✅ openmedic chargée : {len(om):,} lignes")

    # Chargement BDPM
    print("  → Chargement table bdpm...")
    bdpm.to_sql("bdpm", ENGINE, if_exists="replace", index=False)
    print(f"  → ✅ bdpm chargée : {len(bdpm):,} lignes")


# ── Point d'entrée principal ──────────────────────────────────────────────────
# Ce bloc s'exécute uniquement quand on lance : python src/cleaning/clean_openmedic.py
# Il ne s'exécute pas si le fichier est importé par un autre script
if __name__ == "__main__":
    print("=== NETTOYAGE OPENMEDIC & BDPM ===")
    om   = clean_openmedic()
    bdpm = clean_bdpm()
    load_to_postgres(om, bdpm)
    print("=== TERMINÉ ✅ ===")