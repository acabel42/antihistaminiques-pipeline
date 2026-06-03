import pandas as pd
from sqlalchemy import create_engine

# ── Connexion PostgreSQL ──────────────────────────────────────────────────────
ENGINE = create_engine("postgresql://pipeline:pipeline2026@127.0.0.1:5432/antihistaminiques")

# ── Chemin vers le fichier Silver ─────────────────────────────────────────────
PATH_OPENMEDIC = "data/silver/J0_silver_openmedic_2021_2025.csv"


def build_features_openmedic():
    """Construit les features OpenMedic et charge la table openmedic_gold."""

    print("Chargement OpenMedic...")
    om = pd.read_csv(PATH_OPENMEDIC, low_memory=False)
    print(f"  → {om.shape[0]:,} lignes chargées")

    # S'assurer que BOITES est bien numérique
    om["BOITES"] = pd.to_numeric(om["BOITES"], errors="coerce")

    # ── Agrégation par région, année et sous-classe ATC4 ─────────────────────
    # On regroupe pour avoir une ligne par combinaison (région, année, ATC4)
    # C'est la maille à laquelle on va calculer les features
    om_agg = om.groupby(["BEN_REG", "annee", "ATC4"]).agg(
        boites=("BOITES", "sum"),
    ).reset_index()

    print(f"  → Agrégé : {om_agg.shape[0]:,} lignes")

    # ── Feature 1 : Moyenne glissante sur 3 ans ───────────────────────────────
    # Pour chaque région et sous-classe, on calcule la moyenne des 3 dernières années
    # Ça permet de lisser les variations ponctuelles et de voir la tendance
    # min_periods=1 : on calcule même si on a moins de 3 ans de données
    om_agg = om_agg.sort_values(["BEN_REG", "ATC4", "annee"])
    om_agg["rolling_mean_3a"] = (
        om_agg.groupby(["BEN_REG", "ATC4"])["boites"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # ── Feature 2 : Croissance Year-over-Year (YoY) ───────────────────────────
    # Variation en % des boîtes par rapport à l'année précédente
    # Ex : +0.15 = +15% de boîtes remboursées par rapport à l'année d'avant
    # shift(1) décale d'une ligne vers le bas pour récupérer la valeur précédente
    om_agg["boites_lag1"] = om_agg.groupby(["BEN_REG", "ATC4"])["boites"].shift(1)
    om_agg["croissance_yoy"] = (
        (om_agg["boites"] - om_agg["boites_lag1"]) / om_agg["boites_lag1"]
    )

    # fillna(0) : première année n'a pas de valeur précédente → on met 0
    # clip(-1, 5) : on plafonne entre -100% et +500% pour éviter les valeurs aberrantes
    om_agg["croissance_yoy"] = om_agg["croissance_yoy"].fillna(0).clip(-1, 5)

    # ── Feature 3 : Part de marché R06AX dans le total R06 par région ─────────
    # R06AX = sous-classe des antihistaminiques modernes (Desloratadine, Bilastine...)
    # Plus cette part est élevée, plus la région consomme des molécules récentes
    total_region_annee = om_agg.groupby(["BEN_REG", "annee"])["boites"].sum().reset_index()
    total_region_annee.columns = ["BEN_REG", "annee", "boites_total_region"]

    r06ax = om_agg[om_agg["ATC4"] == "R06AX"][["BEN_REG", "annee", "boites"]].copy()
    r06ax.columns = ["BEN_REG", "annee", "boites_r06ax"]

    om_agg = om_agg.merge(total_region_annee, on=["BEN_REG", "annee"], how="left")
    om_agg = om_agg.merge(r06ax, on=["BEN_REG", "annee"], how="left")

    om_agg["part_marche_r06ax"] = (
        om_agg["boites_r06ax"] / om_agg["boites_total_region"]
    ).fillna(0).round(4)

    print(f"  → Features calculées : rolling_mean_3a, croissance_yoy, part_marche_r06ax")

    # ── Chargement dans PostgreSQL ────────────────────────────────────────────
    print("Chargement dans PostgreSQL...")
    try:
        om_agg.to_sql("openmedic_gold", ENGINE, if_exists="replace", index=False)
        print(f"  → ✅ openmedic_gold chargée : {len(om_agg):,} lignes")
    except Exception as e:
        print(f"  → Erreur PostgreSQL : {e}")
        print("  → Vérifiez que Docker tourne : docker compose up -d postgres")

    print(f"  → Colonnes : {om_agg.columns.tolist()}")
    return om_agg


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== FEATURE ENGINEERING OPENMEDIC ===")
    om_gold = build_features_openmedic()
    print("=== TERMINÉ ✅ ===")