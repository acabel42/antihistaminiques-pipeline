from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "antihistaminiques_R06_classifier"
MODEL_ALIAS = "champion"


def build_prediction_dataset(classe_atc: str = "R06") -> pd.DataFrame:
    """
    Reconstruit les données exactement comme train_model_Copy.py :
    - charge gold_ml_<classe>.csv ;
    - fusionne gold_ml_advanced.csv sur annee_mois_str ;
    - ne filtre pas gold_ml_advanced par classe, afin de rester compatible
      avec le modèle déjà entraîné.
    """
    gold_path = PROJECT_ROOT / "data" / "gold" / f"gold_ml_{classe_atc}.csv"
    advanced_path = PROJECT_ROOT / "data" / "gold" / "gold_ml_advanced.csv"

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold introuvable : {gold_path}")

    df = pd.read_csv(gold_path)

    if df.empty:
        raise ValueError(f"Le fichier {gold_path.name} est vide.")

    if advanced_path.exists():
        df_advanced = pd.read_csv(advanced_path)

        advanced_columns = [
            "annee_mois_str",
            "ruptures_lag1",
            "gram_lag_mois",
            "cumul_thermique",
        ]
        advanced_columns = [
            column
            for column in advanced_columns
            if column in df_advanced.columns
        ]

        # Même fusion que dans le script d'entraînement d'origine.
        df = df.merge(
            df_advanced[advanced_columns],
            on="annee_mois_str",
            how="left",
        )

    return df


def predict_latest_rupture(classe_atc: str = "R06") -> dict:
    """
    Charge le modèle MLflow portant l'alias champion et prédit le risque
    pour la dernière période complète disponible.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    df = build_prediction_dataset(classe_atc)

    # La liste exacte des variables est récupérée depuis le modèle entraîné.
    if hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)
    else:
        raise AttributeError(
            "Le modèle ne contient pas feature_names_in_. "
            "Réentraîne-le avec un DataFrame pandas."
        )

    missing_columns = [
        column for column in feature_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Colonnes attendues par le modèle mais absentes : "
            + ", ".join(missing_columns)
        )

    usable_rows = df.dropna(subset=feature_columns).copy()

    if usable_rows.empty:
        null_counts = (
            df[feature_columns]
            .isna()
            .sum()
            .sort_values(ascending=False)
        )
        raise ValueError(
            "Aucune ligne complète n'est disponible pour la prédiction.\n"
            f"Valeurs manquantes par variable :\n{null_counts}"
        )

    usable_rows = usable_rows.sort_values("annee_mois_str")
    latest_row = usable_rows.iloc[[-1]].copy()

    X_latest = latest_row[feature_columns].astype("float64")

    prediction = int(model.predict(X_latest)[0])
    probability = float(model.predict_proba(X_latest)[0, 1])

    return {
        "classe_atc": classe_atc,
        "periode": str(latest_row["annee_mois_str"].iloc[0]),
        "prediction": prediction,
        "niveau_risque": "élevé" if prediction == 1 else "faible",
        "probabilite_rupture": round(probability * 100, 2),
        "gram_moy": float(latest_row["gram_moy"].iloc[0]),
        "temp_moy": float(latest_row["temp_moy"].iloc[0]),
        "precip": float(latest_row["precip"].iloc[0]),
        "ruptures_lag1": (
            float(latest_row["ruptures_lag1"].iloc[0])
            if "ruptures_lag1" in latest_row.columns
            else "Non disponible"
        ),
    }


if __name__ == "__main__":
    print(predict_latest_rupture())
