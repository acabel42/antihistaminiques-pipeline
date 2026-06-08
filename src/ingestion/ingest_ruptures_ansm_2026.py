"""
ingest_ruptures_ansm_2026.py
────────────────────────────
Ingestion du fichier export ANSM "disponibilités médicaments" (XLS)
Enrichissement des codes ATC via jointure avec J0_silver_openmedic_2021_2025.csv
Sortie : data/silver/J0_silver_ruptures_ansm_2026.csv

Usage :
    python src/ingestion/ingest_ruptures_ansm_2026.py \
        --input  data/raw/export_disponibilites-des-medicaments_08-06-2026.xls \
        --openmedic data/silver/J0_silver_openmedic_2021_2025.csv \
        --output data/silver/J0_silver_ruptures_ansm_2026.csv
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
# Mapping domaine médical ANSM → classe ATC (fallback si jointure échoue)
DOMAINE_ATC_FALLBACK = {
    "Allergologie": "R06",
    "ORL":          "R06",
    "Pneumologie":  "R03",
    "Infectiologie":"J01",
}

# Classes ATC cibles du projet
CLASSES_CIBLES = {"R06", "R03", "J01"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def convert_xls_to_xlsx(xls_path: str) -> str:
    """Convertit un .xls en .xlsx via LibreOffice (gère la corruption xlrd)."""
    out_dir = tempfile.mkdtemp()
    log.info(f"  Conversion XLS → XLSX via LibreOffice...")
    result = subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "xlsx",
         "--outdir", out_dir, xls_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice conversion failed:\n{result.stderr}")

    stem = Path(xls_path).stem
    xlsx_path = os.path.join(out_dir, stem + ".xlsx")
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Fichier converti introuvable : {xlsx_path}")
    log.info(f"  → {xlsx_path}")
    return xlsx_path


def normalize_dci(text: str) -> str:
    """
    Normalise une DCI pour la jointure :
    - minuscules
    - supprime le sel/forme entre parenthèses : 'amikacine (sulfate d')' → 'amikacine'
    - supprime ponctuation résiduelle
    - strip
    """
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    # Supprime contenu entre parenthèses (sel pharmaceutique)
    t = re.sub(r"\s*\([^)]*\)", "", t)
    # Supprime accents pour comparaison plus large (optionnel)
    # t = unicodedata.normalize('NFD', t).encode('ascii','ignore').decode()
    t = t.strip(" .,;-")
    return t


def extract_dci(titre: str):
    """Extrait la DCI entre crochets dans le titre ANSM."""
    match = re.search(r"\[([^\]]+)\]", str(titre))
    return match.group(1).strip() if match else None


def extract_nom_commercial(titre: str):
    """Extrait le nom commercial (avant le premier tiret long ou crochet)."""
    match = re.match(r"^([^\[–\-]+)", str(titre))
    return match.group(1).strip() if match else str(titre)[:50]


def domaine_to_atc_fallback(domaine: str):
    """Mapping domaine médical → code ATC (fallback)."""
    if not isinstance(domaine, str):
        return None
    for kw, atc in DOMAINE_ATC_FALLBACK.items():
        if kw.lower() in domaine.lower():
            return atc
    return None


# ── Chargement ANSM ───────────────────────────────────────────────────────────

def load_ansm(input_path: str) -> pd.DataFrame:
    """Charge le fichier ANSM XLS/XLSX et prépare les colonnes."""
    path = input_path
    if input_path.lower().endswith(".xls"):
        path = convert_xls_to_xlsx(input_path)

    log.info(f"  Lecture ANSM : {path}")
    df = pd.read_excel(path)
    log.info(f"  → {len(df)} lignes, {len(df.columns)} colonnes")

    # Renommage snake_case
    df = df.rename(columns={
        "Titre":                        "titre",
        "Date de création":             "date_creation",
        "Date de mise à jour":          "date_maj",
        "Date de début de situation":   "date_debut",
        "Date de remise à disposition": "date_fin",
        "Statut":                       "statut",
        "Produit(s) de santé":          "type_produit",
        "Domaine(s) médical(aux)":      "domaine_medical",
        "URL de la page":               "url_ansm",
    })

    # Extraction DCI & nom commercial
    df["dci_extraite"]    = df["titre"].apply(extract_dci)
    df["nom_commercial"]  = df["titre"].apply(extract_nom_commercial)
    df["dci_norm"]        = df["dci_extraite"].apply(normalize_dci)

    # Parsing dates
    for col in ["date_creation", "date_maj", "date_debut", "date_fin"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Colonnes dérivées utiles
    df["annee_debut"]   = df["date_debut"].dt.year
    df["mois_debut"]    = df["date_debut"].dt.month
    df["annee_mois"]    = df["date_debut"].dt.to_period("M").astype(str)
    df["en_cours"]      = df["date_fin"].isna().astype(int)

    # Statut normalisé (binaire)
    df["is_rupture"] = df["statut"].str.contains(
        "Rupture de stock", case=False, na=False
    ).astype(int)
    df["is_tension"] = df["statut"].str.contains(
        "Tension", case=False, na=False
    ).astype(int)

    df["loaded_at"] = datetime.now().strftime("%Y%m%d")
    df["source"]    = "ANSM_DISPO_2026"

    return df


# ── Construction du référentiel ATC depuis OpenMedic ─────────────────────────

def build_atc_referentiel(openmedic_path: str) -> pd.DataFrame:
    """
    Construit un référentiel DCI → ATC à partir du Silver OpenMedic.
    On prend uniquement les colonnes ATC et L_ATC5 (DCI niveau 5).
    """
    log.info(f"  Lecture OpenMedic pour référentiel ATC : {openmedic_path}")

    cols = ["ATC1", "ATC2", "ATC3", "ATC4", "ATC5",
            "L_ATC5", "l_cip13"]

    df_om = pd.read_csv(
        openmedic_path,
        usecols=cols,
        low_memory=False
    )

    # Dédupliquer : on veut un mapping DCI → ATC unique
    df_atc = df_om[["L_ATC5", "ATC1", "ATC2", "ATC3", "ATC4", "ATC5"]].drop_duplicates()
    df_atc = df_atc.dropna(subset=["L_ATC5"])
    df_atc["dci_norm"] = df_atc["L_ATC5"].str.lower().str.strip()

    log.info(f"  → {len(df_atc)} DCI uniques dans OpenMedic")
    return df_atc


# ── Jointure ANSM × OpenMedic ─────────────────────────────────────────────────

def join_atc(df_ansm: pd.DataFrame, df_atc: pd.DataFrame) -> pd.DataFrame:
    """
    Jointure ANSM × OpenMedic sur dci_norm.
    Stratégie en 2 passes :
      1. Jointure exacte sur dci_norm
      2. Fallback domaine médical pour les non-matchés
    """
    log.info("  Jointure ATC (passe 1 : exacte sur DCI)...")

    # Passe 1 : jointure exacte
    df_merged = df_ansm.merge(
        df_atc[["dci_norm", "ATC1", "ATC2", "ATC3", "ATC4", "ATC5", "L_ATC5"]],
        on="dci_norm",
        how="left"
    )

    matched   = df_merged["ATC4"].notna().sum()
    unmatched = df_merged["ATC4"].isna().sum()
    log.info(f"  → Matchés (passe 1) : {matched} / {len(df_merged)}")
    log.info(f"  → Non matchés       : {unmatched}")

    # Passe 2 : fallback domaine médical
    log.info("  Jointure ATC (passe 2 : fallback domaine médical)...")
    mask_null = df_merged["ATC4"].isna()
    df_merged.loc[mask_null, "ATC4"] = df_merged.loc[mask_null, "domaine_medical"].apply(
        domaine_to_atc_fallback
    )
    df_merged.loc[mask_null & df_merged["ATC4"].notna(), "ATC4_source"] = "fallback_domaine"
    df_merged.loc[~mask_null, "ATC4_source"] = "openmedic_join"

    fallback_count = (df_merged["ATC4_source"] == "fallback_domaine").sum()
    still_null     = df_merged["ATC4"].isna().sum()
    log.info(f"  → Récupérés via fallback domaine : {fallback_count}")
    log.info(f"  → Sans ATC (exclus)              : {still_null}")

    return df_merged


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(input_path: str, openmedic_path: str, output_path: str,
        classes=None) -> pd.DataFrame:

    log.info("=" * 60)
    log.info("INGEST RUPTURES ANSM 2026")
    log.info("=" * 60)

    # 1. Chargement ANSM
    log.info("[1/4] Chargement fichier ANSM...")
    df_ansm = load_ansm(input_path)

    # 2. Référentiel ATC depuis OpenMedic
    log.info("[2/4] Construction référentiel ATC (OpenMedic)...")
    df_atc = build_atc_referentiel(openmedic_path)

    # 3. Jointure
    log.info("[3/4] Jointure ANSM × OpenMedic...")
    df_final = join_atc(df_ansm, df_atc)

    # 4. Filtre classes ATC cibles (optionnel)
    if classes:
        log.info(f"[4/4] Filtre classes ATC : {classes}...")
        df_final = df_final[df_final["ATC4"].isin(classes)]
        log.info(f"  → {len(df_final)} lignes après filtre")
    else:
        log.info("[4/4] Pas de filtre ATC (toutes classes conservées)")

    # Résumé par classe
    log.info("\n── Résumé par classe ATC ──")
    vc = df_final["ATC4"].value_counts(dropna=False)
    log.info(f"\n{vc.to_string()}")

    log.info("\n── Répartition par statut ──")
    log.info(f"\n{df_final['statut'].value_counts().to_string()}")

    # Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding="utf-8")
    log.info(f"\n✅  Fichier sauvegardé : {output_path}  ({len(df_final)} lignes)")

    return df_final


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingestion ANSM disponibilités médicaments → Silver enrichi ATC"
    )
    parser.add_argument(
        "--input", required=True,
        help="Chemin vers le fichier XLS/XLSX ANSM"
    )
    parser.add_argument(
        "--openmedic", required=True,
        help="Chemin vers J0_silver_openmedic_2021_2025.csv"
    )
    parser.add_argument(
        "--output", default="data/silver/J0_silver_ruptures_ansm_2026.csv",
        help="Chemin de sortie CSV Silver"
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Filtrer par classes ATC ex: --classes R06 R03 J01 (défaut: tout garder)"
    )
    args = parser.parse_args()

    df = run(
        input_path=args.input,
        openmedic_path=args.openmedic,
        output_path=args.output,
        classes=args.classes,
    )

    # Aperçu final
    print("\n── Aperçu (10 premières lignes) ──")
    cols_display = [
        "nom_commercial", "dci_extraite", "statut",
        "ATC4", "ATC4_source", "domaine_medical",
        "date_debut", "annee_mois", "en_cours"
    ]
    print(df[[c for c in cols_display if c in df.columns]].head(10).to_string())
