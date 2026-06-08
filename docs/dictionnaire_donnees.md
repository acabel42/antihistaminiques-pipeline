# Dictionnaire de données — Antihistaminiques Pipeline

Projet Jedha Data Science Bootcamp — Prédiction ruptures de stock R06A en France  
Pipeline : Bronze → Silver → Gold → ML → Dashboard

---

## Table des matières

1. [Silver — medicaments](#1-silver--medicaments)
2. [Silver — ruptures](#2-silver--ruptures)
3. [Silver — bdpm](#3-silver--bdpm)
4. [Silver — openmedic](#4-silver--openmedic)
5. [Gold — gold_ml_advanced](#5-gold--gold_ml_advanced)
6. [Modèles ML](#6-modèles-ml)

---

## 1. Silver — medicaments

**Fichier :** `data/silver/J0_silver_medicaments.csv`  
**Source :** OpenMedic (AMELI) + BDPM (ANSM)  
**Description :** Un médicament par ligne, avec profil démographique des patients et appartenance à la classe antihistaminique.

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `cis` | int | Code Identifiant de Spécialité — identifiant unique du médicament dans la BDPM | 61266250 |
| `nom_medicament` | str | Nom commercial du médicament | "cetirizine biogaran 10mg" |
| `laboratoire` | str | Nom du laboratoire fabricant | "biogaran" |
| `code_atc` | str | Code ATC (classification anatomique thérapeutique) sur 7 caractères | "R06AE07" |
| `molecule` | str | Nom de la molécule active | "cetirizine" |
| `nb_patients_ville` | float | Nombre de patients ayant consommé ce médicament en ville | 125430.0 |
| `substance_active` | str | Substance active principale (BDPM) | "CETIRIZINE DICHLORHYDRATE" |
| `pct_age_0_19_ans` | float | Part des patients âgés de 0 à 19 ans (%) | 18.4 |
| `pct_age_20_59_ans` | float | Part des patients âgés de 20 à 59 ans (%) | 52.1 |
| `pct_age_60_ans_et_plus` | float | Part des patients âgés de 60 ans et plus (%) | 29.5 |
| `pct_sexe_female` | float | Part des patientes femmes (%) | 61.0 |
| `pct_sexe_male` | float | Part des patients hommes (%) | 39.0 |
| `est_antihistaminique` | bool | True si le médicament appartient à la classe ATC R06A | True |

---

## 2. Silver — ruptures

**Fichier :** `data/silver/J0_silver_ruptures.csv`  
**Source :** ANSM — Déclarations de ruptures et risques de rupture de stock  
**Description :** Un événement de rupture ou de risque par ligne.

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `cis` | int | Code Identifiant de Spécialité du médicament concerné | 66136969 |
| `nom_medicament` | str | Nom commercial du médicament | "cetirizine mylan 10mg" |
| `cause` | str | Cause déclarée de la rupture (texte libre ANSM) | "Problème d'approvisionnement en matière première" |
| `classification` | str | Type d'événement : "rupture" ou "risque" | "rupture" |
| `date_evenement` | str | Date de l'événement au format YYYY-MM-DD | "2022-05-14" |
| `code_atc` | str | Code ATC du médicament sur 7 caractères | "R06AE07" |
| `molecule` | str | Molécule active concernée | "cetirizine" |
| `laboratoire` | str | Laboratoire déclarant | "mylan" |
| `nb_patients_ville` | float | Nombre de patients potentiellement impactés | 45000.0 |
| `est_antihistaminique` | bool | True si appartient à la classe R06A | True |
| `annee` | int | Année extraite de date_evenement | 2022 |
| `mois` | int | Mois extrait de date_evenement (1-12) | 5 |
| `trimestre` | int | Trimestre (1-4) | 2 |
| `saison_allergies` | int | 1 si mois de pic allergique (avril-août), 0 sinon | 1 |
| `cause_categorie` | str | Catégorie normalisée de la cause | "Approvisionnement matiere" |

---

## 3. Silver — bdpm

**Fichier :** `data/silver/J0_silver_bdpm.csv`  
**Source :** Base de Données Publique des Médicaments (ANSM)  
**Description :** Référentiel des médicaments autorisés en France.

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `cis` | int | Code Identifiant de Spécialité — clé de jointure avec les autres tables | 60003620 |
| `atc` | str | Code ATC complet (peut dépasser 7 caractères) | "R06AE07" |
| `denomination` | str | Dénomination officielle complète du médicament | "CETIRIZINE BIOGARAN 10 mg, comprimé pelliculé" |
| `lien` | str | URL de la fiche médicament sur base-donnees-publique.medicaments.gouv.fr | "https://..." |
| `substance` | str | Substance active en majuscules | "CETIRIZINE DICHLORHYDRATE" |
| `dosage` | str | Dosage unitaire (peut être null pour certaines formes) | "10,00 mg" |
| `est_antihistaminique` | bool | True si code ATC commence par R06A | True |

---

## 4. Silver — openmedic

**Fichier :** `data/silver/J0_silver_openmedic_2021_2025.csv`  
**Source :** OpenMedic (AMELI) — Consommation remboursée en ville  
**Description :** Consommation de médicaments par région, année, tranche d'âge et sexe. 1,24 million de lignes. Filtré sur les classes ATC R06A (antihistaminiques), R03 (respiratoire), J01 (antibiotiques).

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `annee` | int | Année de consommation | 2023 |
| `BEN_REG` | int | Code INSEE de la région bénéficiaire | 11 |
| `region_nom` | str | Nom de la région (décodé depuis BEN_REG) | "Île-de-France" |
| `ATC4` | str | Code ATC sur 4 caractères (classe thérapeutique) | "R06A" |
| `BOITES` | float | Nombre de boîtes remboursées | 254300.0 |
| `REM_clean` | float | Montant remboursé en euros | 485230.50 |
| `BSE_clean` | float | Base de remboursement en euros | 512000.00 |
| `age` | float | Tranche d'âge (0=0-19, 20=20-59, 60=60+, 99=non renseigné) | 20 |
| `age_label` | str | Tranche d'âge en texte lisible | "20-59 ans" |
| `sexe` | int | Code sexe (1=Homme, 2=Femme, 9=Non renseigné) | 2 |
| `sexe_label` | str | Sexe en texte lisible | "Femme" |

---

## 5. Gold — gold_ml_advanced

**Fichier :** `data/gold/gold_ml_advanced.csv`  
**Source :** Construit par `build_gold.py` + `features_advanced.py`  
**Description :** Table de features agrégées au niveau mensuel, utilisée pour l'entraînement des modèles ML. Une ligne = un mois.

### Features pollen (graminees)

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `annee_mois_str` | str | Période au format YYYY-MM — clé temporelle | "2022-05" |
| `annee` | int | Année | 2022 |
| `mois` | int | Mois (1-12) | 5 |
| `gram_moy` | float | Concentration moyenne mensuelle en graminées (grains/m³) | 12.4 |
| `gram_max` | float | Concentration maximale mensuelle en graminées (grains/m³) | 45.2 |
| `gram_roll7` | float | Moyenne mobile sur 7 jours des graminées | 10.8 |
| `gram_roll30` | float | Moyenne mobile sur 30 jours des graminées | 8.3 |
| `gram_lag_mois` | float | Valeur de gram_moy du mois précédent (lag 1 mois) | 9.1 |
| `gram_lag_2mois` | float | Valeur de gram_moy de 2 mois avant (lag 2 mois) | 6.5 |
| `gram_roll3m` | float | Moyenne de gram_moy sur les 3 derniers mois | 9.4 |
| `nb_jours_pic` | int | Nombre de jours avec pic de graminées dans le mois | 4 |
| `ratio_pic_saison` | float | Ratio entre nb_jours_pic et la moyenne de la saison | 0.73 |

### Features autres pollens

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `bouleau_moy` | float | Concentration moyenne mensuelle en pollen de bouleau (grains/m³) | 5.2 |
| `ambroisie_moy` | float | Concentration moyenne mensuelle en pollen d'ambroisie (grains/m³) | 3.1 |
| `aulne_moy` | float | Concentration moyenne mensuelle en pollen d'aulne (grains/m³) | 1.8 |
| `armoise_moy` | float | Concentration moyenne mensuelle en pollen d'armoise (grains/m³) | 2.4 |
| `olivier_moy` | float | Concentration moyenne mensuelle en pollen d'olivier (grains/m³) | 0.9 |
| `nb_jours_pic_bouleau` | int | Nombre de jours avec pic de bouleau dans le mois | 2 |
| `nb_jours_pic_ambroisie` | int | Nombre de jours avec pic d'ambroisie dans le mois | 3 |
| `nb_jours_pic_armoise` | int | Nombre de jours avec pic d'armoise dans le mois | 1 |
| `nb_jours_pic_olivier` | int | Nombre de jours avec pic d'olivier dans le mois | 0 |
| `nb_jours_pic_aulne` | int | Nombre de jours avec pic d'aulne dans le mois | 1 |

### Features météo

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `temp_moy` | float | Température moyenne mensuelle (°C) | 18.5 |
| `temp_max` | float | Température maximale mensuelle (°C) | 28.3 |
| `temp_min` | float | Température minimale mensuelle (°C) | 10.1 |
| `temp_roll30` | float | Moyenne mobile sur 30 jours de la température | 17.2 |
| `cumul_thermique` | float | Cumul des degrés-jours au-dessus de 5°C depuis janvier — indicateur phénologique | 345.0 |
| `precip` | float | Précipitations moyennes mensuelles (mm) | 42.3 |
| `precip_lag7` | float | Précipitations de la semaine précédente (mm) | 38.1 |
| `wind` | float | Vitesse moyenne du vent mensuelle (km/h) | 12.4 |

### Features temporelles et contextuelles

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `saison_allergies` | int | 1 si mois de saison allergique (avril-août), 0 sinon | 1 |
| `source_encoded` | float | Source des données pollen encodée numériquement (biais capteur) | 0.62 |

### Features ruptures (target et historique)

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `nb_ruptures` | int | Nombre de ruptures de stock déclarées dans le mois — **non utilisé en feature ML (leakage)** | 2 |
| `nb_risques` | int | Nombre de risques de rupture déclarés dans le mois — **non utilisé en feature ML (leakage)** | 1 |
| `target_rupture` | int | Variable cible : 1 si au moins une rupture ou tension ce mois, 0 sinon | 1 |
| `ruptures_lag1` | float | Valeur de target_rupture du mois précédent — utilisée comme feature | 0.0 |

### Features consommation

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `boites_total` | float | Total de boîtes d'antihistaminiques R06A remboursées cette année | 4523000.0 |
| `classe_atc` | str | Classe ATC filtrée pour cette table (R06, R03 ou J01) | "R06" |

---

## 6. Modèles ML

**Dossier :** `models/`

| Fichier | Algorithme | Tâche | Métrique principale |
|---------|-----------|-------|-------------------|
| `lr_baseline.joblib` | Régression Logistique | Classification rupture/pas rupture | ROC-AUC = 0.457 |
| `rf_classifier.joblib` | Random Forest Classifier | Classification rupture/pas rupture | ROC-AUC = 0.771, F1 CV = 0.520 ± 0.176 |
| `rf_regressor.joblib` | Random Forest Regressor | Prédiction gram_moy mois suivant | R² = 0.513, RMSE = 6.300 grains/m³ |

**Notes importantes :**
- `nb_ruptures` et `nb_risques` sont **exclus des features** du classifier car ils composent directement `target_rupture` (data leakage)
- `class_weight='balanced'` appliqué sur le classifier pour compenser le déséquilibre de classes (~37% de mois avec rupture)
- GridSearchCV testé sur 24 combinaisons × 5 folds — meilleurs paramètres : `max_depth=5, n_estimators=200, min_samples_split=2`
- SHAP identifie `ambroisie_moy` comme feature la plus prédictive des ruptures

---
