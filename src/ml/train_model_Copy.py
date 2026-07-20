import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (r2_score, mean_squared_error,
                             classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, balanced_accuracy_score,)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('models', exist_ok=True)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("antihistaminiques_ml")

print(f"MLflow Tracking URI : {mlflow.get_tracking_uri()}")

FEATURES_REG = [
    'gram_moy', 'gram_max', 'gram_roll7', 'nb_jours_pic',
    'temp_moy', 'precip', 'mois', 'saison_allergies'
]

FEATURES_CLF = [
    'gram_moy', 'gram_max', 'gram_roll7', 'gram_roll30', 'nb_jours_pic',
    'bouleau_moy', 'ambroisie_moy', 'nb_jours_pic_bouleau',
    'temp_moy', 'temp_max', 'temp_roll30',
    'precip', 'wind',
    'mois', 'saison_allergies', 'source_encoded',
    'ruptures_lag1', 'gram_lag_mois', 'cumul_thermique',
    # Sentinelles (R03/J01 uniquement — ignorees si absent du Gold)
    'grippal_inc100_moy', 'grippal_inc100_max',
    'ira_inc100_moy', 'ira_inc100_max',
    'diarrhee_inc100_moy', 'diarrhee_inc100_max',
    'varicelle_inc100_moy', 'varicelle_inc100_max',
]

def calculate_file_sha256(file_path):
    """Calcule l'empreinte SHA-256 d'un fichier."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            sha256.update(block)

    return sha256.hexdigest()

def train_baseline(df, classe_atc="R06"):
    print("\n=== BASELINE — Regression Logistique ===")
    features_base = ['gram_moy', 'temp_moy', 'precip', 'mois']
    df_b = df.dropna(subset=features_base + ['target_rupture'])
    X = df_b[features_base]
    y = df_b['target_rupture']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)  # 0.3 au lieu de 0.2

    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, zero_division=0))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"  ROC-AUC Baseline : {roc_auc:.3f}")
    mlflow.log_metric("baseline_roc_auc", roc_auc)

    joblib.dump(lr, f'models/{classe_atc}/lr_baseline.joblib')
    print(f'  Modele sauvegarde : models/{classe_atc}/lr_baseline.joblib')
    return lr

def train_regressor(df, classe_atc="R06"):
    print("\n=== MODELE 1 — Regression gram_moy mois suivant ===")
    df = df.copy().sort_values('annee_mois_str').reset_index(drop=True)
    df['gram_moy_next'] = df['gram_moy'].shift(-1)
    df = df.dropna(subset=['gram_moy_next'] + FEATURES_REG)

    X = df[FEATURES_REG]
    y = df['gram_moy_next']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)  # 0.3 au lieu de 0.2

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_reg.fit(X_train, y_train)

    y_pred = rf_reg.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("regressor_r2", r2)
    mlflow.log_metric("regressor_rmse", rmse)

    print(f"  R²   : {r2:.3f}")
    print(f"  RMSE : {rmse:.3f} grains/m3")

    cv = cross_val_score(rf_reg, X, y, cv=5, scoring='r2')
    mlflow.log_metric("regressor_r2_cv_mean", float(cv.mean()))
    mlflow.log_metric("regressor_r2_cv_std", float(cv.std()))
    print(f"  R² CV (5-fold) : {cv.mean():.3f} +/- {cv.std():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RF Regressor — Prediction graminees mois suivant', fontsize=13, fontweight='bold')

    axes[0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=50)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--', linewidth=1)
    axes[0].set_xlabel('Reel (grains/m3)')
    axes[0].set_ylabel('Predit (grains/m3)')
    axes[0].set_title(f'Reel vs Predit\nR² = {r2:.3f} | RMSE = {rmse:.3f}')
    axes[0].grid(True, alpha=0.3)

    imp = pd.DataFrame({
        'feature': FEATURES_REG,
        'importance': rf_reg.feature_importances_
    }).sort_values('importance', ascending=True)
    axes[1].barh(imp['feature'], imp['importance'], color='steelblue')
    axes[1].set_title('Feature Importance')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('notebooks/ml_regressor_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    joblib.dump(rf_reg, f'models/{classe_atc}/rf_regressor.joblib')
    print(f'  Modele sauvegarde : models/{classe_atc}/rf_regressor.joblib')
    return rf_reg, df

def train_classifier(df, classe_atc="R06"):
    print("\n=== MODELE 2 — Classification rupture/tension R06 ===")

    features_disponibles = [f for f in FEATURES_CLF if f in df.columns]

    mlflow.log_param(
        "classifier_nombre_features",
        len(features_disponibles),
        )

    mlflow.log_param(
        "classifier_features",
        ", ".join(features_disponibles),
        )

    df_clf = df.dropna(subset=features_disponibles + ['target_rupture'])
    X = df_clf[features_disponibles]
    y = df_clf['target_rupture']

    print(f"  Target : {y.value_counts().to_dict()}")
    print(f"  Features utilisees : {len(features_disponibles)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)  # 0.3 au lieu de 0.2

    # SMOTE : rééquilibrage des classes sur le train uniquement (jamais sur le test)
    sm = SMOTE(random_state=42, k_neighbors=min(3, y_train.value_counts().min() - 1))
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"  Apres SMOTE — train : {dict(pd.Series(y_train).value_counts())}")

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,           # réduit de 10 → 5 pour limiter l'overfitting
        class_weight='balanced',
        random_state=42)
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)
    y_prob = rf_clf.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, zero_division=0))
    
    metrics_classifier = {
        "classifier_test_accuracy": accuracy_score(y_test, y_pred),
        "classifier_test_balanced_accuracy": balanced_accuracy_score(
            y_test,
            y_pred,
        ),
        
        "classifier_test_precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        
        "classifier_test_recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),


        "classifier_test_f1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        
        "classifier_test_pr_auc": average_precision_score(
            y_test,
            y_prob,
        ),
            
        "classifier_test_roc_auc": roc_auc_score(
            y_test,
            y_prob,
        ),
    }
    
    mlflow.log_metrics(metrics_classifier)
    
    roc = metrics_classifier["classifier_test_roc_auc"]
    
    print(f"  Accuracy          : {metrics_classifier['classifier_test_accuracy']:.3f}")
    print(f"  Balanced accuracy : {metrics_classifier['classifier_test_balanced_accuracy']:.3f}")
    print(f"  Precision         : {metrics_classifier['classifier_test_precision']:.3f}")
    print(f"  Recall            : {metrics_classifier['classifier_test_recall']:.3f}")
    print(f"  F1-score          : {metrics_classifier['classifier_test_f1']:.3f}")
    print(f"  PR-AUC            : {metrics_classifier['classifier_test_pr_auc']:.3f}")
    print(f"  ROC-AUC           : {metrics_classifier['classifier_test_roc_auc']:.3f}")

    roc_cv = cross_val_score(rf_clf, X, y, cv=5, scoring='roc_auc')
    f1_cv = cross_val_score(rf_clf, X, y, cv=5, scoring='f1_weighted')
    mlflow.log_metric("classifier_initial_roc_auc_cv_mean", float(roc_cv.mean()))
    mlflow.log_metric("classifier_initial_roc_auc_cv_std", float(roc_cv.std()))
    mlflow.log_metric("classifier_initial_f1_cv_mean", float(f1_cv.mean()))
    mlflow.log_metric("classifier_initial_f1_cv_std", float(f1_cv.std()))
    print(f"  ROC-AUC CV (5-fold) : {roc_cv.mean():.3f} +/- {roc_cv.std():.3f}")
    print(f"  F1 CV (5-fold) : {f1_cv.mean():.3f} +/- {f1_cv.std():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RF Classifier — Detection rupture/tension R06', fontsize=13, fontweight='bold')

    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Matrice de confusion')
    axes[0].set_xlabel('Predit')
    axes[0].set_ylabel('Reel')

    imp = pd.DataFrame({
        'feature': features_disponibles,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=True)
    axes[1].barh(imp['feature'], imp['importance'], color='coral')
    axes[1].set_title('Feature Importance')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('notebooks/ml_classifier_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    joblib.dump(rf_clf, f'models/{classe_atc}/rf_classifier.joblib')
    print(f'  Modele sauvegarde : models/{classe_atc}/rf_classifier.joblib')
    return rf_clf, roc

def tune_classifier(df, roc_baseline, classe_atc="R06"):
    print("\n=== GRIDSEARCHCV - RF Classifier ===")

    features_disponibles = [f for f in FEATURES_CLF if f in df.columns]
    df_clf = df.dropna(subset=features_disponibles + ['target_rupture'])
    X = df_clf[features_disponibles]
    y = df_clf['target_rupture']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)  # 0.3 au lieu de 0.2

    # SMOTE : rééquilibrage sur le train uniquement
    sm = SMOTE(random_state=42, k_neighbors=min(3, y_train.value_counts().min() - 1))
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"  Apres SMOTE — train : {dict(pd.Series(y_train).value_counts())}")

    param_grid = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [3, 5, 7],       # valeurs basses pour limiter overfitting
        'min_samples_split': [2, 5],
    }

    rf_base = RandomForestClassifier(
        class_weight='balanced', random_state=42)

    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )

    print("  Lancement GridSearchCV (18 combinaisons x 5 folds)...")
    grid_search.fit(X_train, y_train)

    print(f"  Meilleurs parametres : {grid_search.best_params_}")
    roc_cv = grid_search.best_score_
    mlflow.log_metric("gridsearch_best_cv", roc_cv)
    print(f"  Meilleur ROC-AUC CV  : {roc_cv:.3f}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, zero_division=0))
    
    metrics_tuned = {
        "tuned_test_accuracy": accuracy_score(y_test, y_pred),
        "tuned_test_balanced_accuracy": balanced_accuracy_score(
            y_test,
            y_pred,
        ),
        
        "tuned_test_precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
            
            
        "tuned_test_recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
            ),

        "tuned_test_f1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
            ),
            
            "tuned_test_pr_auc": average_precision_score(
                y_test,
                y_prob,
            ),
            
            "tuned_test_roc_auc": roc_auc_score(
                y_test,
                y_prob,
            ),
        }
    
    mlflow.log_metrics(metrics_tuned)
    
    roc_test = metrics_tuned["tuned_test_roc_auc"]
    
    print(f"  Accuracy          : {metrics_tuned['tuned_test_accuracy']:.3f}")
    print(f"  Balanced accuracy : {metrics_tuned['tuned_test_balanced_accuracy']:.3f}")
    print(f"  Precision         : {metrics_tuned['tuned_test_precision']:.3f}")
    print(f"  Recall            : {metrics_tuned['tuned_test_recall']:.3f}")
    print(f"  F1-score          : {metrics_tuned['tuned_test_f1']:.3f}")
    print(f"  PR-AUC            : {metrics_tuned['tuned_test_pr_auc']:.3f}")
    print(f"  ROC-AUC           : {metrics_tuned['tuned_test_roc_auc']:.3f}")

    # Comparaison sur ROC-AUC CV (plus fiable que test set sur 60 lignes)
    if roc_cv > roc_baseline:
        selected_model = best_model
        selected_model_name = "random_forest_tuned"
        joblib.dump(best_model, f'models/{classe_atc}/rf_classifier.joblib')
        print(f'  Modele ameliore (CV {roc_baseline:.3f} -> {roc_cv:.3f}), sauvegarde')
    else:
        selected_model = joblib.load(f'models/{classe_atc}/rf_classifier.joblib')
        selected_model_name = "random_forest_initial"
        print(f"  Pas d'amelioration (CV {roc_cv:.3f} <= {roc_baseline:.3f}), ancien modele conserve")

    mlflow.log_param("best_model", selected_model_name)
    mlflow.log_params({
        "grid_best_n_estimators": grid_search.best_params_["n_estimators"],
        "grid_best_max_depth": grid_search.best_params_["max_depth"],
        "grid_best_min_samples_split": grid_search.best_params_["min_samples_split"],
    })
    return selected_model

import shap

def explain_classifier(df, classe_atc="R06"):
    print("\n=== SHAP - Explication RF Classifier ===")

    features_disponibles = [f for f in FEATURES_CLF if f in df.columns]
    df_clf = df.dropna(subset=features_disponibles + ['target_rupture'])
    X = df_clf[features_disponibles]

    rf_clf = joblib.load(f'models/{classe_atc}/rf_classifier.joblib')

    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_rupture = shap_values[:, :, 1]
    elif isinstance(shap_values, list):
        shap_rupture = shap_values[1]
    else:
        shap_rupture = shap_values

    print("  SHAP values calculees")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_rupture, X, plot_type='bar', show=False)
    plt.title('SHAP - Importance moyenne des features (rupture)')
    plt.tight_layout()
    plt.savefig(f'notebooks/shap_importance_{classe_atc}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Graphique sauvegarde : notebooks/shap_importance_{classe_atc}.png')

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_rupture, X, show=False)
    plt.title('SHAP - Impact des features sur la prediction rupture')
    plt.tight_layout()
    plt.savefig(f'notebooks/shap_beeswarm_{classe_atc}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Graphique sauvegarde : notebooks/shap_beeswarm_{classe_atc}.png')

def train_model(classe_atc='R06'):
    print(f"Chargement Gold — classe {classe_atc}...")

    # Gold par classe (target_rupture propre à chaque classe)
    gold_classe = Path(
        f"data/gold/gold_ml_{classe_atc}.csv"
    )

    # Fusion avec features_advanced pour récupérer ruptures_lag1, gram_lag_mois, cumul_thermique
    gold_advanced = Path(
        "data/gold/gold_ml_advanced.csv"
    )

    if not os.path.exists(gold_classe):
        raise FileNotFoundError(f"Gold introuvable : {gold_classe} — lance build_gold.py --classe {classe_atc}")

    df = pd.read_csv(gold_classe)

    if os.path.exists(gold_advanced):
        df_adv = pd.read_csv(gold_advanced)
        cols_adv = ['annee_mois_str', 'ruptures_lag1', 'gram_lag_mois', 'cumul_thermique']
        cols_adv = [c for c in cols_adv if c in df_adv.columns]
        df = df.merge(df_adv[cols_adv], on='annee_mois_str', how='left')
        print(f"  Features advanced fusionnees")

    print(f"  Fichier Gold : {gold_classe}")
    print(f"  Shape : {df.shape}")
    print(f"  Target : {df['target_rupture'].value_counts().to_dict()}")

    os.makedirs(f'models/{classe_atc}', exist_ok=True)

    with mlflow.start_run(run_name=f"Pipeline_{classe_atc}"):

         # Version des fichiers de données utilisés
        mlflow.log_param(
            "dataset_classe_filename",
            gold_classe.name,
        )

        mlflow.log_param(
            "dataset_classe_sha256",
            calculate_file_sha256(gold_classe),
        )

        mlflow.log_param(
            "dataset_advanced_filename",
            gold_advanced.name,
        )

        mlflow.log_param(
            "dataset_advanced_sha256",
            calculate_file_sha256(gold_advanced),
        )

        # Informations complémentaires sur les données
        mlflow.log_param(
            "dataset_nombre_lignes",
            len(df),
        )

        mlflow.log_param(
            "dataset_nombre_colonnes",
            len(df.columns),
        )

        mlflow.log_params({
            "classe_atc": classe_atc,
            "nb_lignes": len(df),
            "nb_colonnes": len(df.columns),
            "test_size": 0.3,
            "random_state": 42,
            "rf_reg_n_estimators": 200,
            "rf_reg_max_depth": 10,
            "rf_clf_n_estimators": 200,
            "rf_clf_max_depth": 5,
            "smote_enabled": True,
            "gridsearch_cv": 5,
        })

        lr_baseline = train_baseline(df, classe_atc)
        rf_reg, df_reg = train_regressor(df, classe_atc)
        rf_clf, roc_baseline = train_classifier(df, classe_atc)
        selected_model = tune_classifier(df, roc_baseline, classe_atc)
        explain_classifier(df, classe_atc)

        features_baseline = [
            "gram_moy",
            "temp_moy",
            "precip",
            "mois",
            ]
        
        features_classifier = [
            feature
            for feature in FEATURES_CLF
            if feature in df.columns
            ]
        
        baseline_input_example = (
            df
            .dropna(subset=features_baseline)
            [features_baseline]
            .head(5)
            .astype("float64")
            )
        
        regressor_input_example = (
            df
            .dropna(subset=FEATURES_REG)
            [FEATURES_REG]
            .head(5)
            .astype("float64")
            )
        
        classifier_input_example = (
            df
            .dropna(subset=features_classifier)
            [features_classifier]
            .head(5)
            .astype("float64")
            )
        
        baseline_signature = infer_signature(
            baseline_input_example,
            lr_baseline.predict(baseline_input_example),
            )
        
        regressor_signature = infer_signature(
            regressor_input_example,
            rf_reg.predict(regressor_input_example),
            )
        
        
        initial_classifier_signature = infer_signature(
            classifier_input_example,
            rf_clf.predict(classifier_input_example),
            )
        
        selected_model_signature = infer_signature(
            classifier_input_example,
            selected_model.predict(classifier_input_example),
            )

        mlflow.sklearn.log_model(
            sk_model=lr_baseline,
            artifact_path="models/logistic_baseline",
            signature=baseline_signature,
            input_example=baseline_input_example,
            )
        
        mlflow.sklearn.log_model(
            sk_model=rf_reg,
            artifact_path="models/random_forest_regressor",
            signature=regressor_signature,
            input_example=regressor_input_example,
            )
        
        mlflow.sklearn.log_model(
            sk_model=rf_clf,
            artifact_path="models/random_forest_initial",
            signature=initial_classifier_signature,
            input_example=classifier_input_example,
            )
        
        registered_model_name = f"antihistaminiques_{classe_atc}_classifier"
        
        model_info = mlflow.sklearn.log_model(
            sk_model=selected_model,
            artifact_path="models/random_forest_selected",
            signature=selected_model_signature,
            input_example=classifier_input_example,
            registered_model_name=registered_model_name,
            )
        
        client = MlflowClient()
        
        versions = client.search_model_versions(
            f"name='{registered_model_name}'"
            )
        
        latest_version = max(
            versions,
            key=lambda version: int(version.version),
            )
        
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="champion",
            version=latest_version.version,
            )
        
        client.set_model_version_tag(
            name=registered_model_name,
            version=latest_version.version,
            key="classe_atc",
            value=classe_atc,
            )
        
        client.set_model_version_tag(
            name=registered_model_name,
            version=latest_version.version,
            key="model_role",
            value="rupture_classifier",
            )
        
        print(
            f"Alias 'champion' -> version {latest_version.version}"
            )

        artifacts = [
            "notebooks/ml_regressor_results.png",
            "notebooks/ml_classifier_results.png",
            f"notebooks/shap_importance_{classe_atc}.png",
            f"notebooks/shap_beeswarm_{classe_atc}.png",
            ]
        
        for artifact in artifacts:
            if os.path.exists(artifact):
                mlflow.log_artifact(
                    artifact,
                    artifact_path="plots"
                    )

    print(f"\n=== PIPELINE ML TERMINE — {classe_atc} ===")
    print(f"  models/{classe_atc}/lr_baseline.joblib")
    print(f"  models/{classe_atc}/rf_regressor.joblib")
    print(f"  models/{classe_atc}/rf_classifier.joblib")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--classe', default='R06', help='Classe ATC (R06, R03, J01)')
    args = parser.parse_args()
    train_model(classe_atc=args.classe)

