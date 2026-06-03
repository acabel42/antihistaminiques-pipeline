import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (r2_score, mean_squared_error,
                             classification_report, confusion_matrix, roc_auc_score)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('models', exist_ok=True)

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
    'ruptures_lag1', 'gram_lag_mois', 'cumul_thermique'
]

def train_baseline(df):
    print("\n=== BASELINE — Regression Logistique ===")
    features_base = ['gram_moy', 'temp_moy', 'precip', 'mois']
    df_b = df.dropna(subset=features_base + ['target_rupture'])
    X = df_b[features_base]
    y = df_b['target_rupture']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"  ROC-AUC Baseline : {roc_auc_score(y_test, y_prob):.3f}")

    joblib.dump(lr, 'models/lr_baseline.joblib')
    print("  Modele sauvegarde : models/lr_baseline.joblib")
    return lr

def train_regressor(df):
    print("\n=== MODELE 1 — Regression gram_moy mois suivant ===")
    df = df.copy().sort_values('annee_mois_str').reset_index(drop=True)
    df['gram_moy_next'] = df['gram_moy'].shift(-1)
    df = df.dropna(subset=['gram_moy_next'] + FEATURES_REG)

    X = df[FEATURES_REG]
    y = df['gram_moy_next']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_reg.fit(X_train, y_train)

    y_pred = rf_reg.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  R²   : {r2:.3f}")
    print(f"  RMSE : {rmse:.3f} grains/m3")

    cv = cross_val_score(rf_reg, X, y, cv=5, scoring='r2')
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

    joblib.dump(rf_reg, 'models/rf_regressor.joblib')
    print("  Modele sauvegarde : models/rf_regressor.joblib")
    return rf_reg, df

def train_classifier(df):
    print("\n=== MODELE 2 — Classification rupture/tension R06 ===")

    features_disponibles = [f for f in FEATURES_CLF if f in df.columns]

    df_clf = df.dropna(subset=features_disponibles + ['target_rupture'])
    X = df_clf[features_disponibles]
    y = df_clf['target_rupture']

    print(f"  Target : {y.value_counts().to_dict()}")
    print(f"  Features utilisees : {len(features_disponibles)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=42)
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)
    y_prob = rf_clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"  ROC-AUC : {roc_auc_score(y_test, y_prob):.3f}")

    cv = cross_val_score(rf_clf, X, y, cv=5, scoring='f1_weighted')
    print(f"  F1 CV (5-fold) : {cv.mean():.3f} +/- {cv.std():.3f}")

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

    joblib.dump(rf_clf, 'models/rf_classifier.joblib')
    print("  Modele sauvegarde : models/rf_classifier.joblib")
    return rf_clf

def train_model():
    print("Chargement Gold...")
    gold_path = 'data/gold/gold_ml_advanced.csv' if os.path.exists('data/gold/gold_ml_advanced.csv') else 'data/gold/gold_ml.csv'
    df = pd.read_csv(gold_path)
    print(f"  Fichier Gold : {gold_path}")
    print(f"  Shape : {df.shape}")

    lr_baseline = train_baseline(df)
    rf_reg, df_reg = train_regressor(df)
    rf_clf = train_classifier(df)

    print("\n=== PIPELINE ML TERMINE ===")
    print("  models/lr_baseline.joblib   — regression logistique baseline")
    print("  models/rf_regressor.joblib  — prediction graminees mois suivant")
    print("  models/rf_classifier.joblib — detection rupture/tension R06")

if __name__ == '__main__':
    train_model()