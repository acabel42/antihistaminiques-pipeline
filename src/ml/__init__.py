import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

os.makedirs('models', exist_ok=True)

def train_model():
    print("Chargement Gold...")
    df = pd.read_csv('data/gold/gold_ml.csv')
    df = df.dropna()

    print(f"  Shape : {df.shape}")
    print(f"  Target distribution : {df['target_rupture'].value_counts().to_dict()}")

    features = [
        'gram_moy', 'gram_max', 'gram_roll7', 'nb_jours_pic',
        'temp_moy', 'precip', 'mois', 'saison_allergies',
        'boites_total'
    ]

    X = df[features]
    y = df['target_rupture']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Modèle
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=3,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Evaluation
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\n=== RESULTATS ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.3f}")

    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1_weighted')
    print(f"F1 CV (5-fold) : {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Graphiques
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Random Forest Classifier — Ruptures R06A', fontsize=14, fontweight='bold')

    # Matrice de confusion
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Matrice de confusion')
    axes[0].set_xlabel('Predit')
    axes[0].set_ylabel('Reel')

    # Feature importance
    imp = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    axes[1].barh(imp['feature'], imp['importance'], color='steelblue')
    axes[1].set_title('Feature Importance')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('notebooks/ml_baseline_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Sauvegarder
    joblib.dump(rf, 'models/rf_baseline.joblib')
    print("\nModele sauvegarde : models/rf_baseline.joblib")
    return rf

if __name__ == '__main__':
    os.chdir('/Users/nellyta/Jedha')
    train_model()
    