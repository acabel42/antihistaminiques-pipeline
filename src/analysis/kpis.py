import pandas as pd
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)

def compute_kpis():
    print("=== KPIs PROJET ANTIHISTAMINIQUES R06 ===")
    df = pd.read_csv('data/gold/gold_ml.csv')

    # KPI 1 — Taux de rupture global
    taux = df['target_rupture'].mean() * 100
    print(f"\n1. Taux de rupture global R06 : {taux:.1f}%")
    print(f"   {df['target_rupture'].sum()} mois en tension sur {len(df)} mois (2021-2026)")

    # KPI 2 — Mois le plus a risque
    mois_risque = df.groupby('mois')['target_rupture'].mean()
    mois_noms = {1:'Janvier',2:'Fevrier',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                 7:'Juillet',8:'Aout',9:'Septembre',10:'Octobre',11:'Novembre',12:'Decembre'}
    print(f"\n2. Mois le plus a risque : {mois_noms[mois_risque.idxmax()]} ({mois_risque.max()*100:.1f}% de rupture)")

    # KPI 3 — Seuil critique graminees
    seuil = df.groupby('target_rupture')['gram_moy'].mean()
    print(f"\n3. Graminees moy sans rupture : {seuil[0]:.1f} g/m3")
    print(f"   Graminees moy avec rupture : {seuil[1]:.1f} g/m3")

    # KPI 4 — Correlations
    print(f"\n4. Correlations avec rupture R06 :")
    print(f"   graminees    : {df['gram_moy'].corr(df['target_rupture']):.3f}")
    print(f"   bouleau      : {df['bouleau_moy'].corr(df['target_rupture']):.3f}")
    print(f"   ambroisie    : {df['ambroisie_moy'].corr(df['target_rupture']):.3f} <- signal principal")
    print(f"   temp_moy     : {df['temp_moy'].corr(df['target_rupture']):.3f}")
    print(f"   precip       : {df['precip'].corr(df['target_rupture']):.3f}")
    print(f"   boites_total : {df['boites_total'].corr(df['target_rupture']):.3f}")

    # KPI 5 — Annee la plus touchee
    annee_risque = df.groupby('annee')['nb_ruptures'].sum()
    print(f"\n5. Annee la plus touchee : {annee_risque.idxmax()} ({annee_risque.max()} ruptures)")

    # KPI 6 — Taux par saison
    print(f"\n6. Taux de rupture par saison :")
    for s, label in {0:'Hiver/Automne', 1:'Printemps/Ete'}.items():
        t = df[df['saison_allergies']==s]['target_rupture'].mean()*100
        print(f"   {label} : {t:.1f}%")

    # KPI 7 — Performance modele ML
    print(f"\n7. Performance modele ML :")
    print(f"   Baseline LR   : ROC-AUC = 0.457")
    print(f"   RF Classifier : ROC-AUC = 0.771")
    print(f"   RF Regressor  : R2 = 0.513 | RMSE = 6.3 g/m3")

    print("\n=== FIN KPIs ===")

if __name__ == '__main__':
    compute_kpis()