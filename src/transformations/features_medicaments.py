import pandas as pd
from sqlalchemy import create_engine

ENGINE = create_engine('postgresql://pipeline:pipeline2026@localhost:5432/antihistaminiques')

def build_features_medicaments():
    med = pd.read_csv('data/silver/J0_silver_medicaments.csv')
    rup = pd.read_csv('data/silver/J0_silver_ruptures.csv')
    bdpm = pd.read_csv('data/silver/J0_silver_bdpm.csv')

    # Flag MITM — médicaments d'intérêt thérapeutique majeur
    mitm_cis = bdpm[bdpm['est_antihistaminique']==True]['cis'].astype(str).tolist()
    med['flag_mitm'] = med['cis'].astype(str).isin(mitm_cis).astype(int)

    # Ratio ruptures par molécule
    rup_r06a = rup[rup['est_antihistaminique']==True]
    ratio = rup_r06a.groupby('molecule').agg(
        nb_ruptures=('classification', lambda x: (x=='rupture').sum()),
        nb_total=('classification', 'count')
    ).reset_index()
    ratio['ratio_rupture'] = ratio['nb_ruptures'] / ratio['nb_total']
    med = med.merge(ratio[['molecule','ratio_rupture','nb_ruptures','nb_total']],
                    on='molecule', how='left')
    med['ratio_rupture'] = med['ratio_rupture'].fillna(0)

    # Charger dans PostgreSQL
    med.to_sql('medicaments_gold', ENGINE, if_exists='replace', index=False)
    print(f'medicaments_gold : {med.shape}')
    return med

if __name__ == '__main__':
    build_features_medicaments()
