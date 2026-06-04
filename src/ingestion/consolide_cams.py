import xarray as xr
import pandas as pd
import os
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('data/silver', exist_ok=True)

DATE = datetime.now().strftime('%Y%m%d')

ANNEES = {
    '2023': '2023-05-27',
    '2024': '2024-01-01',
    '2025': '2025-01-01',
    '2026': '2026-01-01',
}

def nc_to_df(annee, date_debut):
    path = f'data/raw/cams_pollen_{annee}_france.nc'
    ds   = xr.open_dataset(path)
    df   = ds.to_dataframe().reset_index()
    df['date'] = pd.to_datetime(date_debut) + pd.to_timedelta(df['time'])
    df['longitude'] = df['longitude'].apply(lambda x: x-360 if x>180 else x)
    df = df.rename(columns={
        'apg_conc':  'aulne_conc',
        'bpg_conc':  'bouleau_conc',
        'gpg_conc':  'graminees_conc',
        'mpg_conc':  'armoise_conc',
        'opg_conc':  'olivier_conc',
        'rwpg_conc': 'ambroisie_conc'})
    df = df[['date','longitude','latitude',
             'aulne_conc','bouleau_conc','graminees_conc',
             'armoise_conc','olivier_conc','ambroisie_conc']]
    df['annee']     = int(annee)
    df['source']    = 'CAMS'
    df['loaded_at'] = DATE
    print(f'  {annee} : {df.shape} — {df.date.min().date()} -> {df.date.max().date()}')
    return df

if __name__ == '__main__':
    print('Consolidation CAMS pollen 2023-2026...')
    dfs = [nc_to_df(a, d) for a, d in ANNEES.items()]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=['graminees_conc'], how='all')

    # Sauvegarde horodatee + fichier courant
    combined.to_csv(f'data/silver/J0_silver_cams_pollen_2023_2026_{DATE}.csv', index=False)
    combined.to_csv('data/silver/J0_silver_cams_pollen_2023_2026.csv', index=False)
    print(f'Shape : {combined.shape} — Date : {DATE}')
    print(f'Periode : {combined.date.min().date()} -> {combined.date.max().date()}')