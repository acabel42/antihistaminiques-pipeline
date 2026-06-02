import cdsapi
import xarray as xr
import pandas as pd
import os

os.makedirs('data/silver', exist_ok=True)

def fetch_cams_pollen(date_debut, date_fin, output_nc, output_csv):
    client = cdsapi.Client()
    os.makedirs('data/raw', exist_ok=True)
    print(f'Telechargement {date_debut} -> {date_fin}...')
    client.retrieve(
        'cams-europe-air-quality-forecasts',
        {
            'variable': ['alder_pollen','birch_pollen','grass_pollen',
                         'mugwort_pollen','olive_pollen','ragweed_pollen'],
            'model': 'ensemble',
            'level': '0',
            'date': [f'{date_debut}/{date_fin}'],
            'time': ['00:00'],
            'leadtime_hour': ['0'],
            'type': ['forecast'],
            'format': 'netcdf',
            'area': [51, -5, 42, 10]
        },
        output_nc)
    print(f'Taille : {round(os.path.getsize(output_nc)/1024/1024,1)} Mo')
    ds = xr.open_dataset(output_nc)
    df = ds.to_dataframe().reset_index()
    df.columns = [c.lower() for c in df.columns]
    df['source'] = 'CAMS'
    df.to_csv(output_csv, index=False)
    print(f'Sauvegarde : {output_csv} -- {df.shape}')
    return df

if __name__ == '__main__':
    for annee in ['2023','2024','2025','2026']:
        fetch_cams_pollen(
            f'{annee}-01-01', f'{annee}-12-31',
            f'data/raw/cams_pollen_{annee}_france.nc',
            f'data/raw/cams_pollen_{annee}_france.csv')
        
