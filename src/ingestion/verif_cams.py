import xarray as xr
import os

for annee in ['2023','2024','2025','2026']:
    path = f'data/raw/cams_pollen_{annee}_france.nc'
    if os.path.exists(path):
        ds = xr.open_dataset(path)
        print(f'\n=== {annee} ===')
        print(f'Dimensions : {dict(ds.sizes)}')
        print(f'FORECAST   : {ds.attrs.get("FORECAST", "?")}')
        print(f'Taille     : {round(os.path.getsize(path)/1024/1024,1)} Mo')
    else:
        print(f'\n=== {annee} === fichier non disponible')