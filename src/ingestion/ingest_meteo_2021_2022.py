import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)
os.makedirs('data/silver', exist_ok=True)

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

stations = {
    'Ile-de-France':           (48.85, 2.35),
    'Centre-Val-de-Loire':     (47.75, 1.67),
    'Bourgogne-Franche-Comte': (47.28, 5.00),
    'Normandie':               (49.18, 0.37),
    'Hauts-de-France':         (50.45, 2.97),
    'Grand Est':               (48.57, 7.75),
    'Pays-de-la-Loire':        (47.47, -0.55),
    'Bretagne':                (48.11, -1.68),
    'Nouvelle-Aquitaine':      (44.84, -0.58),
    'Occitanie':               (43.60, 1.44),
    'Auvergne-Rhone-Alpes':    (45.75, 4.85),
    'PACA':                    (43.30, 5.37),
    'Corse':                   (42.03, 9.01),
}

all_dfs = []
for region, (lat, lon) in stations.items():
    print(f"  {region}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2021-01-01",
        "end_date": "2026-05-31",
        "daily": ["temperature_2m_max","temperature_2m_min",
                  "temperature_2m_mean","precipitation_sum",
                  "wind_speed_10m_max"],
        "timezone": "Europe/Paris"
    }
    responses = openmeteo.weather_api(url, params=params)
    r = responses[0]
    daily = r.Daily()
    df = pd.DataFrame({
        "time": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temperature_2m_max":  daily.Variables(0).ValuesAsNumpy(),
        "temperature_2m_min":  daily.Variables(1).ValuesAsNumpy(),
        "temperature_2m_mean": daily.Variables(2).ValuesAsNumpy(),
        "precipitation_sum":   daily.Variables(3).ValuesAsNumpy(),
        "wind_speed_10m_max":  daily.Variables(4).ValuesAsNumpy(),
    })
    df['time']   = df['time'].dt.tz_localize(None)
    df['region'] = region
    df['aasqa']  = 0
    df['source'] = 'Open-Meteo'
    all_dfs.append(df)
    print(f"    OK {len(df)} lignes")

meteo = pd.concat(all_dfs, ignore_index=True)
meteo.to_csv('data/silver/J0_silver_meteo_openmeteo_2021_2026.csv', index=False)
print(f"\nSauvegarde : {meteo.shape}")