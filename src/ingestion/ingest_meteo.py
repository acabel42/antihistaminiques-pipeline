import requests
import pandas as pd
import time

aasqa_regions = {
    "11": ("Île-de-France",           48.8566,  2.3522),
    "24": ("Centre-Val-de-Loire",     47.9029,  1.9093),
    "27": ("Bourgogne-Franche-Comté", 47.3220,  5.0415),
    "28": ("Normandie",               49.1829, -0.3707),
    "32": ("Hauts-de-France",         50.6292,  3.0573),
    "44": ("Grand Est",               48.5734,  7.7521),
    "52": ("Pays-de-la-Loire",        47.2184, -1.5536),
    "53": ("Bretagne",                48.1173, -1.6778),
    "75": ("Nouvelle-Aquitaine",      44.8378, -0.5792),
    "76": ("Occitanie",               43.6047,  1.4442),
    "84": ("Auvergne-Rhône-Alpes",    45.7640,  4.8357),
    "93": ("PACA",                    43.2965,  5.3698),
    "94": ("Corse",                   42.0396,  9.0129),
}

all_dfs = []

for aasqa, (region, lat, lon) in aasqa_regions.items():
    print(f"  {region}...")
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude":   lat,
        "longitude":  lon,
        "start_date": "2023-01-01",  
        "end_date":   "2026-05-31",  
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]),
        "timezone": "Europe/Paris"
    })
    df = pd.DataFrame(r.json()["daily"])
    df["aasqa"]  = aasqa
    df["region"] = region
    all_dfs.append(df)
    time.sleep(0.5)

meteo = pd.concat(all_dfs, ignore_index=True)
meteo.to_csv("J0_silver_meteo_2023_2026.csv", index=False)
print(f"\n✅ {meteo.shape[0]} lignes | {meteo['aasqa'].nunique()} régions")
print(f"   Dates : {meteo['time'].min()} → {meteo['time'].max()}")
print(meteo.head())