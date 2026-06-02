import pandas as pd

m1 = pd.read_csv('data/silver/J0_silver_meteo_2021_2022.csv')
m2 = pd.read_csv('data/silver/J0_silver_meteo_2023_2026.csv')
m2 = m2.rename(columns={'time':'time'})

combined = pd.concat([m1, m2], ignore_index=True)
combined = combined.sort_values(['region','time']).reset_index(drop=True)
combined.to_csv('data/silver/J0_silver_meteo_2021_2026.csv', index=False)
print(f"Shape : {combined.shape}")