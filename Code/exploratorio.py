import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

BUITRE= "T10"
RUTA_CSV = f"datos_buitre/AviFauna_Muladares/{BUITRE}.csv"

def filtrar_anomalos(df, altura_max=3000, verbose=True):
    df_filtrado = df[df['altura_relativa'] <= altura_max].copy()
    return df_filtrado


#1.CARGA
df = pd.read_csv(RUTA_CSV, low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df['altura_relativa'] = (df['height-above-sea'] - df['height-terrain']).clip(lower=0)
df = df[['timestamp', 'location-lat', 'location-long', 'altura_relativa']]
df = df.set_index('timestamp')
df = filtrar_anomalos(df, altura_max=3000)
df_resampled = df.resample('10min').mean().interpolate()

#2. INFORMACIÓN GENERAL
print("=== INFORMACIÓN GENERAL ===")
print(f"Periodo:       {df_resampled.index.min()} → {df_resampled.index.max()}")
print(f"Nº registros:  {len(df_resampled)}")
print(f"Duración:      {(df_resampled.index.max() - df_resampled.index.min()).days} días")

#3. SERIE TEMPORAL COMPLETA
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(df_resampled.index, df_resampled['location-lat'], color='steelblue', linewidth=0.5)
axes[0].set_ylabel('Latitud (°)')
axes[0].set_title(f'Serie temporal completa — Buitre {BUITRE}')

axes[1].plot(df_resampled.index, df_resampled['location-long'], color='darkorange', linewidth=0.5)
axes[1].set_ylabel('Longitud (°)')

axes[2].plot(df_resampled.index, df_resampled['altura_relativa'], color='seagreen', linewidth=0.5)
axes[2].set_ylabel('Altura relativa (m)')
axes[2].set_xlabel('Fecha')

plt.tight_layout()
plt.savefig("Grafs/serie_temporal.png", dpi=150)

#4. REGISTROS POR MES (densidad temporal)
registros_mes = df_resampled.resample('ME').count()['location-lat']

plt.figure(figsize=(14, 4))
plt.bar(registros_mes.index, registros_mes.values, color='steelblue', width=20)
plt.title(f'Número de registros por mes — Buitre {BUITRE}')
plt.xlabel('Fecha')
plt.ylabel('Nº registros')
plt.tight_layout()
plt.savefig("Grafs/registros_mes.png", dpi=150)

#5. ACTIVIDAD POR HORA DEL DÍA
df_resampled['hora'] = df_resampled.index.hour
actividad_hora = df_resampled.groupby('hora')['altura_relativa'].mean()

plt.figure(figsize=(10, 4))
plt.plot(actividad_hora.index, actividad_hora.values, marker='o', color='seagreen')
plt.fill_between(actividad_hora.index, actividad_hora.values, alpha=0.2, color='seagreen')
plt.title(f'Altura media de vuelo por hora del día — Buitre {BUITRE}')
plt.xlabel('Hora del día')
plt.ylabel('Altura relativa media (m)')
plt.xticks(range(0, 24))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("Grafs/actividad_hora.png", dpi=150)

#6. ACTIVIDAD POR MES (estacionalidad)
df_resampled['mes'] = df_resampled.index.month
altura_mes = df_resampled.groupby('mes')['altura_relativa'].mean()
meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

plt.figure(figsize=(10, 4))
plt.bar(altura_mes.index, altura_mes.values, color='darkorange', edgecolor='white')
plt.xticks(range(1, 13), meses)
plt.title(f'Altura media de vuelo por mes — Buitre {BUITRE}')
plt.xlabel('Mes')
plt.ylabel('Altura relativa media (m)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("Grafs/estacionalidad.png", dpi=150)

#7. EVOLUCIÓN DE LATITUD POR MES
df_resampled['mes'] = df_resampled.index.month
lat_mes = df_resampled.groupby('mes')['location-lat'].mean()

plt.figure(figsize=(10, 4))
plt.plot(lat_mes.index, lat_mes.values, marker='o', color='steelblue')
plt.fill_between(lat_mes.index, lat_mes.values, alpha=0.2, color='steelblue')
plt.xticks(range(1, 13), meses)
plt.title(f'Latitud media por mes — Buitre {BUITRE}')
plt.xlabel('Mes')
plt.ylabel('Latitud media (°)')
plt.ylim(35, 40)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("Grafs/latitud_mes.png", dpi=150)

#8. LONGITUD MEDIA POR MES
lon_mes = df_resampled.groupby('mes')['location-long'].mean()

plt.figure(figsize=(10, 4))
plt.plot(lon_mes.index, lon_mes.values, marker='o', color='darkorange')
plt.fill_between(lon_mes.index, lon_mes.values, alpha=0.2, color='darkorange')
plt.xticks(range(1, 13), meses)
plt.title(f'Longitud media por mes — Buitre {BUITRE}')
plt.xlabel('Mes')
plt.ylabel('Longitud media (°)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"Grafs/longitud_mes.png", dpi=150)

print("\nTodas las gráficas guardadas correctamente.")