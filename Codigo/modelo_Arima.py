from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIGURACIÓN
BUITRE = "T10"
RUTA_CSV = f"datos_buitre/AviFauna_Muladares/{BUITRE}.csv"


# CARGA Y PREPROCESAMIENTO (idéntico al código LSTM)
df = pd.read_csv(RUTA_CSV, low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df['altura_relativa'] = (df['height-above-sea'] - df['height-terrain']).clip(lower=0)
df = df[['timestamp', 'location-lat', 'location-long', 'altura_relativa']]
df = df.set_index('timestamp')
df = df[df['altura_relativa'] <= 3000]
df_resampled = df.resample('10min').mean().interpolate()

# DIVISIÓN (idéntica al código LSTM)
n = len(df_resampled)
train_end  = int(n * 0.70)
test_start = int(n * 0.85)
N_TEST = n - test_start
variables = ['location-lat', 'location-long', 'altura_relativa']
nombres   = ['Latitud', 'Longitud', 'Altura relativa']
unidades  = ['grados', 'grados', 'metros']

resultados_arima = {}
import time

for var, nombre, unidad in zip(variables, nombres, unidades):
    print(f"\n {nombre}")
    start = time.time()
    serie       = df_resampled[var].values
    serie_train = serie[:train_end]
    serie_test  = serie[test_start:test_start + N_TEST]

    # Ajuste inicial del modelo sobre datos de entrenamiento
    modelo    = ARIMA(serie_train, order=(5, 1, 0))
    resultado = modelo.fit()

    predicciones = []

    for i in range(N_TEST):
        if i % 25 == 0:
            print(f"  Paso {i}/{N_TEST}...")
        # Predice un paso sin reajustar desde cero
        pred = resultado.forecast(steps=1)[0]
        predicciones.append(pred)
        # Actualiza el modelo con el valor real sin reajustar completamente
        resultado = resultado.append([serie_test[i]], refit=False)
    
    end = time.time()
    print(f"  Tiempo: {(end-start)/60:.1f} minutos")

    predicciones = np.array(predicciones)

    mae  = mean_absolute_error(serie_test, predicciones)
    rmse = np.sqrt(mean_squared_error(serie_test, predicciones))

    print(f"  MAE:  {mae:.4f} {unidad}")
    print(f"  RMSE: {rmse:.4f} {unidad}")

    resultados_arima[nombre] = {
        'real':   serie_test,
        'pred':   predicciones,
        'mae':    mae,
        'rmse':   rmse,
        'unidad': unidad
    }

    plt.figure(figsize=(12, 5))
    plt.plot(serie_test,   label=f"Real {nombre}",  color='steelblue')
    plt.plot(predicciones, label=f"ARIMA {nombre}", color='darkorange')
    plt.title(f"Baseline ARIMA — {nombre} — {BUITRE}")
    plt.xlabel("Instantes de tiempo (cada 10 min)")
    plt.ylabel(f"{nombre} ({unidad})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Grafs/arima_{nombre.lower().replace(' ', '_')}_{BUITRE}.png")
    plt.show()
