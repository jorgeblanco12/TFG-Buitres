from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
BUITRE = "T10"
RUTA_CSV = f"datos_buitre/AviFauna_Muladares/{BUITRE}.csv"

# ── CARGA Y PREPROCESAMIENTO (idéntico al código LSTM) ────────────────────────
df = pd.read_csv(RUTA_CSV, low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df['altura_relativa'] = (df['height-above-sea'] - df['height-terrain']).clip(lower=0)
df = df[['timestamp', 'location-lat', 'location-long', 'altura_relativa']]
df = df.set_index('timestamp')
df = df[df['altura_relativa'] <= 3000]
df_resampled = df.resample('10min').mean().interpolate()

# ── DIVISIÓN (idéntica al código LSTM) ────────────────────────────────────────
n = len(df_resampled)
train_end  = int(n * 0.70)
test_start = int(n * 0.85)

variables = ['location-lat', 'location-long', 'altura_relativa']
nombres   = ['Latitud', 'Longitud', 'Altura relativa']
unidades  = ['grados', 'grados', 'metros']

resultados_regresion = {}

for var, nombre, unidad in zip(variables, nombres, unidades):
    print(f"\n─── {nombre} ────────────────────────────")

    serie       = df_resampled[var].values
    serie_train = serie[:train_end]
    serie_test  = serie[test_start:]

    # Tiempo como variable independiente (índice numérico)
    t_train = np.arange(len(serie_train)).reshape(-1, 1)
    t_test  = np.arange(len(serie_train), len(serie_train) + len(serie_test)).reshape(-1, 1)

    # ── Regresión lineal ──────────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(t_train, serie_train)
    pred_lr = lr.predict(t_test)

    mae_lr  = mean_absolute_error(serie_test, pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(serie_test, pred_lr))
    print(f"  Lineal    → MAE: {mae_lr:.4f} {unidad} | RMSE: {rmse_lr:.4f}")

    # ── Regresión no lineal (grado 4 como en el paper) ────────────────────────
    poly = Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('lr',   LinearRegression())
    ])
    poly.fit(t_train, serie_train)
    pred_poly = poly.predict(t_test)

    mae_poly  = mean_absolute_error(serie_test, pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(serie_test, pred_poly))
    print(f"  No lineal → MAE: {mae_poly:.4f} {unidad} | RMSE: {rmse_poly:.4f}")

    resultados_regresion[nombre] = {
        'real':      serie_test,
        'pred_lr':   pred_lr,
        'pred_poly': pred_poly,
        'mae_lr':    mae_lr,   'rmse_lr':   rmse_lr,
        'mae_poly':  mae_poly, 'rmse_poly': rmse_poly,
        'unidad':    unidad
    }

    # ── Gráficas ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(serie_test,  label=f"Real {nombre}",           color='steelblue')
    plt.plot(pred_lr,     label=f"Reg. lineal {nombre}",    color='darkorange')
    plt.plot(pred_poly,   label=f"Reg. no lineal {nombre}", color='seagreen')
    plt.title(f"Regresión lineal y no lineal — {nombre} — {BUITRE}")
    plt.xlabel("Instantes de tiempo (cada 10 min)")
    plt.ylabel(f"{nombre} ({unidad})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Grafs/regresion_{nombre.lower().replace(' ', '_')}_{BUITRE}.png")
    plt.show()

# ── TABLA COMPARATIVA COMPLETA ────────────────────────────────────────────────
lstm = {
    'Latitud':         {'mae': 0.0011, 'rmse': 0.0024},
    'Longitud':        {'mae': 0.0031, 'rmse': 0.0054},
    'Altura relativa': {'mae': 19.13,  'rmse': 25.90},
}

arima = {
    'Latitud':         {'mae': 0.0008, 'rmse': 0.0069},
    'Longitud':        {'mae': 0.0009, 'rmse': 0.0044},
    'Altura relativa': {'mae': 6.72,   'rmse': 17.01},
}

print("\n─── COMPARATIVA COMPLETA ─────────────────────────────────────────────────────")
print(f"{'Variable':<20} {'LSTM MAE':>12} {'ARIMA MAE':>12} {'Lin. MAE':>12} {'NoLin. MAE':>12}")
print("─" * 70)
for nombre in lstm:
    if nombre in resultados_regresion:
        print(f"{nombre:<20} "
              f"{lstm[nombre]['mae']:>12.4f} "
              f"{arima[nombre]['mae']:>12.4f} "
              f"{resultados_regresion[nombre]['mae_lr']:>12.4f} "
              f"{resultados_regresion[nombre]['mae_poly']:>12.4f}")