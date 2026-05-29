# Detección de patrones temporales en entornos de predicción de accidentes aéreos

## Descripción

Este repositorio contiene la implementación desarrollada como parte del Trabajo de Fin de Grado **"Detección de patrones temporales en entornos de predicción de accidentes aéreos"**.

El sistema analiza datos de telemetría GPS de 31 ejemplares de buitre leonado (*Gyps fulvus*) procedentes de una investigación del Centro de Doñana (CSIC), aplicando redes LSTM y un modelo ARIMA para predecir la trayectoria de vuelo en términos de latitud, longitud y altura relativa.

## Estructura del repositorio

### Carpeta `Codigo`

Contiene los scripts principales utilizados durante el desarrollo:

* **code.py** — Pipeline completo de entrenamiento y evaluación del modelo LSTM.
* **exploratorio.py** — Análisis exploratorio de los datasets utilizados.
* **modelo_Arima.py** — Implementación del modelo ARIMA utilizado como baseline de comparación.
* **regresion_baseline.py** — Modelo de regresión utilizado como baseline adicional para la evaluación de resultados.

### Carpeta `Codigo-ipynb`

Contiene los cuadernos Jupyter utilizados durante el desarrollo y experimentación:

* Notebook de entrenamiento y evaluación LSTM .
* Notebook de modelo ARIMA.

## Preparación de los datos

Para ejecutar cualquiera de los scripts es necesario crear la siguiente estructura de directorios desde la raíz del proyecto:

```text
datos_buitre/
└── AviFauna_Muladares/
```

Dentro de la carpeta `AviFauna_Muladares` deben colocarse los archivos CSV correspondientes a los distintos ejemplares de buitre analizados:

```text
datos_buitre/
└── AviFauna_Muladares/
    ├── T1.csv
    ├── T2.csv
    ├── ...
    └── T31.csv
```

Los scripts leen directamente estos archivos para realizar el análisis exploratorio, el entrenamiento de modelos y la evaluación de resultados.

## Requisitos

Las dependencias pueden instalarse mediante:

```bash
pip install -r requirements.txt
```

## Autor

**Jorge Blanco Carrasco**
Universidad Politécnica de Madrid (UPM)
