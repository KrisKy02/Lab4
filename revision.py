"""
Análisis de Datos Ambientales Usando `proceso.py`.

Este script utiliza el módulo `proceso.py` para realizar una serie de análisis
estadísticos y visualizaciones de datos ambientales. El script demuestra cómo
utilizar cada una de las funciones proporcionadas por el módulo, incluyendo
la generación de muestras, análisis de proceso, visualizaciones en 2D y 3D,
y cálculos de autocorrelación, autocovarianza, media temporal y ergodicidad.

El script se estructura de la siguiente manera:
- Carga y preparación de los datos ambientales.
- Ejemplo de uso de cada función del módulo `proceso`.
- Análisis estadístico de los datos utilizando las funciones del módulo.
"""
import os
import pandas as pd
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial.polynomial import Polynomial
from proceso import *

# ----- Carga y Preparación de Datos -----
data = pd.read_csv('seoul.csv')
data['dt'] = pd.to_datetime(data['dt'], format='%Y%m%d%H')
data.set_index('dt', inplace=True)

# ----- Ejemplos de Uso de Funciones -----
# Función muestra
muestra_ejemplo = muestra(data, 'o3', 113, 2020031400, 2020031723)
print("Ejemplo de función muestra:\n", muestra_ejemplo.head())

# Función proceso
proceso_ejemplo = proceso(data, 'o3', 113, 20200314, 20200317)
print("Claves del proceso ejemplo:", list(proceso_ejemplo.keys()))

# Función grafica2d
lista_fechas = [['2020031400', '2020031423'], ['2020031500', '2020031523']]
grafica2d(data, 'o3', 113, lista_fechas, 'graficas2d')

# Función grafica3d
grafica3d(data, 'o3', 113, '2020031400', '2020031723')

# Función distribucion
distribucion(data, 'o3', 113)

# ----- Análisis Estadístico -----
# Autocorrelación y Autocovarianza
autocorr_value = autocorrelacion(data, 'o3', 9, 17)
print(f"Autocorrelación entre las horas 9 y 17: {autocorr_value}")

autocovarianza_val = autocovarianza(data, 'o3', 9, 17)
print(f"Autocovarianza entre las 9 y las 17: {autocovarianza_val}")

# Media Temporal y Ergodicidad
promedio = prom_temporal(data, 'o3', inicio='2020031400', fin='2020031723')
print(f"El promedio temporal es: {promedio}")

resultado_ergodicidad = ergodicidad(data, 'o3')
print(f"La secuencia aleatoria M(t) es ergódica: {resultado_ergodicidad}")

# Estacionariedad en Sentido Amplio
resultado_wss_muestra = wss(data, 'o3', 'dt')
print(f"La muestra es estacionaria en sentido amplio: {resultado_wss_muestra}")
