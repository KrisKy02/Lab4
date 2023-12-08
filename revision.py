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

Estudiantes:

Kristel Herrera Rodríguez C13769
Oscar Porras Silesky C16042
Fabrizzio Herrera Calvo B83849
"""
# Importa las bibliotecas necesarias y todas las funciones del módulo proceso
from proceso import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
import scipy.stats as st
import seaborn as sns
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

# Carga el conjunto de datos de calidad del aire
data = pd.read_csv('seoul.csv')

# ----- Análisis con Función 'muestra' -----
# Extrae y muestra un subconjunto de datos para un contaminante y sensor específicos
muestra_ejemplo = muestra(data, 'o3', 113, 2020031400, 2020031723)
print("Primeras filas de la muestra extraída:\n", muestra_ejemplo.head())

# ----- Análisis con Función 'proceso' -----
# Extrae funciones muestra diarias y muestra las fechas correspondientes
proceso_ejemplo = proceso(data, 'o3', 113, 20200314, 20200317)
print("Fechas para las que se han extraído funciones muestra:", list(proceso_ejemplo.keys()))

# Genera gráficas 2D y 3D para la visualización de datos
lista_fechas = [['2020031400', '2020031423'], ['2020031500', '2020031523']]
grafica2d(data, 'o3', 113, lista_fechas, 'graficas2d')
grafica3d(data, 'o3', 113, '2020031400', '2020031723')

# Evalúa y compara distribuciones de probabilidad para el contaminante
distribucion(data, 'o3', 113)

# ----- Preparación de Datos para Análisis Estadístico -----
# Convierte la columna de fecha/hora al formato datetime y la establece como índice
data['dt'] = pd.to_datetime(data['dt'], format='%Y%m%d%H')
data.set_index('dt', inplace=True)

# ----- Análisis Estadístico -----
# Calcula y muestra la autocorrelación y autocovarianza para horas específicas
autocorr_value = autocorrelacion(data, 'o3', 9, 17)
print(f"Autocorrelación entre las horas 9 y 17: {autocorr_value}")

autocovarianza_val = autocovarianza(data, 'o3', 9, 17)
print(f"Autocovarianza entre las 9 y las 17: {autocovarianza_val}")

# Calcula y muestra la media temporal del contaminante en un intervalo específico
promedio = prom_temporal(data, 'o3', inicio='2020031400', fin='2020031723')
print(f"El promedio temporal es: {promedio}")

# Evalúa si la secuencia de datos es ergódica
resultado_ergodicidad = ergodicidad(data, 'o3')
print(f"La secuencia aleatoria M(t) es ergódica: {resultado_ergodicidad}")

# Prepara la muestra para evaluar la estacionariedad en sentido amplio
muestra_ejemplo['dt'] = pd.to_datetime(muestra_ejemplo['dt'], format='%Y%m%d%H')
muestra_ejemplo.set_index('dt', inplace=True)

# Evalúa si la muestra es estacionaria en sentido amplio
resultado_wss_muestra = wss(muestra_ejemplo, 'o3', 'dt')
print(f"La muestra es estacionaria en sentido amplio: {resultado_wss_muestra}")
