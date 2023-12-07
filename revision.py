from proceso import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

# Carga el DataFrame
data = pd.read_csv('seoul.csv')

# Ejemplo de uso de la función muestra:
muestra_ejemplo = muestra(data, 'o3', 113, 2020031400, 2020031723)
print(muestra_ejemplo.head())

# Ejemplo de uso de la función proceso:
proceso_ejemplo = proceso(data, 'o3', 113, 20200314, 20200317)
# Imprimir los días para los que se han extraído funciones muestra.
print(list(proceso_ejemplo.keys()))
lista_fechas = [['2020031400', '2020031423'], ['2020031500', '2020031523']]
grafica2d(data, 'o3', 113, lista_fechas, 'graficas2d')
grafica3d(data, 'o3', 113, '2020031400', '2020031723')
# Ejemplo de uso de la función:
distribucion(data, 'o3', 113)

# Prepara el DataFrame
data['dt'] = pd.to_datetime(data['dt'], format='%Y%m%d%H')
data.set_index('dt', inplace=True)

# Ahora puedes llamar a las funciones
autocorr_value = autocorrelacion(data, 'o3', 9, 17)
print(f"Autocorrelación entre las horas 9 y 17: {autocorr_value}")

autocovarianza_val = autocovarianza(data, 'o3', 9, 17)
print(f"Autocovarianza entre las 9 y las 17: {autocovarianza_val}")


promedio = prom_temporal(data, 'o3', inicio='2020031400', fin='2020031723')
print(f"El promedio temporal es: {promedio}")
resultado_ergodicidad = ergodicidad(data, 'o3')
print(f"La secuencia aleatoria M(t) es ergódica: {resultado_ergodicidad}")
# Prepara la muestra para wss
muestra_ejemplo['dt'] = pd.to_datetime(muestra_ejemplo['dt'],
                                       format='%Y%m%d%H')
muestra_ejemplo.set_index('dt', inplace=True)

# Llama a la función wss para la muestra
resultado_wss_muestra = wss(muestra_ejemplo, 'o3', 'dt')
print(f"La muestra es estacionaria en sentido amplio: {resultado_wss_muestra}")
