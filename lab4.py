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

def muestra(data, variable, loc, inicio, fin):
    """
    Extrae una función muestra m(t) del proceso aleatorio M(t) para un contaminante específico,
    sensor y rango de tiempo.

    :param data: DataFrame con las mediciones de contaminantes.
    :param variable: Contaminante ambiental a considerar (e.g., 'o3').
    :param loc: Identificador del sensor.
    :param inicio: Año, mes, día y hora de inicio en formato AAAAMMDDHH.
    :param fin: Año, mes, día y hora de final en formato AAAAMMDDHH.
    :return: DataFrame filtrado según los parámetros especificados.
    """
    # Filtrando por ubicación del sensor y rango de tiempo
    filtered_data = data[(data['loc'] == loc) & (data['dt'] >= inicio) & (data['dt'] <= fin)]
    
    # Seleccionando solo la columna del contaminante especificado
    return filtered_data[['dt', 'loc', variable]]


def proceso(data, variable, loc, inicio, fin):
    """
    Devuelve el conjunto de funciones muestra que forman el proceso aleatorio M(t) para un contaminante específico,
    sensor y rango de tiempo, indexado por un intervalo diario.

    :param data: DataFrame con las mediciones de contaminantes.
    :param variable: Contaminante ambiental a considerar (e.g., 'o3').
    :param loc: Identificador del sensor.
    :param inicio: Año, mes y día de inicio en formato AAAAMMDD.
    :param fin: Año, mes y día de final en formato AAAAMMDD.
    :return: Diccionario de DataFrames, cada uno representando una función muestra para un día específico.
    """
    # Convertir las columnas de fecha a formato datetime para facilitar el filtrado
    data['dt'] = pd.to_datetime(data['dt'], format='%Y%m%d%H')

    # Filtrando por ubicación del sensor y rango de tiempo
    inicio_datetime = pd.to_datetime(str(inicio), format='%Y%m%d')
    fin_datetime = pd.to_datetime(str(fin), format='%Y%m%d')
    filtered_data = data[(data['loc'] == loc) & (data['dt'] >= inicio_datetime) & (data['dt'] < fin_datetime)]

    # Creando un diccionario para almacenar cada función muestra por día
    proceso_dict = {}
    for single_date in pd.date_range(start=inicio_datetime, end=fin_datetime, freq='D'):
        day_data = filtered_data[filtered_data['dt'].dt.date == single_date.date()][['dt', 'loc', variable]]
        if not day_data.empty:
            proceso_dict[single_date.date()] = day_data

    return proceso_dict


def distribucion(data, variable, loc):
    filtered_data = data[(data['loc'] == loc)][['dt', variable]]
    filtered_data['hour'] = filtered_data['dt'].dt.hour

    parametros_globales = {}
    histogramas = {}
    mejor_distribucion_global = {}

    # Función interna para comparar distribuciones
    def comparar_distribuciones(data):
        distribuciones = [st.norm, st.lognorm, st.expon, st.gamma, st.weibull_min, st.weibull_max]
        resultados = pd.DataFrame()

        for distribucion in distribuciones:
            try:
                parametros = distribucion.fit(data)
                log_likelihood = np.sum(np.log(distribucion.pdf(data, *parametros)))
                aic = 2 * len(parametros) - 2 * log_likelihood
                fila = pd.DataFrame({'Distribucion': [distribucion.name], 'AIC': [aic]})
                resultados = pd.concat([resultados, fila], ignore_index=True)
            except Exception:
                pass

        return resultados.sort_values(by='AIC').iloc[0]

    for hour in range(24):
        hourly_data = filtered_data[filtered_data['hour'] == hour][variable].dropna()

        # Crear y guardar histograma
        plt.figure(figsize=(6, 4))
        sns.histplot(hourly_data, kde=True)
        plt.title(f'Distribución de {variable} en la hora {hour} para el sensor {loc}')
        plt.xlabel(f'Valores de {variable}')
        plt.ylabel('Frecuencia')
        plt.savefig(f'histogramas_distribucion/histograma_{variable}_loc{loc}_hora{hour}.png')
        plt.close()

        if len(hourly_data) > 1:
            mejor_distribucion = comparar_distribuciones(hourly_data)
            distribucion_nombre = mejor_distribucion['Distribucion']
            mejor_distribucion_global[hour] = distribucion_nombre
            parametros = getattr(st, distribucion_nombre).fit(hourly_data)
            parametros_globales[hour] = parametros

    # Identificar la distribución más común
    distribucion_mas_comun = pd.Series(mejor_distribucion_global).mode()[0]

    # Preparar los datos para el ajuste polinomial
    horas_con_datos = [hour for hour in range(24) if mejor_distribucion_global[hour] == distribucion_mas_comun]
    parametros_seleccionados = [parametros_globales[hour] for hour in horas_con_datos]
    parametros_transpuestos = list(zip(*parametros_seleccionados))

    # Ajuste de funciones polinomiales
    polinomios = []
    for parametro in parametros_transpuestos:
        if len(horas_con_datos) == len(parametro):
            polinomio = Polynomial.fit(horas_con_datos, parametro, deg=5)
            polinomios.append(polinomio)
    # Impresión de resultados
    print(f"La distribución más común es: {distribucion_mas_comun}")
    print("Los parámetros correspondientes son:")
    for hour, parametros in parametros_globales.items():
        if mejor_distribucion_global[hour] == distribucion_mas_comun:
            print(f"Hora {hour}: {parametros}")
    print("Los polinomios serían:")
    for i, polinomio in enumerate(polinomios):
        print(f"Polinomio {i + 1}: {polinomio}")

    return {
        "distribucion_mas_comun": distribucion_mas_comun,
        "polinomios": polinomios
    }

def grafica2d(data, variable, loc, lista_fechas, carpeta='graficas2d'):
    """
    Grafica varias funciones muestra del proceso M(t) en una gráfica.

    :param data: DataFrame con las mediciones de contaminantes.
    :param variable: Contaminante ambiental a considerar (e.g., 'o3').
    :param loc: Identificador del sensor.
    :param lista_fechas: Lista de listas, donde cada sublista contiene fechas de inicio y fin ['AAAAMMDDHH', 'AAAAMMDDHH'].
    :param carpeta: Nombre del directorio donde se guardarán las gráficas.
    """
    # Crear el directorio si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    plt.figure(figsize=(10, 6))

    for fechas in lista_fechas:
        inicio, fin = fechas
        inicio_datetime = datetime.strptime(inicio, '%Y%m%d%H')
        fin_datetime = datetime.strptime(fin, '%Y%m%d%H')

        # Filtrar los datos
        filtered_data = data[(data['loc'] == loc) & 
                             (data['dt'] >= inicio_datetime) & 
                             (data['dt'] <= fin_datetime)]

        plt.plot(filtered_data['dt'], filtered_data[variable], label=f'{inicio} a {fin}')

    plt.xlabel('Fecha y Hora')
    plt.ylabel(f'Nivel de {variable} (Unidades)')
    plt.title(f'Niveles de {variable} en el sensor {loc}')
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica
    filename = f"{carpeta}/grafica_comparativa_{variable}_loc{loc}.png"
    plt.savefig(filename)
    plt.close()

def grafica3d(data, variable, loc, inicio, fin, carpeta='graficas3d'):
    """
    Genera una gráfica tridimensional con los modelos de las distribuciones de probabilidad de cada hora del día.

    :param data: DataFrame con las mediciones de contaminantes.
    :param variable: Contaminante ambiental a considerar (e.g., 'o3').
    :param loc: Identificador del sensor.
    :param inicio: Fecha y hora de inicio en formato 'AAAAMMDDHH'.
    :param fin: Fecha y hora de fin en formato 'AAAAMMDDHH'.
    :param carpeta: Carpeta donde se guardará la gráfica.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear la carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    # Asegurarse de que 'dt' es un datetime
    if not pd.api.types.is_datetime64_any_dtype(data['dt']):
        data['dt'] = pd.to_datetime(data['dt'])

    # Crear la columna 'hour' si no existe
    if 'hour' not in data.columns:
        data['hour'] = data['dt'].dt.hour
    # Generar y graficar las distribuciones para cada hora
    colors = plt.cm.jet(np.linspace(0, 1, 24))  # Utilizar un mapa de colores
    for i, hour in enumerate(range(24)):
        hourly_data = data[(data['loc'] == loc) & (data['hour'] == hour)][variable].dropna()
        if not hourly_data.empty:
            # Ajustar una distribución a los datos
            mu, sigma = st.norm.fit(hourly_data)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            y = st.norm.pdf(x, mu, sigma)
            # Cambiar eje X a la hora del día y centrar la distribución en la hora
            ax.plot(np.full_like(x, hour), x, y, color=colors[i])

    ax.set_xlabel('Hora del Día')
    ax.set_ylabel('Valor del Contaminante')
    ax.set_zlabel('Densidad de Probabilidad')
    ax.view_init(30, -60)  # Ajustar la perspectiva de la visualización

    # Guardar la gráfica
    filename = f"{carpeta}/grafica3d_{variable}_loc{loc}.png"
    plt.savefig(filename)
    plt.close(fig)
def autocorrelacion(data, variable, t1, t2):
    """
    Calcula la autocorrelación para dos horas específicas t1 y t2 para la secuencia aleatoria M(t).
    """
    # Convertir 'dt' a datetime si aún no lo es
    if not pd.api.types.is_datetime64_any_dtype(data['dt']):
        data['dt'] = pd.to_datetime(data['dt'], format='%Y%m%d%H')

    # Establecer 'dt' como índice si aún no lo es
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data = data.set_index('dt')

    # Filtrar los datos por las dos horas específicas
    data_t1 = data[data.index.hour == t1][variable]
    data_t2 = data[data.index.hour == t2][variable]

    # Agrupar por fecha y calcular la media
    data_t1_mean = data_t1.groupby(data_t1.index.date).mean()
    data_t2_mean = data_t2.groupby(data_t2.index.date).mean()

    # Intersección de fechas para obtener fechas comunes
    common_dates = data_t1_mean.index.intersection(data_t2_mean.index)

    # Seleccionar datos en fechas comunes
    data_t1_aligned = data_t1_mean.loc[common_dates]
    data_t2_aligned = data_t2_mean.loc[common_dates]

    # Calcular la autocorrelación si hay pares suficientes
    if not data_t1_aligned.empty and not data_t2_aligned.empty:
        return np.corrcoef(data_t1_aligned, data_t2_aligned)[0, 1]
    else:
        return np.nan  # Retornar NaN si no hay pares suficientes

data = pd.read_csv('seoul.csv')

# Ejemplo de uso de la función muestra:
muestra_ejemplo = muestra(data, 'o3', 113, 2020031400, 2020031723)
print(muestra_ejemplo.head())

# Ejemplo de uso de la función proceso:
proceso_ejemplo = proceso(data, 'o3', 113, 20200314, 20200317)
print(list(proceso_ejemplo.keys())) # Imprimir los días para los que se han extraído funciones muestra.
lista_fechas = [['2020031400', '2020031423'], ['2020031500', '2020031523']]
grafica2d(data, 'o3', 113, lista_fechas, 'graficas2d')
grafica3d(data, 'o3', 113, '2020031400', '2020031723')
autocorr_value = autocorrelacion(data, 'o3', 9, 17)
print(f"Autocorrelación entre las horas 9 y 17: {autocorr_value}")
data['dt'] = pd.to_datetime(data['dt'], format='%Y%m%d%H')
# Ejemplo de uso de la función:
distribucion(data, 'o3', 113)
