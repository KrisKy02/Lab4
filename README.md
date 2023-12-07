
# Estudiantes:

Kristel Herrera Rodríguez C13769

Oscar Porras Silesky C16042

Fabrizzio Herrera Calvo B83849


# Análisis de Contaminantes Atmosféricos

## Descripción General
Este proyecto proporciona un conjunto de herramientas para analizar mediciones de contaminantes atmosféricos. Las funciones permiten evaluar distribuciones de probabilidad, calcular medias temporales, determinar ergodicidad y más, facilitando el análisis estadístico de los datos ambientales.

## Requisitos
Este código requiere las siguientes bibliotecas:
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn

## Instalación y Ejecución
Para instalar las dependencias, ejecuta el siguiente comando:
```bash
pip install pandas numpy scipy matplotlib seaborn
```
Para ejecutar el análisis, primero importa las funciones desde `proceso.py` y luego utiliza cada función con los datos apropiados.

## Documentación Teórica
Las funciones implementadas en este proyecto se basan en la teoría estadística y matemática para el análisis de series temporales y datos ambientales. Se utilizan métodos como ajustes de distribución, cálculos de medias y análisis de ergodicidad para interpretar los datos de contaminantes.
## Función `muestra`

### Teoría detrás de la Función `muestra`

#### Concepto de Función Muestra
- Una función muestra `m(t)` es una representación de una variable a lo largo del tiempo. En el contexto de mediciones de contaminantes atmosféricos, `m(t)` representa las mediciones de un contaminante específico en momentos específicos.
- La función `muestra` se utiliza para extraer una secuencia específica de mediciones del contaminante en un intervalo de tiempo y ubicación específicos, de un proceso aleatorio `M(t)`.

#### Proceso Aleatorio M(t)
- `M(t)` es un concepto matemático que representa una serie de variables aleatorias indexadas en el tiempo. Cada medición de contaminante puede considerarse como una realización de este proceso aleatorio.
- Al extraer una función muestra de `M(t)`, seleccionamos una serie específica de mediciones basándonos en el tiempo y la ubicación.

#### Parámetros de la Función `muestra`
- `data`: El conjunto completo de datos, que incluye mediciones de varios contaminantes en diferentes momentos y ubicaciones.
- `variable`: El contaminante específico que se desea analizar.
- `loc`: Identificador del sensor, crucial para estudiar diferencias en mediciones basadas en la ubicación.
- `inicio` y `fin`: Definen el intervalo de tiempo para el cual queremos extraer las mediciones, permitiendo estudios enfocados en períodos específicos.

#### Aplicación Práctica
- Esta función es útil para investigadores y analistas que necesitan enfocarse en un conjunto específico de datos dentro de un marco temporal y espacial determinado, como analizar los cambios en los niveles de un contaminante en una ubicación específica durante un periodo concreto.
## Función `proceso`

### Teoría detrás de la Función `proceso`

#### Concepto de Proceso Aleatorio M(t)
- `M(t)` representa un proceso aleatorio de mediciones de un contaminante a lo largo del tiempo, permitiendo analizar cómo estas mediciones varían y se comportan.

#### Creación de Funciones Muestra Diarias
- La función `proceso` crea un conjunto de funciones muestra, cada una representando un día específico dentro de un rango de fechas, facilitando el análisis de variaciones diarias.

#### Parámetros de la Función `proceso`
- `data`: El conjunto de datos con mediciones de contaminantes.
- `variable`: El contaminante específico a estudiar.
- `loc`: Identificador del sensor, para enfocarse en mediciones de una ubicación específica.
- `inicio` y `fin`: Definen el rango de fechas para el análisis.

#### Indexación por Intervalo Diario
- La función genera un diccionario donde cada clave es una fecha específica y cada valor es un DataFrame con mediciones para ese día, permitiendo un análisis detallado y comparativo.

#### Aplicación Práctica
- Ideal para investigadores interesados en analizar patrones temporales en los datos de contaminación, como cambios diarios o identificación de eventos de contaminación específicos.

## Función `distribucion`

### Teoría detrás de la Función `distribucion`

#### Análisis de Distribución de Probabilidad
- Evalúa cómo se distribuyen los valores de los contaminantes, comparando varias distribuciones conocidas para encontrar la más adecuada.

#### Comparación y Ajuste de Distribuciones
- Utiliza el Criterio de Información de Akaike (AIC) para determinar la mejor distribución y ajusta un modelo polinomial a los parámetros de esta distribución.

#### Visualización y Análisis Horario
- Genera histogramas para cada hora del día, proporcionando una visualización de cómo cambia la distribución del contaminante a lo largo de un día.

#### Parámetros y Salida
- `data`: DataFrame con mediciones de contaminantes.
- `variable`: Contaminante a analizar.
- `loc`: Identificador del sensor.
- Salida: Un diccionario que contiene la distribución más común y los polinomios ajustados.
## Función `grafica2d`

### Teoría detrás de la Función `grafica2d`

#### Visualización de Datos en Series Temporales
- Gráficas 2D para visualizar series temporales de mediciones de contaminantes, mostrando cómo evoluciona la variable a lo largo del tiempo.

#### Comparación de Intervalos Temporales
- Permite comparar visualmente los niveles de contaminantes en diferentes periodos, identificando tendencias o cambios significativos.

#### Parámetros de la Función
- `data`: DataFrame con mediciones de contaminantes.
- `variable`: Contaminante a graficar.
- `loc`: Identificador del sensor.
- `lista_fechas`: Intervalos de tiempo para generar gráficas.
- `carpeta`: Directorio para guardar las gráficas.

#### Creación y Guardado de Gráficas
- Genera gráficas diferenciando cada intervalo de tiempo, facilitando la comparación visual.
- Muestra el nivel del contaminante en función de la fecha y hora, y guarda las gráficas para análisis posteriores.
## Función `grafica3d`

### Teoría detrás de la Función `grafica3d`

#### Visualización Tridimensional
- Las gráficas 3D permiten visualizar relaciones complejas entre múltiples variables, mostrando las distribuciones de probabilidad de las mediciones de un contaminante a lo largo del día.

#### Distribuciones de Probabilidad Horarias
- Analiza y grafica las distribuciones de probabilidad de un contaminante específico para cada hora del día, revelando variaciones en sus características.

#### Parámetros de la Función
- `data`: DataFrame con mediciones de contaminantes.
- `variable`: Contaminante a analizar.
- `loc`: Identificador del sensor.
- `inicio` y `fin`: Período de tiempo para el análisis.
- `carpeta`: Directorio para guardar la gráfica.

#### Análisis y Visualización
- Crea una gráfica 3D con ejes representando la hora del día, valores del contaminante, y densidad de probabilidad, utilizando colores para diferenciar cada hora.
## Función `autocorrelacion`

### Teoría detrás de la Función `autocorrelacion`

#### Concepto de Autocorrelación
- La autocorrelación mide la relación entre observaciones de una misma variable en diferentes momentos del tiempo, crucial para identificar patrones o dependencias en series temporales.

#### Cálculo de la Autocorrelación
- Calcula la correlación entre mediciones de un contaminante en dos horas específicas del día, agrupando los datos por fecha y calculando medias diarias.

#### Consideraciones Estadísticas
- Utiliza el coeficiente de correlación de Pearson, proporcionando un valor entre -1 y 1 para indicar la fuerza y la dirección de la correlación.
- Maneja situaciones con datos insuficientes, retornando NaN cuando no es posible calcular una correlación significativa.
#### Fórmula de la Autocorrelación
La autocorrelación entre dos puntos en el tiempo t1 y t2 se calcula como:

R(t1, t2) = E[(X(t1) - μ(t1)) * (X(t2) - μ(t2))] / (σ(t1) * σ(t2))

donde:
- R(t1, t2) es el coeficiente de autocorrelación.
- E es el valor esperado.
- X(t1) y X(t2) son los valores de la serie en los tiempos t1 y t2.
- μ(t1) y μ(t2) son las medias en los tiempos t1 y t2.
- σ(t1) y σ(t2) son las desviaciones estándar en los tiempos t1 y t2.

#### Cálculo en la Función `autocorrelacion`
En la implementación de `autocorrelacion`, se calculan primero las medias diarias para cada hora específica (\( t_1 \) y \( t_2 \)), y luego se utiliza la función `np.corrcoef` de NumPy para calcular el coeficiente de autocorrelación. Esta función aplica la fórmula anterior para medir la relación lineal entre las dos series temporales.

Este enfoque combina principios matemáticos y estadísticos fundamentales para proporcionar una comprensión profunda del comportamiento de la serie temporal.

## Función `autocovarianza`

### Teoría detrás de la Función `autocovarianza`

#### Concepto de Autocovarianza
- Mide cómo dos puntos diferentes en el tiempo de una misma serie temporal varían juntos, proporcionando una perspectiva más amplia que la autocorrelación.

#### Cálculo de la Autocovarianza
- Calcula la covarianza entre mediciones de un contaminante en dos horas específicas del día, agrupando los datos por fecha y calculando medias diarias.

#### Aplicación en Datos Ambientales
- Útil para identificar patrones temporales y la persistencia de condiciones ambientales a lo largo del tiempo.

#### Consideraciones Estadísticas
- Maneja situaciones con datos insuficientes, retornando NaN cuando no es posible realizar un cálculo significativo.
#### Fórmula de la Autocovarianza
La autocovarianza entre dos puntos en el tiempo t1 y t2 se calcula como:

C(t1, t2) = E[(X(t1) - μ(t1)) * (X(t2) - μ(t2))]

donde:
- C(t1, t2) es el valor de autocovarianza.
- E representa el valor esperado.
- X(t1) y X(t2) son los valores de la serie en los tiempos t1 y t2.
- μ(t1) y μ(t2) son las medias en los tiempos t1 y t2.

#### Cálculo en la Función `autocovarianza`
En la implementación de `autocovarianza`, se calculan las medias diarias para cada hora específica (\( t_1 \) y \( t_2 \)) y luego se evalúa la covarianza entre estas dos series temporales. La función se enfoca en calcular el valor esperado del producto de las desviaciones de \( X_{t_1} \) y \( X_{t_2} \) de sus respectivas medias.

Este enfoque permite comprender cómo dos puntos en el tiempo de una serie temporal están relacionados en términos de variación conjunta, proporcionando una perspectiva más profunda que la autocorrelación.

## Función `wss`

### Teoría detrás de la Función `wss`

#### Estacionariedad en Sentido Amplio
- Evalúa si una serie temporal, como las mediciones de un contaminante, tiene propiedades estadísticas consistentes a lo largo del tiempo, incluyendo la media y la autocovarianza/autocorrelación.

#### Cálculo de la Estacionariedad
- Verifica la consistencia de la media y las medidas de autocorrelación/autocovarianza de la variable a lo largo del tiempo, dentro de un umbral de variación aceptable.

#### Parámetros y Umbral
- `data`: DataFrame con mediciones de contaminantes.
- `variable`: Variable de interés.
- `datetime_col`: Columna con marcas de tiempo.
- `threshold`: Umbral para la variación aceptable en la media y autocorrelación/autocovarianza.

#### Aplicación en Análisis Ambiental
- Útil para determinar si las características de un contaminante son consistentes a lo largo del tiempo, importante para la predicción y modelización del comportamiento del contaminante.

## Función `prom_temporal`

### Teoría detrás de la Función `prom_temporal`

#### Media Temporal en Series Temporales
- Calcula el valor medio de una variable en un intervalo de tiempo específico, proporcionando una medida fundamental para entender tendencias generales en series temporales.

#### Selección de Intervalos Temporales
- Permite especificar un intervalo de tiempo para el análisis, facilitando el estudio de datos en marcos temporales concretos.

#### Implementación y Uso
- Verifica que los datos estén indexados por fecha y hora, filtra según intervalos temporales específicos y calcula la media de la variable seleccionada.

#### Aplicación Práctica
- Fundamental para el análisis preliminar de datos ambientales, ayudando a entender los niveles promedio de contaminantes en períodos específicos.

## Función `ergodicidad`

### Teoría detrás de la Función `ergodicidad`

#### Concepto de Ergodicidad
- Indica que las propiedades estadísticas de una serie temporal son consistentes tanto en todo el conjunto de datos como en cualquier subconjunto temporal.

#### Evaluación de la Ergodicidad
- Compara las medias temporales de la variable de interés en diferentes momentos con la media general del conjunto de datos, usando un margen de tolerancia para determinar si las diferencias están dentro de un rango aceptable.

#### Parámetros y Umbral
- `data`: DataFrame con mediciones de contaminantes.
- `variable`: Variable de interés.
- `margen_tolerancia`: Margen de tolerancia para la comparación de medias, generalmente establecido en 5%.

#### Aplicación en Análisis Ambiental
- Importante para determinar si las características de un contaminante son consistentes a lo largo del tiempo y si los datos son representativos de la serie temporal completa.

