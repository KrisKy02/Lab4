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

### Función `prom_temporal`
Calcula la media temporal de una variable ambiental en un intervalo de tiempo seleccionado.

### Función `ergodicidad`
Determina si una secuencia aleatoria es ergódica basándose en un margen de tolerancia para la comparación de medias temporales y del conjunto.

## Ejemplos de Uso
Aquí se muestra cómo utilizar cada una de las funciones principales del proyecto:

### Uso de `distribucion`
```python
# Ejemplo de uso de la función distribucion
resultado_distribucion = distribucion(data, 'o3', 113)
```
