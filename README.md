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
``bash
pip install pandas numpy scipy matplotlib seaborn

Para ejecutar el análisis, primero importa las funciones desde `proceso.py` y luego utiliza cada función con los datos apropiados.

## Documentación Teórica
Las funciones implementadas en este proyecto se basan en la teoría estadística y matemática para el análisis de series temporales y datos ambientales. Se utilizan métodos como ajustes de distribución, cálculos de medias y análisis de ergodicidad para interpretar los datos de contaminantes.

### Función `distribucion`
Calcula y compara diferentes distribuciones de probabilidad para los datos de un contaminante, identificando la distribución más común y ajustando un modelo polinomial.

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
