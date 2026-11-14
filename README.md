# Tarea 3 - Diseño de Experimentos (DOE)

## Descripción
Proyecto de diseño experimental aplicado al algoritmo de Recocido Simulado para resolver el Problema del Viajante (TSP).

Se analiza el efecto de la temperatura inicial sobre el costo de las rutas para diferentes tamaños de problema.

## Contenido del proyecto
- **codigo/**: contiene el programa principal `doe.py`.
- **resultados/**: contiene los archivos generados durante los experimentos.
- **resultados/graficas/**: contiene las gráficas de efectos principales e interacción.
- **reporte/**: contiene el reporte final en PDF.

## Configuración experimental
- Algoritmo: Recocido Simulado.
- Movimiento: 2-opt.
- Temperaturas: baja, media y alta.
- Tamaños del problema: 5, 10 y 15 ciudades.
- Repeticiones: 10 por combinación.
- Total: 90 experimentos.

## Herramientas utilizadas
- Python
- NumPy
- Pandas
- Matplotlib
- Statsmodels

## Autor
Matrícula: 2006646