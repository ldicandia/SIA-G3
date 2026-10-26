# Referencias teóricas : TP4

Las páginas se cuentan desde uno en cada PDF. El [enunciado](SIATP42026.pdf) define los ejercicios y las clases de la tabla fundamentan los métodos utilizados.

## Métodos

| Técnica | Fuente | Aplicación |
|---|---|---|
| Descriptivos, covarianza y correlación | [PCA](<../clases/5 - PCA (2).pdf>), pp. 7 a 11 | EDA de los creadores |
| Boxplots y dispersión | PCA, pp. 43 a 45; [Análisis de Datos en Python](<../clases/Análisis de Datos en Python (1).pdf>), pp. 14 a 15 | Distribuciones y relaciones entre variables |
| Estandarización y efecto de escala | PCA, pp. 8 a 11, 25 a 27; [Kohonen](<../clases/2 - Modelo de Kohonen.pdf>), p. 19 | Mismas variables estandarizadas en PCA, Kohonen y Oja |
| Autovectores, proyecciones y varianza explicada | PCA, pp. 20 a 23, 31 a 32; [Autovalores y Autovectores](<../clases/4 - Autovalores y Autovectores (1).pdf>) | Componentes principales y representación de los datos |
| Biplot e interpretación del índice | PCA, pp. 34 a 38, 45 a 46 | Lectura de PC1/PC2 y comparación de creadores |
| PCA y StandardScaler de sklearn | Análisis de Datos en Python, p. 8 | Implementación de PCA autorizada para TP4 |
| Grillas rectangular/hexagonal e inicialización uniforme/con muestras | Kohonen, pp. 13 a 15, 24, 30 | Barridos de tamaño, topología e inicialización |
| Ganadora euclídea, vecindad binaria y actualización | Kohonen, pp. 25 a 29 | Entrenamiento del mapa |
| Tasa decreciente y radio decreciente o constante | Kohonen, pp. 26 a 27, 31 | η(i)=η₀/(1+i), radio final constante 1 |
| Impactos, matriz U, mapas de variables y prototipos | Kohonen, pp. 16, 33 a 37 | Interpretación de celdas y vecinos |
| Oja, norma y componente principal | [Oja y Sanger](<../clases/6 - Regla de Oja y Sanger (1).pdf>), pp. 12, 14 a 16 | Una neurona con pesos iniciales uniformes y η=.001 constante |
| Hebb, estados ±1 y diagonal cero | [Hopfield](<../clases/4.2 - Modelo de Hopfield.pdf>), pp. 15, 20 a 22 | Almacenamiento de letras y QR |
| Energía y actualización de una neurona por vez | Hopfield, pp. 34 a 36 | Recuperación asincrónica y trazas de energía |
| Ortogonalidad, interferencia y capacidad aproximada | Hopfield, pp. 33, 37 | Comparación de conjuntos de letras y QR |
| Espurios y ciclos | Hopfield, p. 38 | Clasificación de estados finales y condición de parada |

## Representación y convenciones

El catálogo tiene 50 creadores, 31 columnas numéricas y dos identificadores. Los cinco PNG coinciden con los datos suministrados; el CSV conserva sus valores y solo cambia los finales de línea.

Kohonen y Oja utilizan 27 variables estandarizadas. Se excluyen precio ARS, engagement total, alcance argentino estimado y porcentaje orgánico, cuyos componentes se conservan. Los cuatro tópicos y tipos de contenido se mantienen completos para tratarlos simétricamente en las distancias. La dependencia entre proporciones puede producir autovalores nulos y es compatible con PCA.

| Convención | Criterio |
|---|---|
| Boxplots | Bigotes a 1,5 RIC. Las clases muestran la gráfica sin fijar la fórmula de los bigotes. Se marcan observaciones y se conserva el catálogo completo |
| Correlaciones altas | El umbral de \|r\|≥.95 filtra una tabla descriptiva; la selección de columnas se justifica por su contenido |
| Escala | `ddof=0` para estandarizar, como StandardScaler; `ddof=1` para descriptivas muestrales |
| Signo de componentes | PCA presenta positivo el coeficiente de mayor magnitud. Para agregar repeticiones de Oja se alinea el signo con PC1 solo al informar; el aprendizaje conserva sus pesos originales. El autovector opuesto representa el mismo eje (PCA, p. 46) |
| Normalización de Oja | Se normaliza el vector para informar cargas e índice; la actualización utiliza los pesos originales |
| Orden de ejemplos | Barajado por época con semilla fija |
| Vecindario de Kohonen | `d <= R`, según los ejemplos de R=1 de pp. 15 y 31. Tolerancia `1e-9` para coordenadas hexagonales |
| Evaluación de Kohonen | Distancia euclídea media a la ganadora y ocupación resumen las distancias e impactos enseñados. La elección final se razona en el informe |
| Campañas | Tres creadores, mínimo 40% del tópico y orden por una variable observada. Son supuestos comerciales permitidos por el enunciado p. 4 |
| QR | Los PNG originales se convierten sin recorte ni redimensionado a 1369 estados ±1. La similitud divide el producto por las dos normas (37×37). Para el lector se agrega borde blanco y escala por repetición de píxeles |
| Campo local cero | Se conserva el estado anterior, según Hopfield p. 15, con tolerancia `1e-9` |
| Diagonal de Hopfield | Se fija `wᵢᵢ=0`, según p. 20. La expresión de p. 28 con `−I` requiere ajustar la diagonal cuando la cantidad de patrones difiere de N |

## Repeticiones

Las medias y los desvíos muestrales se basan en los descriptivos de la clase de PCA, pp. 7 a 11. Se usan 30 semillas para medir sensibilidad al azar; en los barridos se compara cada configuración con la misma lista de semillas.

Las barras muestran media ± un desvío estándar muestral entre corridas, con `ddof=1`. Describen la variación observada del experimento; no son intervalos de confianza ni una prueba de significación estadística. Las consultas de una misma corrida se agregan antes de calcular la dispersión entre semillas. PCA y los pesos de Hebb, para patrones fijos, son deterministas.

Los experimentos y sus interpretaciones se presentan en [Ejercicio 1](hallazgos/ej1.md) y [Ejercicio 2](hallazgos/ej2.md).
