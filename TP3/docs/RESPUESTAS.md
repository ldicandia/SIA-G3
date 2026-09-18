# Respuestas a las Preguntas del Enunciado — TP3

Este documento consolida, en prosa y con números concretos citados de cada `REPORT.md` de exercise, las respuestas a las preguntas graduadas del enunciado de TP3 para los tres ejercicios (Ej.1 a/b/c ×2, Ej.2 a/b, Ej.3 a/b/c).

## Ejercicio 1 — Comparación de Aprendizaje (Lineal vs. No Lineal)

### a) ¿Observan underfitting?

Sí, el perceptrón lineal (activación `identity`) exhibe underfitting marcado frente al no lineal (`sigmoid`). En el split de entrenamiento, el MSE del modelo lineal es **0.030937**, casi 3 veces mayor que el **0.011011** del modelo sigmoide (`docs/ejercicio1/REPORT.md` Sección 4). Esta brecha se confirma en el $R^2$ de test: **0.6895** para el lineal contra **0.8847** para el sigmoide — el hiperplano lineal carece de la capacidad para modelar la curvatura y las interacciones no lineales que genera BigModel, mientras que ambos modelos muestran un Test MSE similar o menor al Train MSE, descartando overfitting como explicación del bajo desempeño lineal.

### b) ¿Observan saturación de las capacidades?

Sí, ambos modelos presentan comportamiento en los extremos del rango $[0,1]$, aunque de naturaleza distinta. El perceptrón sigmoide registra un `saturation_fraction` de **0.0804** (8.04%) — predicciones dentro de $\epsilon=0.02$ de 0 o 1, donde la derivada $\sigma'(h)$ se aproxima a cero — mientras que el perceptrón lineal, al no tener función de activación acotada, registra un `clip_fraction` de **0.0948** (9.48%, 711 transacciones) que debieron truncarse artificialmente a $[0,1]$ porque sus predicciones sin clip alcanzan un máximo de 2.65 y un mínimo de -0.51.

### c) ¿Cuál seleccionarían para el estudio de generalización?

El modelo sigmoide fue seleccionado para el estudio de generalización. La regla de selección prioriza el mayor $R^2$ de test, y el sigmoide alcanza **0.8847** frente al **0.6895** del lineal — una ganancia absoluta de +0.1952, muy por encima de cualquier umbral de empate — además de producir probabilidades naturalmente acotadas en $[0,1]$ sin necesitar el clipping post-hoc que sí requiere el 9.48% de las predicciones lineales.

## Ejercicio 1 — Estudio de Generalización

### a) ¿Qué métricas de evaluación seleccionaron y por qué?

Se seleccionaron precisión, recall y $F_1$-score, no accuracy pura, porque el dataset está severamente desbalanceado: las transacciones legítimas (`flagged_fraud = 0`) representan **88.41%** (6,631 de 7,500) del total, mientras que las fraudulentas son solo **11.59%** (869 de 7,500). Un modelo trivial que siempre predice "legítimo" alcanzaría 88.41% de accuracy sin detectar un solo fraude, por lo que accuracy por sí sola resulta engañosa; precisión, recall y $F_1$ exponen el trade-off operacional real entre falsos positivos y frauds perdidos.

### b) ¿Qué estrategia utilizaron para manipular el conjunto de datos?

Se aplicó validación cruzada de 5 folds (5-fold cross-validation) sobre las 7,500 transacciones completas, con cada fold reservando 1,500 muestras (20%) para test y entrenando sobre las 6,000 restantes (80%). El escalado (`split-before-scale`, estandarización z-score) se ajusta de manera independiente dentro de cada fold, usando exclusivamente las 6,000 filas de entrenamiento de ese fold, evitando así cualquier fuga de información desde las filas de test hacia los parámetros de normalización.

### c) ¿Cuál es el mejor modelo y el umbral de detección recomendado?

El mejor modelo es el perceptrón sigmoide (seleccionado en la pregunta 1.c), y el umbral de detección recomendado es **$\tau = 0.90$**, obtenido mediante un barrido de 19 umbrales equidistantes ($\tau \in [0.05, 0.95]$) maximizando $F_1$-score sobre las predicciones pooled fuera de fold. A $\tau = 0.90$, la precisión sube a **89.15%** y el recall se mantiene en **84.12%** (731 de 869 fraudes capturados), logrando un $F_1$ de **0.8656** y reduciendo los falsos positivos en casi 95% (de 1,745 a solo 90) respecto del umbral por defecto $\tau = 0.50$.

## Ejercicio 2 — Digits MLP

### a) ¿Cómo evalúo el desempeño de mi sistema?

El desempeño se evalúa con dos métricas complementarias: accuracy global y recall por clase. La accuracy agregada sola resulta insuficiente porque el dígito 8 está completamente ausente de `digits.csv` (0 instancias) y el dígito 5 es una clase minoritaria severa (271 instancias, 2.18% del dataset) — un clasificador que nunca predice esas clases correctamente no sufre penalización visible en la accuracy agregada, mientras que el recall por clase ($R_c = TP_c / (TP_c + FN_c)$) diagnostica explícitamente esa ceguera estructural, reportando `None` (no 0.0) cuando el soporte de una clase es 0.

### b) ¿Qué variantes realizo para encontrar la solución?

Se exploraron tres ejes de variantes — tasa de aprendizaje, arquitectura y mecanismo de optimización — todos ajustados únicamente sobre `digits.csv`, con `digits_test.csv` reservado como el único chequeo final de producción. El baseline (`[784, 32, 10]`, sigmoid, SGD, $\eta=0.05$) alcanzó **94.13%** de val accuracy sobre el split interno; la configuración combinada finalmente seleccionada, chequeada una única vez contra `digits_test.csv`, alcanzó **82.78%** de accuracy final, con el dígito 8 en **0.0%** de recall (0 de 243 instancias) — confirmando la ceguera estructural por ausencia total de ejemplos de entrenamiento para esa clase.

## Ejercicio 3 — Accuracy Push

### a) ¿Cuál es el mejor resultado que pudieron obtener?

El mejor resultado alcanzado, medido en el único chequeo final sobre `data/derived/more_digits/heldout.csv` (`docs/ejercicio3/final_heldout_metrics.json`), es **92.85%** de accuracy (`docs/ejercicio3/REPORT.md` Sección 5) — reportado honestamente por debajo del objetivo de **98%** solicitado por CompanyX en el enunciado. `meets_98_percent_target` se registra explícitamente como `false`; el objetivo no fue redefinido ni suavizado.

### b) ¿Qué técnicas utilizaron para mejorar el rendimiento?

Se aplicaron tres técnicas deliberadas sobre el baseline de "más datos" (`val_accuracy = 0.9161`, Sección 2): variación de arquitectura (widening/deepening), variación de la semilla de inicialización de pesos, y ensembling por voto mayoritario. La arquitectura `[784, 64, 32, 10]` fue la mejor individual, alcanzando **0.9269** de val accuracy (+1.08 puntos porcentuales sobre el baseline); las cuatro semillas de inicialización dieron un rango genuino de 0.9161–0.9237; y el ensemble por voto mayoritario sobre esas cuatro semillas alcanzó **0.9266**, sin superar estrictamente a la mejor arquitectura individual, por lo que la regla de decisión determinística (`choose_final`) seleccionó el modelo único `arch_64-32` como configuración final.

### c) ¿Existen otros factores que influyeron en el cambio de rendimiento?

Sí — el factor "más datos" en sí mismo, separado explícitamente de las técnicas de la pregunta anterior (`docs/ejercicio3/REPORT.md` Sección 4, ACC-04). Numéricamente, el config idéntico de Ejercicio 2 re-entrenado sin cambios sobre `more_digits.csv` produjo una regresión de **-0.0152** en val accuracy (0.9313 → 0.9161) — más datos por sí solos no mejoraron la accuracy agregada. Estructuralmente, sin embargo, el dígito 8 pasó de 0 a 585 filas de entrenamiento y el dígito 5 de 271 a 542, lo cual explica por sí solo que el recall del dígito 8 pase de exactamente **0.0%** en `digits_test.csv` (Ejercicio 2) a **85.47%** en el chequeo final de Ejercicio 3 — un cambio de rendimiento causado enteramente por la composición del dataset, no por ninguna técnica de la pregunta 3.b.
