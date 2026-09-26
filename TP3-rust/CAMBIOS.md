# Cambios en TP3-rust

Resumen de las correcciones hechas a `TP3-rust` para que cumpla con todos los puntos obligatorios de la consigna del TP3.

Estado final:

- Validaciones de la consigna: `validate_all` pasa (AND, `y=x`, `y=tanh(x)`, XOR `[2,2,1]` y `[2,3,2,1]`, escalón vs XOR).
- Tests: 43 OK. `clippy -D warnings` y `fmt --check` limpios.
- Ejercicio 3: **98,64% de accuracy en `digits_test.csv`**, así que se cumple el objetivo de ≥98% (antes daba 95,79%).
- Todos los outputs regenerados en `output/`.

## Qué se corrigió

### Ejercicio 1 (fraude)

- **Generalización con k-fold en lugar de un único split.**
  - Se separa un test estratificado del 15% que se consulta una sola vez, al final.
  - Sobre el 85% restante se hace validación cruzada estratificada de 5 folds, con un scaler ajustado dentro de cada fold.
  - Los resultados se reportan como media ± desvío (F1 0,898 ± 0,007).
- **Umbral elegido por F1 en lugar de accuracy.**
  - Se calcula sobre predicciones *out-of-fold*: cada score viene de un modelo que no vio esa fila.
  - Umbral recomendado: **0,858**.
  - Test: precision 0,965, recall 0,809, F1 0,880, average precision 0,950.
- **Decisiones de features justificadas con datos.**
  - `timestamp` se descarta: no hay señal por hora del día ni por día de la semana (`timestamp_profile.csv`).
  - `log1p` en las columnas con asimetría mayor a 2 (`amount_usd`, `days_since_last_purchase`, `time_since_last_login_s`).
  - Trade-off documentado: `log1p` baja la fidelidad a BigModel (R² de 0,88 a 0,82) pero sube el F1 real (de 0,875 a 0,898). La variante sin `log1p` queda en `configs/fraud_raw_features.toml`.
- **Hallazgo:** BigModel separa `flagged_fraud` perfectamente en 0,85 (legítimo máximo 0,8499, fraude mínimo 0,8501). El umbral del TinyModel cae casi ahí.
- **ReLU (opcional de la consigna)** agregado a la comparación de aprendizaje.
- **Persistencia del modelo:** `fraud_model.toml` guarda preprocesamiento, scaler, pesos y umbral. El comando `score` lo vuelve a levantar.

### Ejercicio 2 (dígitos)

- **Grilla de 26 candidatos que varía un factor por vez** alrededor de un modelo de referencia:
  - barrido de tasa de aprendizaje para cada optimizador (SGD, Momentum y Adam);
  - comparación de optimizadores, cada uno con su mejor tasa. Antes cada optimizador usaba una tasa distinta y el efecto del optimizador se mezclaba con el de la tasa;
  - 8 arquitecturas;
  - activación, tamaño de batch e inicialización.
- **Ganador:** `[784,256,128,10]`, con 96,83% en validación y 86,86% en test. El techo es de ~90% porque `digits.csv` no tiene ningún 8 (recall del 8 en test: 0%).
- **Respuestas (a) y (b)** de la consigna escritas en prosa en el README, con tablas.

### Ejercicio 3 (more_digits, objetivo ≥98%)

- **Escalera acumulativa de técnicas:** cada paso suma una sola técnica al anterior, así se lee el aporte de cada una.

  | Paso | Técnica agregada | Validación | Δ |
  |---|---|---:|---:|
  | Base | Ganador del Ej2 sin cambios, sobre los datos nuevos | 95,90% | |
  | t1 | Más capacidad `[784,512,256,10]` + inicialización He | 95,96% | +0,06 |
  | t2 | Decaimiento de la tasa ×0,5 cada 5 épocas | 96,73% | +0,76 |
  | t3 | Weight decay L2 desacoplado 1e-4 | 96,73% | 0,00 |
  | t4 | Dropout 0,2 | 96,63% | −0,10 |
  | t5 | Data augmentation (traslación ±2 px, rotación ±10°) | **98,06%** | **+1,43** |

- **Separación entre el aporte de los datos y el de las técnicas (pregunta c):**
  - Reentrenando el ganador del Ej2 sin cambios sobre `more_digits.csv`, el test pasa de 86,9% a **96,1%** solo por los datos (+9,2 pp). Las técnicas suman después +2,6 pp, hasta 98,64%.
  - El nuevo `data_shift.csv` documenta el cambio de datos: aparecen 585 imágenes del 8, las del 5 se duplican (de 271 a 542) y `more_digits.csv` comparte solo 3.689 imágenes con `digits.csv`.

### Recomendaciones de la cátedra

- **Operaciones matriciales:** el MLP entrena por mini-batch con productos de matrices, procesando sub-batches en paralelo. Da resultados idénticos al motor anterior y es unas 2 veces más rápido. Un test compara el gradiente analítico contra diferencias finitas.
- **Retomar entrenamientos:** nuevo comando `continue`. Los momentos de Momentum y Adam no se guardan, así que arrancan de cero al retomar.
- **Reporte de progreso:** siempre se imprime el inicio y el fin de cada candidato, fold y refit.

### Otros cambios a tener en cuenta

- **Versión mínima de Rust:** sube a 1.88, porque el clippy 1.98 instalado exige `as_chunks`. Ese clippy ya fallaba antes en un archivo que no se había tocado.
- **Configuraciones reemplazadas:** `configs/exercise2.toml` y `configs/exercise3.toml` se reescribieron completas.
- **Carpeta vieja:** `output/exercise3_noheldout/` es de una corrida anterior y no se tocó.

## Cambios por parámetro

| Parámetro | Antes | Ahora | Por qué |
|---|---|---|---|
| **Batch / online / mini-batch** | Ej1: online (una muestra por paso). Dígitos: mini-batch de 64, pero calculado muestra por muestra. | Ej1: **sin cambios**, sigue online. Dígitos: mini-batch con **operaciones matriciales**. El Ej2 compara **batch de 16, 64 y 256**; el Ej3 usa 64. | La cátedra recomienda operaciones matriciales, y así el Ej2 corre unas 2 veces más rápido. El tamaño de batch pasó a ser una variante estudiada: 64 fue el mejor (96,14%), 256 el peor (95,62%, menos actualizaciones por época). |
| **Bias / sin bias** | Todas las neuronas con bias. | **Sin cambios.** Lo único nuevo es que el weight decay **no se aplica a los biases**. | El bias es un desplazamiento, no complejidad del modelo: penalizarlo no reduce el sobreajuste y solo corre la frontera de decisión. |
| **Función de activación** | Ej1: lineal y sigmoide. Ej2: candidatos mezclados (sigmoide en unos, ReLU en otros). | Ej1: se agrega **ReLU**, que era opcional. Ej2: eje de activación **ReLU, sigmoide y tanh** con todo lo demás fijo. Salida softmax sin cambios. | Antes la activación variaba junto con otros factores y no se podía aislar su efecto. En el Ej1, ReLU se comporta casi como el lineal (R² 0,766 contra 0,762). En el Ej2, sigmoide y ReLU empatan (96,18% y 96,14%), pero la sigmoide necesita 35 épocas contra 12. |
| **Learning rate** | Ej1: grilla 0,001 a 0,05 elegida con un solo split, ganó 0,01. Ej2: solo 2 tasas comparables, y cada optimizador con una tasa distinta. | Ej1: misma grilla, elegida por **MSE medio en 5-fold**, gana 0,05. Ej2: **barrido por optimizador** (SGD 4 valores, Momentum 5, Adam 5). Ej3: **decaimiento ×0,5 cada 5 épocas**. | En el Ej1 las cuatro tasas empatan en MSE y solo cambian la velocidad de convergencia. En el Ej2, la tasa baja no converge en 40 épocas y la alta se estanca antes. En el Ej3, el decaimiento aporta **+0,76 pp** porque permite afinar los pesos cerca del mínimo. |
| **Función de pérdida** | Ej1: MSE contra BigModel. Dígitos: softmax + cross-entropy. | **Sin cambios.** Cambió el **criterio del umbral** del Ej1: de accuracy a **F1** sobre predicciones out-of-fold. Se agregó **weight decay L2** como regularización. | MSE contra la probabilidad de BigModel es justamente la destilación. El umbral por accuracy no sirve con 11,6% de fraudes: un modelo que nunca marca fraude ya tiene 88% de accuracy. |
| **Neuronas / arquitectura** | Ej1: 1 neurona, 9 entradas. Ej2: 3 topologías, ganó `[784,64,32,10]`. Ej3: ganó `[784,128,64,10]`. | Ej1: 1 neurona con **8 entradas** (se descarta `timestamp`, `log1p` en 3 columnas). Ej2: **8 arquitecturas**, gana `[784,256,128,10]`. Ej3: **`[784,512,256,10]`**. | El ancho importa más que la profundidad: de 32 a 128 neuronas suma +1,6 pp y una tercera capa oculta no mejora. En el Ej3, más capacidad sola aporta solo +0,06 pp: sirve cuando se combina con regularización y augmentation. |
| **Inicialización** | Xavier uniforme en todas las capas. | Nueva opción **He**, estudiada como eje en el Ej2 y usada en el Ej3. | He compensa que ReLU anula la mitad de las entradas. Con redes poco profundas el efecto es marginal (96,18% contra 96,14%); en la red más grande del Ej3 es la opción teóricamente correcta. |
| **Optimizadores** | Ej1: descenso por gradiente simple. Ej2: SGD, Momentum y Adam, cada uno con una tasa distinta. | Ej1: **sin cambios**. Ej2: cada optimizador comparado **con su mejor tasa**. Ej3: **Adam con weight decay desacoplado** (estilo AdamW). | Con la tasa bien elegida, los tres quedan a 0,3 pp de diferencia (SGD 96,02%, Momentum 95,82%, Adam 96,14%). La ventaja de Adam es la robustez: rinde bien en todo el rango de tasas, mientras que SGD cae 1,6 pp si la tasa se aleja de su óptimo. El weight decay desacoplado evita que Adam lo diluya con su normalización adaptativa. |
| **Épocas** | Ej1: early stopping (máximo 2000, paciencia 100); el refit usaba la mejor época de un solo split (6). Ej2: máximo 20, paciencia 4. Ej3: máximo 30, paciencia 5; el ganador corrió 18. | Ej1: el refit usa la **mediana de las épocas de early stopping entre folds** (76). Ej2: **máximo 40, paciencia 5**. Ej3: **máximo 60, paciencia 8**, y **100 y 12 con augmentation**; el ganador corre **51**. | La época de un solo split es ruidosa y la mediana entre folds es más estable. Con 20 épocas algunas tasas bajas no llegaban a converger. Con augmentation el modelo sigue mejorando por mucho más tiempo: el mejor llega en la época 51, contra la 6 sin augmentation. |

**Fuera de esta lista, en el Ej3** también se agregaron **dropout 0,2** y **data augmentation** (traslaciones de ±2 px y rotaciones de ±10°). La augmentation fue la técnica decisiva para pasar el 98%: aporta **+1,43 pp**. El dropout no aportó por sí solo (−0,10 pp), pero quedó en el modelo final porque cada paso de la escalera suma sobre el anterior.
