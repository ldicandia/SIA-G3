# TP3 - Perceptrons in Rust

Implementación educativa de perceptrones, *knowledge distillation* para detección de fraude y clasificación de dígitos manuscritos. El proyecto privilegia un core explícito y legible: no usa frameworks de machine learning y deja visibles el forward pass, la regla de actualización y backpropagation.

Todo el código, las interfaces, la configuración y los artefactos generados están en inglés. Este README está en español para responder directamente la consigna. Todos los números citados salen de la corrida verificada y se regeneran con los comandos de abajo.

## Requisitos y datos

- Rust 1.88 o posterior.
- Los archivos `fraud_dataset.csv`, `digits.csv`, `digits_test.csv` y `more_digits.csv` provistos por la materia. Los comandos de dígitos apuntan por defecto a `../TP3/data/data and documentation/`.
- El dataset no se copia ni se versiona en este repositorio.

## Ejecución

```bash
# Ejercicio 1 (fraude): exploración + comparación + generalización
cargo run --release -- run-all --data "/path/to/fraud_dataset.csv" --output output
# Etapas sueltas: inspect | compare | generalize (mismos argumentos)
# Levantar el modelo guardado y puntuar un CSV:
cargo run --release -- score --data "/path/to/fraud_dataset.csv" --model output/fraud_model.toml

# Ejercicio 2 (dígitos)
cargo run --release -- exercise2 train
cargo run --release -- exercise2 evaluate --model output/exercise2/selected_model.toml --output output/exercise2/evaluation

# Ejercicio 3 (more_digits, objetivo 98%)
cargo run --release -- exercise3 train
cargo run --release -- exercise3 evaluate --model output/exercise3/selected_model.toml --output output/exercise3/evaluation

# Retomar el entrenamiento de un modelo guardado (ambos ejercicios)
cargo run --release -- exercise3 continue --data "../TP3/data/data and documentation/more_digits.csv" \
  --model output/exercise3/selected_model.toml --epochs 10 --output output/exercise3/continued

# Validaciones de la consigna (AND, y=x, y=tanh(x), XOR [2,2,1] y [2,3,2,1], escalón vs XOR)
cargo run --release --example validate_all
```

Tiempos de referencia con 16 hilos: Ejercicio 1 completo en unos 5 s; los 26 candidatos del Ejercicio 2 en aproximadamente 1 min; el Ejercicio 3 en unos 4 min.

## Funcionalidades recomendadas por la cátedra

| Recomendación | Implementación |
|---|---|
| Operaciones matriciales | El MLP de dígitos entrena por mini-batch como productos de matrices: `Z = A·Wᵀ + b`, `D_prev = D·W`, `∇W = Dᵀ·A` (`src/matrix.rs`). Cada batch se divide en *shards* que se procesan en paralelo con `rayon` y cuyos gradientes se suman. Un test compara el gradiente analítico con diferencias finitas. |
| Reportar progreso | Siempre se imprime el inicio y el fin de cada candidato, fold y refit, con la accuracy o el MSE y el tiempo. El feature `training-logs` agrega una línea por época. `--live` abre un dashboard web en vivo (ver abajo). |
| Configuración extensible | Archivos TOML en `configs/`. Cada candidato de dígitos define todos sus hiperparámetros, y la configuración completa se guarda dentro del modelo persistido. |
| Guardar y levantar modelos | `selected_model.toml` para dígitos (pesos + configuración) y `fraud_model.toml` para fraude (preprocesamiento + scaler + pesos + umbral). `continue` retoma un entrenamiento guardado y `score` y `evaluate` levantan modelos. |
| Separar experimento y análisis | Cada corrida guarda CSVs con la loss por época, los hiperparámetros, la época elegida y el tiempo de ejecución (`learning_history.csv`, `candidate_summary.csv`, `generalization_trials.csv`...). Los gráficos se derivan de esos datos. |

### Monitor gráfico en vivo

Con `--live`, `exercise2 train` y `exercise3 train` lanzan un proceso monitor separado (`http://127.0.0.1:7878`) que grafica la loss y la accuracy de train y validación de todos los candidatos. El entrenamiento publica los eventos con un `try_send` no bloqueante: si la cola se llena, el evento se descarta en lugar de frenar el entrenamiento. Sin `--live`, el publisher no hace nada. Las direcciones se configuran con `--live-address` y `--live-http-address`.

## Ejercicio 1: TinyModel de fraude

### Exploración del dataset

`inspect` produce `dataset_summary.csv`, `data_profile.csv`, `feature_analysis.csv`, `timestamp_profile.csv` y los histogramas.

- 7.500 transacciones, 9 features, sin faltantes ni filas duplicadas.
- 869 fraudes confirmados (11,59%): el problema está desbalanceado.
- `big_model_fraud_probability` es la salida del BigModel y `flagged_fraud` es la verdad de campo. Según la documentación, `flagged_fraud` **no debe usarse para entrenar**. Por eso el TinyModel se entrena contra la salida del BigModel (destilación con *soft targets*), y `flagged_fraud` se usa solo para evaluar y elegir el umbral.
- El BigModel separa `flagged_fraud` perfectamente: el legítimo con score más alto vale 0,8499 y el fraude con score más bajo vale 0,8501 (average precision 1,0). Su umbral implícito es 0,85.

**Decisiones de features** (`[features]` en `configs/default.toml`, justificadas con `feature_analysis.csv`):

| Columna | Decisión | Motivo |
|---|---|---|
| `timestamp` | Se descarta | Correlación de Pearson con BigModel de 0,001. La media del BigModel por hora del día y por día de la semana es plana (0,40 a 0,46; ver `timestamp_profile.csv`). Un epoch absoluto no generaliza a transacciones futuras. |
| `amount_usd`, `days_since_last_purchase`, `time_since_last_login_s` | `log1p` | Asimetría > 2 (4,5, 2,1 y 2,1). Comprimir la cola evita que unas pocas filas extremas dominen el gradiente de una sola neurona. |
| Resto | Se mantiene | `device_screen_resolution` y `time_since_last_login_s` casi no correlacionan con el target, pero se dejan y el modelo decide su peso. |

Todas las columnas se estandarizan después (z-score), porque sus escalas difieren en varios órdenes de magnitud.

Las transformaciones tienen un trade-off medido: `log1p` baja la fidelidad al BigModel pero mejora la detección de fraude real. Se eligió `log1p` porque el objetivo de negocio es detectar fraude. La variante sin `log1p` queda en `configs/fraud_raw_features.toml` para reproducir la comparación:

| Features | R² vs BigModel (todas las muestras) | F1 CV (media) | Average precision CV | F1 test |
|---|---:|---:|---:|---:|
| Sin `timestamp`, sin `log1p` | 0,881 | 0,875 | 0,954 | 0,858 |
| Sin `timestamp`, con `log1p` (**elegida**) | 0,822 | **0,898** | **0,957** | **0,880** |

### Comparación de aprendizaje: lineal vs no lineal

Se usan todas las muestras, como pide la consigna. Se entrenan con MSE contra la salida del BigModel el perceptrón lineal, el sigmoide y, como opcional, el ReLU. Cada uno prueba la misma grilla de tasas de aprendizaje y se queda con su mejor resultado, para no confundir capacidad con una mala tasa.

| Modelo | MSE | R² | Rango de salida | Fuera de rango o saturado |
|---|---:|---:|---|---:|
| Lineal | 0,02174 | 0,762 | `[-0,31; 1,41]` | 7,6% fuera de `[0,1]` |
| Sigmoide | **0,01629** | **0,822** | `[0,006; 0,999]` | 3,7% saturado (<0,01 o >0,99) |
| ReLU (opcional) | 0,02136 | 0,766 | `[0; 1,43]` | 5,5% por encima de 1 |

**a) ¿Underfitting?** Sí, sobre todo en el lineal. Su error es un 33% mayor que el del sigmoide aun sobre los mismos datos con que entrena, y la curva de loss (`learning_curves.png`) se aplana en un nivel alto. Un hiperplano no puede reproducir la forma en S de la probabilidad del BigModel. ReLU se comporta casi como el lineal: es lineal por tramos y solo recorta la parte negativa.

**b) ¿Saturación de las capacidades?** Sí, en dos sentidos:

- Los tres modelos llegan a un plateau con error residual. Una sola neurona no alcanza para capturar interacciones entre features: el límite es la capacidad, no la cantidad de épocas.
- El sigmoide satura en los extremos: el 3,7% de sus salidas queda por debajo de 0,01 o por encima de 0,99, donde `σ'(h) ≈ 0` y esas muestras casi no aportan gradiente.
- El lineal y el ReLU no saturan, pero producen valores que no son probabilidades.

**c) ¿Cuál se elige para generalizar?** El sigmoide: tiene menor error y su salida está en `[0,1]` por construcción, así que se puede leer como probabilidad sin recortarla.

### Estudio de generalización

**a) Métricas y por qué.**

- **Contra el BigModel (fidelidad de destilación):** MSE, RMSE y R².
- **Contra `flagged_fraud`:**
  - precision (de lo que marco, cuánto es fraude: costo de revisiones inútiles);
  - recall (de los fraudes, cuántos detecto: costo del fraude que se escapa);
  - F1, que resume ambas;
  - average precision (área bajo la curva precision-recall, independiente del umbral).
- Accuracy y specificity se informan, pero no se usan para decidir: un modelo que nunca marca fraude ya tiene 88,4% de accuracy.

**b) Estrategia de manejo de datos.**

- Primero se separa un test estratificado del 15% (1.126 filas) que se consulta **una sola vez**, al final.
- Sobre el 85% restante se hace **validación cruzada estratificada de 5 folds**. La estratificación es por bandas de la probabilidad del BigModel, y cada fold tiene 11,1% a 11,8% de fraude.
- Cada fold ajusta su propio scaler con sus filas de entrenamiento, así que no hay fuga de información hacia validación.
- Con los folds se elige la tasa de aprendizaje (menor MSE medio de validación) y la cantidad de épocas (mediana de las épocas de early stopping).
- El modelo final se reentrena con todo el 85% y se evalúa en test.

¿Cómo se elige el mejor conjunto de entrenamiento? **No se elige**: elegir el split "que mejor da" es sobreajustar a la partición. K-fold hace que cada fila sea validación exactamente una vez y permite reportar media ± desvío. El desvío chico (F1 0,898 ± 0,007) muestra que el resultado no depende de un split afortunado.

| Métrica (5 folds, umbral elegido) | Media ± desvío |
|---|---:|
| MSE vs BigModel | 0,0164 ± 0,0006 |
| R² vs BigModel | 0,821 ± 0,006 |
| Precision | 0,929 ± 0,013 |
| Recall | 0,869 ± 0,013 |
| F1 | 0,898 ± 0,007 |
| Average precision | 0,957 ± 0,005 |

Las cuatro tasas de aprendizaje probadas dan un MSE de validación prácticamente igual (0,01640 a 0,01648). El problema es convexo en la práctica y la tasa solo cambia la velocidad de convergencia. Se eligió 0,05 con 76 épocas.

**c) Mejor modelo y umbral recomendado.** El modelo es un perceptrón simple sigmoide con 8 entradas: 9 parámetros contra un BigModel desconocido. Se guarda en `output/fraud_model.toml` junto con su preprocesamiento.

El umbral **recomendado es 0,858**. Es el que maximiza F1 sobre las predicciones *out-of-fold* de desarrollo: cada score usado para elegirlo viene de un modelo que no vio esa fila. En caso de empate se prefiere mayor recall, porque un fraude que se escapa suele costar más que una revisión manual. Coincide con el corte natural del BigModel en 0,85.

Resultado en test (1.126 transacciones, consultado una vez):

| Métrica | Valor |
|---|---:|
| MSE / R² vs BigModel | 0,0157 / 0,828 |
| Precision / Recall / F1 | 0,965 / 0,809 / 0,880 |
| Average precision | 0,950 |
| Specificity / Accuracy | 0,996 / 0,973 |
| TP / FP / TN / FN | 110 / 4 / 986 / 26 |

`threshold_sweep.csv` y `threshold_metrics.png` guardan precision, recall, F1 y accuracy para cada umbral. Si CompanyX tiene costos explícitos (por ejemplo, un fraude cuesta 20 veces lo que una revisión), puede elegir otro punto operativo con esa tabla sin reentrenar. Por ejemplo, bajar el umbral sube el recall a cambio de precision.

## Ejercicio 2: clasificación de dígitos

### a) ¿Cómo se evalúa el desempeño?

- **Protocolo:** `digits.csv` se divide 80/20 estratificado por clase. Todo el ajuste de parámetros e hiperparámetros usa solo esa partición: el candidato se elige por mayor accuracy de validación y la cantidad de épocas por early stopping sobre la loss de validación. El ganador se reentrena desde cero con el 100% de `digits.csv` y recién entonces `evaluate` lee `digits_test.csv`, una sola vez, como "producción".
- **Métricas:**
  - cross-entropy (lo que se optimiza, sensible a la confianza);
  - accuracy (lo que pide el cliente);
  - **recall por clase y matriz de confusión**: `digits.csv` no tiene ningún 8 y tiene pocos 5 (271, contra ~1.500 de las demás clases), así que la accuracy global esconde clases que el modelo no puede aprender.
- La distancia entre la accuracy de train y la de validación (`train_accuracy_at_best` en `candidate_summary.csv`) diagnostica sobreajuste.

### b) ¿Qué variantes se exploraron?

La grilla (`configs/exercise2.toml`, 26 candidatos) varía **un factor a la vez** alrededor de un modelo de referencia: `[784,64,32,10]`, ReLU, Adam con tasa 0,001, batch 64, inicialización Xavier. Así, cada eje es una comparación controlada. `validation_accuracy.png` muestra todos los candidatos agrupados por eje, y `loss_curves_<eje>.png` las curvas de cada eje.

**Tasa de aprendizaje (por optimizador)** — accuracy de validación (época del mejor modelo):

| SGD | | Momentum 0,9 | | Adam | |
|---|---:|---|---:|---|---:|
| 0,01 | 94,38% (40, no convergió) | 0,001 | 94,34% (40, no convergió) | 0,0001 | 94,98% (38) |
| 0,05 | 95,50% (22) | 0,005 | 95,34% (19) | 0,0003 | 95,74% (27) |
| **0,1** | **96,02%** (15) | 0,01 | 95,42% (12) | **0,001** | **96,14%** (12) |
| 0,3 | 95,82% (5) | 0,05 | 95,78% (5) | 0,003 | 95,82% (4) |
| | | **0,1** | **95,82%** (6) | 0,01 | 95,94% (3) |

- Una tasa demasiado baja no llega a converger en 40 épocas.
- Una tasa alta converge en pocas épocas, pero se estanca en un mínimo peor.
- Momentum con β = 0,9 multiplica la tasa efectiva por ~10: su óptimo está un orden de magnitud por debajo del de SGD.

**Mecanismo de optimización**, cada uno con su mejor tasa: SGD 96,02%, Momentum 95,82% y Adam 96,14%.

- Con una buena tasa, los tres llegan a resultados parecidos (0,3 pp de diferencia).
- La diferencia real está en la **robustez**: Adam queda por encima de 95,7% en todo el rango 0,0003 a 0,01, mientras que SGD cae 1,6 pp si la tasa se aleja de su óptimo.
- En tiempo: Momentum llega a su mejor época antes (5 a 6 épocas) que SGD (15).
- Comparar optimizadores con una misma tasa fija habría medido la tasa, no el optimizador.

**Arquitectura** (Adam 0,001):

| Ocultas | 32 | 64 | 128 | 256 | 64-32 | 128-64 | **256-128** | 128-64-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validación | 95,10% | 96,18% | 96,71% | 96,75% | 96,14% | 96,63% | **96,83%** | 96,58% |

El ancho importa más que la profundidad: pasar de 32 a 128 neuronas suma 1,6 pp, mientras que agregar una tercera capa oculta no mejora. Todas las redes llegan a más de 99% en train, así que el límite es la generalización, no la capacidad.

**Otros hiperparámetros:**

- **Activación oculta:** ReLU 96,14%, sigmoide 96,18% (necesita 35 épocas contra 12) y tanh 95,78%.
- **Batch:** 16 da 96,06%, 64 da 96,14% y 256 da 95,62% (menos actualizaciones por época).
- **Inicialización:** He da 96,18% contra 96,14% de Xavier; con esta profundidad la diferencia es marginal.

**Modelo seleccionado:** `[784,256,128,10]`, ReLU, Adam con tasa 0,001, batch 64, 6 épocas. Tiene 96,83% en validación y **86,86% en `digits_test.csv`**.

La caída es estructural. `digits_test.csv` sí tiene 8 (243 muestras, 9,7% del test) y el modelo nunca vio uno, así que su recall es 0% y el techo alcanzable es ~90,3%. Sin contar el 8, el recall por clase va de 88% (el 5, la clase minoritaria) a 99,6%. Ninguna variante de hiperparámetros puede corregir una clase ausente del entrenamiento.

## Ejercicio 3: más datos y objetivo ≥ 98%

### a) Mejor resultado

**98,64% de accuracy en `digits_test.csv`**, con loss 0,047: se cumple el objetivo de CompanyX (`meets_98_percent_target = true`).

- **Protocolo:** es el mismo del Ejercicio 2. Se ajusta con un 80/20 estratificado de `more_digits.csv`, se reentrena con el 100% y se evalúa una sola vez en `digits_test.csv`, que la consigna define como el conjunto de "producción" también para este ejercicio.
- **Recall por clase:** está entre 96,9% (el 5) y 99,6%, y el 8 pasa de 0% a 97,1%.
- **Modelo seleccionado:** `[784,512,256,10]`, ReLU, inicialización He, Adam con tasa 0,001 y decaimiento ×0,5 cada 5 épocas, weight decay 1e-4, dropout 0,2, data augmentation (traslaciones de ±2 px y rotaciones de ±10°), 51 épocas.

### b) Técnicas aplicadas

Primero se reentrena **sin cambios** el ganador del Ejercicio 2 sobre `more_digits.csv` (línea de base controlada). Después, `configs/exercise3.toml` define una **escalera acumulativa**: cada paso mantiene todo lo anterior y agrega una sola técnica, así que su aporte es la diferencia con el paso previo.

| Paso | Técnica agregada | Validación | Δ |
|---|---|---:|---:|
| Base | Ganador del Ejercicio 2 sobre los datos nuevos | 95,90% | |
| t1 | Más capacidad `[784,512,256,10]` + inicialización He | 95,96% | +0,06 |
| t2 | Decaimiento de la tasa ×0,5 cada 5 épocas | 96,73% | +0,76 |
| t3 | Weight decay L2 desacoplado 1e-4 | 96,73% | 0,00 |
| t4 | Dropout 0,2 en las capas ocultas | 96,63% | −0,10 |
| t5 | Data augmentation (traslación ±2 px, rotación ±10°) | **98,06%** | **+1,43** |

- **Más capacidad sola no ayuda:** la red ya llega a 99,6% en train, así que el problema es la varianza, no el sesgo.
- **El decaimiento de la tasa** permite afinar los pesos cerca del mínimo en lugar de oscilar.
- **Data augmentation es la técnica decisiva:** cada época ve una variante distinta de cada imagen, lo que actúa como un dataset mucho más grande y enseña invariancia a pequeños desplazamientos y rotaciones, la principal fuente de variación en dígitos manuscritos. Con augmentation el modelo sigue mejorando hasta la época 51, mientras que sin ella el early stopping corta en la 6.
- **Weight decay y dropout** no mejoran solos a esta cantidad de épocas: con early stopping temprano casi no hay tiempo para sobreajustar. Se mantienen en la escalera porque es acumulativa; su efecto aislado se lee en su fila.

### c) Otros factores además de las técnicas

Sí: **los datos mismos**, y se separan de las técnicas usando la línea de base. `data_shift.csv` documenta el cambio:

| | `digits.csv` | `more_digits.csv` |
|---|---:|---:|
| Filas | 12.449 | 15.741 |
| Dígito 8 | **0** | **585** |
| Dígito 5 | 271 | 542 |
| Imágenes compartidas con `digits.csv` | | 3.689 (las otras 12.052 son nuevas) |

- **Aparece el 8.** Es el factor dominante: solo esto levanta el techo de ~90% del Ejercicio 2. Reentrenando los hiperparámetros ganadores del Ejercicio 2 sin cambios sobre `more_digits.csv` y evaluando en `digits_test.csv`, la accuracy pasa de 86,9% a **96,1%**, y el recall del 8 de 0% a 91,4%, **únicamente por cambiar los datos**. Las técnicas de (b) suman después los 2,6 pp restantes, hasta 98,6%. En esta mejora pesan más los datos (+9,2 pp) que las técnicas.
- **El 5 duplica sus ejemplos:** su recall en test sube de 88,3% a 93,3% solo por los datos, y a 96,9% con las técnicas.
- **La validación de la línea de base** baja de 96,83% a 95,90%, pero no son comparables: la de `more_digits` incluye 8, una clase nueva y más difícil, y es otra partición.
- **Más datos y en su mayoría nuevos** (12.052 imágenes que no estaban en `digits.csv`) aumentan la variedad de estilos de escritura. Además, el test está balanceado (~250 por clase), así que cualquier clase mal aprendida pesa ~10% de la accuracy.

`exercise3_comparison.csv` resume la cadena: ganador del Ejercicio 2 → mismos hiperparámetros sobre los datos nuevos → candidato ajustado.

## Diseño

- `matrix`: la matriz `DenseMatrix` (row-major) y los kernels de batch.
- `loss` y `model`: el core matemático, con perceptrón escalón, perceptrón simple, MLP e inicialización Xavier o He.
- `training`: SGD online del perceptrón simple y del MLP escalar, con early stopping.
- `data`: loader validado del CSV de fraude, preprocesamiento de features configurable y `StandardScaler`.
- `split`: holdout y k-fold estratificados y reproducibles.
- `metrics`: métricas de regresión y clasificación, F1, average precision y barrido de umbrales.
- `experiment`: orquesta el Ejercicio 1, genera CSV y PNG, y persiste el TinyModel.
- `digits`: loader, configuración de candidatos, entrenamiento por mini-batch (softmax + cross-entropy, SGD, Momentum, Adam, weight decay, dropout, decaimiento de la tasa, augmentation), métricas, persistencia, reportes y dashboard.
- `exercises`: orquestación separada de cada ejercicio.

**Reproducibilidad:** todo el azar (splits, inicialización, orden de las muestras, máscaras de dropout y augmentation) sale de semillas fijas. Los números aleatorios de cada muestra dependen solo de la semilla, la época y el índice de la fila, no del hilo que la procesa. Sumar gradientes en paralelo puede cambiar el redondeo en el último dígito, sin efecto en las métricas.

**Retomar entrenamientos:** `continue` sigue entrenando desde los pesos guardados, continuando el calendario de la tasa de aprendizaje y los streams aleatorios. Los momentos de Momentum y Adam no se persisten y reinician en cero.

## Tests y calidad

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Los tests cubren:

- losses, softmax estable y derivadas de las activaciones;
- los kernels matriciales contra productos a mano, y el gradiente por batch contra diferencias finitas (sumando shards);
- scaler, preprocesamiento de features, holdout y k-fold;
- F1, average precision y selección de umbral;
- la persistencia de ambos modelos (ida y vuelta preservando predicciones), la compatibilidad con configuraciones viejas y el calendario de la tasa;
- la augmentation, y que un paso de SGD, Momentum o Adam reduzca la pérdida.

## Fuera de alcance

- **Opcionales de los Ejercicios 2 y 3:** no se implementan el estudio de robustez ante ruido ni las técnicas de interpretabilidad.
- **Opcionales del fraude:** la calibración probabilística no se implementa. La activación ReLU sí está incluida en la comparación, y la construcción y selección de features se discute arriba.
