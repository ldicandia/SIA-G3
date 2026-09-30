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

# Opcionales del Ejercicio 1 (escriben en output/calibration y output/feature_study)
cargo run --release -- calibrate --data "/path/to/fraud_dataset.csv" --output output --cost-ratio 20
cargo run --release -- feature-study --data "/path/to/fraud_dataset.csv" --output output

# Opcionales de los Ejercicios 2 y 3
# Modelo sin augmentation (paso t4 de la escalera), para comparar robustez al ruido:
cargo run --release -- exercise3 train --config configs/exercise3_no_augmentation.toml \
  --output output/exercise3_no_augmentation
cargo run --release -- exercise3 evaluate --model output/exercise3_no_augmentation/selected_model.toml \
  --output output/exercise3_no_augmentation/evaluation
# Robustez al ruido (el primer --model es el que se muestra en noisy_examples.png):
cargo run --release -- exercise3 noise --model output/exercise3/selected_model.toml \
  --model output/exercise3_no_augmentation/selected_model.toml \
  --model output/exercise2/selected_model.toml --output output/exercise3/noise
# Métodos de atribución (saliency, gradiente × input, Integrated Gradients):
cargo run --release -- exercise3 attribution --model output/exercise3/selected_model.toml \
  --output output/exercise3/attribution
```

`exercise2 noise` y `exercise2 attribution` existen con los mismos argumentos. Los comandos de dígitos aceptan `--data` (y `exercise3 train` también `--previous-data` y `--baseline-model`) si los CSV no están en la ruta por defecto.

Tiempos de referencia con 16 hilos: Ejercicio 1 completo en unos 5 s; los 26 candidatos del Ejercicio 2 en aproximadamente 1 min; el Ejercicio 3 en unos 4 min. Opcionales (medidos en esta corrida, con el binario ya compilado): `calibrate` 1,5 s; `feature-study` 34 s (18 variantes × 20 entrenamientos de CV); entrenar `exercise3_no_augmentation` 42 s; `exercise3 noise` con 3 modelos 2,2 s; `exercise3 attribution` 1,9 s.

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

`threshold_sweep.csv` y `threshold_metrics.png` guardan precision, recall, F1 y accuracy para cada umbral. Si CompanyX tiene costos explícitos (por ejemplo, un fraude cuesta 20 veces lo que una revisión), puede elegir otro punto operativo con esa tabla sin reentrenar. Por ejemplo, bajar el umbral sube el recall a cambio de precision. Con el score calibrado, ese umbral sale directo de los costos (ver "Opcionales", calibración).

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
- `calibration`: Brier, ECE, bins de confiabilidad, Platt scaling (Newton-Raphson) e isotónica (pool-adjacent-violators), sin I/O.
- `experiment`: orquesta el Ejercicio 1, genera CSV y PNG, y persiste el TinyModel. `generalize`, `calibrate` y `feature-study` comparten el mismo protocolo de CV (`cross_validate` + `refit_tinymodel`); `experiment::calibration` y `experiment::feature_study` son los dos opcionales.
- `digits`: loader, configuración de candidatos, entrenamiento por mini-batch (softmax + cross-entropy, SGD, Momentum, Adam, weight decay, dropout, decaimiento de la tasa, augmentation), métricas, persistencia, reportes y dashboard. `digits::noise` (ruido gaussiano y sal y pimienta), `digits::attribution` (saliency, gradiente × input, Integrated Gradients) y `digits::image_grid` (grillas de mapas 28×28) son los opcionales.
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
- la augmentation, y que un paso de SGD, Momentum o Adam reduzca la pérdida;
- calibración: Brier y ECE contra ejemplos a mano, Platt (pendiente positiva, baja Brier y ECE sin cambiar la average precision) e isotónica (monótona, agrupa pares que violan el orden);
- features derivadas: nombres, valores calculados a mano (incluida la hora y el fin de semana), igualdad entre la transformación por matriz y por fila, y compatibilidad de `fraud_model.toml` viejos (`derived` vacío no se serializa);
- ruido: σ = 0 y fracción 0 devuelven la entrada, todo queda en [0, 1], misma semilla da el mismo ruido y la corrupción sal y pimienta está anidada entre fracciones;
- atribución: el gradiente respecto de la entrada contra diferencias finitas centrales (tanh y sigmoide) y la completitud de Integrated Gradients (`Σ IG = z_c(x) − z_c(0)`) en una red suave.

## Opcionales

Los cuatro opcionales de la consigna están implementados y medidos. El opcional de ReLU del Ejercicio 1 ya está cubierto en "Comparación de aprendizaje". Todos los números de esta sección salen de los CSV en `output/calibration`, `output/feature_study`, `output/exercise3/noise` y `output/exercise3/attribution`.

### Ejercicio 1: construcción y descarte de features

`feature-study` evalúa cada variante con **exactamente el mismo protocolo de CV** que `generalize`: el mismo holdout, los mismos 5 folds, el scaler por fold, la búsqueda de tasa y el umbral F1 out-of-fold. **Solo usa las filas de desarrollo: el test no se consulta.** La fila `baseline` reproduce los números documentados (F1 0,898 ± 0,007, AP 0,957 ± 0,005). Las variantes son:

- `drop:<col>`: sacar una columna cruda que hoy se usa;
- `add:<feature>`: sumar una feature derivada;
- `add:all_derived`: sumar todas las derivadas;
- `add:selected`: sumar las derivadas que por separado mejoraron **a la vez** el F1 medio y la AP media de CV.

**Features derivadas.** Se calculan desde las columnas crudas, así que las de hora funcionan aunque `timestamp` esté descartado. Los cocientes de columnas con colas pesadas también tienen colas pesadas, por eso se comprimen con `ln(1 + x)`. `derived_feature_profile.csv` guarda la asimetría antes y después de comprimir, y la correlación de Pearson con el BigModel y con `flagged_fraud` (sobre todas las filas, exploratorio como `inspect`):

| Feature (`derived` en TOML) | Definición | Asimetría (antes de `log1p`) | ρ BigModel | ρ `flagged_fraud` |
|---|---|---:|---:|---:|
| `amount_per_item` | `ln(1 + amount / max(quantity, 1))` | 0,09 (4,40) | 0,269 | 0,185 |
| `items_viewed_per_minute` | `ln(1 + items_viewed / max(session_s / 60, 1/60))` | 2,07 (9,61) | 0,514 | 0,695 |
| `viewed_per_purchased` | `ln(1 + items_viewed / max(quantity, 1))` | 0,85 (2,44) | −0,239 | −0,055 |
| `amount_per_account_day` | `ln(1 + amount / (account_age_days + 1))` | 4,04 (26,50) | 0,590 | 0,768 |
| `amount_x_quantity` | `ln(1 + amount) · ln(1 + quantity)` (interacción) | 0,95 | 0,705 | 0,643 |
| `hour_of_day` | `hour_sin`, `hour_cos` = sin/cos(2π·hora/24), hora UTC | −0,00 / −0,00 | 0,006 / 0,002 | −0,001 / −0,003 |
| `weekend` | `is_weekend` ∈ {0, 1}, misma convención que `timestamp_profile.csv` | 0,98 | 0,010 | 0,011 |

**Resultados** (`feature_study.csv` y `feature_study.png`; media ± desvío sobre los 5 folds, Δ en puntos porcentuales contra el baseline):

| Variante | Features | F1 CV | AP CV | ΔF1 | ΔAP |
|---|---:|---:|---:|---:|---:|
| `baseline` (configs/default.toml) | 8 | 0,898 ± 0,007 | 0,957 ± 0,005 | | |
| `add:amount_per_item` | 9 | 0,899 ± 0,005 | 0,956 ± 0,006 | +0,09 | −0,12 |
| `add:items_viewed_per_minute` | 9 | 0,897 ± 0,008 | 0,957 ± 0,005 | −0,07 | −0,06 |
| `add:viewed_per_purchased` | 9 | 0,896 ± 0,008 | 0,957 ± 0,005 | −0,17 | −0,01 |
| `add:amount_per_account_day` | 9 | **0,906 ± 0,008** | **0,960 ± 0,005** | **+0,79** | **+0,28** |
| `add:amount_x_quantity` | 9 | 0,896 ± 0,009 | 0,957 ± 0,006 | −0,15 | −0,03 |
| `add:hour_of_day` | 10 | 0,896 ± 0,010 | 0,957 ± 0,006 | −0,15 | −0,01 |
| `add:weekend` | 9 | 0,899 ± 0,006 | 0,957 ± 0,005 | +0,09 | +0,02 |
| `add:all_derived` | 16 | 0,908 ± 0,009 | 0,961 ± 0,005 | +1,00 | +0,42 |
| `add:selected` (`amount_per_account_day` + `weekend`) | 10 | 0,903 ± 0,008 | 0,960 ± 0,005 | +0,53 | +0,23 |
| `drop:amount_usd` | 7 | 0,871 ± 0,017 | 0,927 ± 0,009 | −2,64 | −3,07 |
| `drop:quantity_purchased` | 7 | 0,887 ± 0,007 | 0,948 ± 0,005 | −1,04 | −0,90 |
| `drop:session_duration_seconds` | 7 | 0,853 ± 0,012 | 0,935 ± 0,007 | −4,51 | −2,27 |
| `drop:days_since_last_purchase` | 7 | 0,881 ± 0,009 | 0,951 ± 0,006 | −1,64 | −0,58 |
| `drop:account_age_days` | 7 | 0,869 ± 0,019 | 0,936 ± 0,008 | −2,89 | −2,09 |
| `drop:device_screen_resolution` | 7 | 0,896 ± 0,006 | 0,957 ± 0,005 | −0,14 | +0,01 |
| `drop:time_since_last_login_s` | 7 | 0,898 ± 0,011 | 0,957 ± 0,005 | +0,06 | −0,03 |
| `drop:items_viewed_before_purchase` | 7 | 0,907 ± 0,008 | 0,960 ± 0,005 | +0,90 | +0,28 |

**¿Qué features construidas ayudan y por qué?**

- Una sola neurona solo arma combinaciones lineales de sus entradas. Por eso una feature nueva suma capacidad solo si **no** es (casi) combinación lineal de las que ya están.
- `amount_per_account_day` es la única que ayuda sola (+0,79 pp de F1, +0,28 pp de AP). `ln(amount/(edad+1)) ≈ ln(amount) − ln(edad+1)`: `ln(amount)` ya es entrada, pero `account_age_days` entra cruda, así que `ln(edad)` es una transformación nueva que la neurona no podía formar. Además es la derivada más correlacionada con el fraude real (ρ = 0,768).
- Los otros cocientes (`amount_per_item`, `items_viewed_per_minute`, `viewed_per_purchased`) no ayudan, aunque `items_viewed_per_minute` correlacione 0,695 con el fraude. Correlación alta no alcanza: su información ya está en las entradas (sesión y artículos vistos) y el cociente agrega casi nada linealmente nuevo. Se confirma la hipótesis de que un log-cociente de columnas ya presentes aporta poco.
- La interacción explícita `amount_x_quantity`, que la neurona no puede formar sola, **no** ayuda (−0,15 pp): la hipótesis de que un producto suma capacidad útil queda refutada con estos datos.
- Las 7 derivadas juntas dan la mayor mejora (+1,00 pp de F1, +0,42 pp de AP, R² contra BigModel de 0,821 a 0,833), con el doble de entradas (16). El salto es del mismo orden que el desvío entre folds (0,009), así que es una mejora chica.

**¿Qué columnas se podrían descartar?** Según las filas `drop:`:

- `device_screen_resolution` y `time_since_last_login_s` se pueden sacar sin pérdida (−0,14 y +0,06 pp de F1, AP igual), como anticipaba su correlación casi nula.
- Sacar `items_viewed_before_purchase` **mejora** (+0,90 pp de F1, +0,28 pp de AP). Es la sorpresa del estudio: para una sola neurona, esa columna cruda parece agregar más ruido que señal. La causa no está aislada, y la mejora es del orden del desvío entre folds (0,008).
- `amount_usd`, `session_duration_seconds` y `account_age_days` son las columnas indispensables (sacarlas cuesta entre 2,6 y 4,5 pp de F1).

**¿Se confirma el descarte de `timestamp`?** Sí. `hour_of_day` y `weekend` tienen correlación prácticamente nula con el BigModel y con el fraude (|ρ| ≤ 0,011) y no mueven el CV (−0,15 y +0,09 pp de F1, dentro del ruido). Coincide con `timestamp_profile.csv`: la hora y el día no tienen señal.

**Recomendación.** La regla de selección (mejorar F1 y AP a la vez) elige `amount_per_account_day` y `weekend`, y queda guardada en `configs/fraud_constructed_features.toml`. El aporte de `weekend` es ruido (+0,09 pp con un desvío de 0,6 pp), y de hecho `add:selected` (+0,53 pp) rinde menos que `amount_per_account_day` sola (+0,79 pp). En la práctica se recomienda sumar solo `amount_per_account_day` y descartar `device_screen_resolution` y `time_since_last_login_s`. Las mejoras son de menos de 1 pp, del orden del desvío entre folds. Como control final, `generalize --config configs/fraud_constructed_features.toml --output output/fraud_constructed_features` da F1 0,883 en test contra 0,880 del modelo principal (AP 0,948 contra 0,950): no hay diferencia real. **El modelo principal del Ejercicio 1 (`configs/default.toml`, umbral 0,858) no se cambia.**

### Ejercicio 1: calibración

Un score está **calibrado** si `P(fraude | score = s) ≈ s`: de todas las transacciones con score 0,3, alrededor del 30% son fraude. `calibrate` mide la calibración del TinyModel contra `flagged_fraud` y la corrige de dos formas:

- **Platt scaling:** `p = σ(a · logit(s) + b)`, ajustado por Newton-Raphson sobre log-loss.
- **Regresión isotónica:** función escalonada no decreciente, ajustada con pool-adjacent-violators.

Los dos calibradores se ajustan **solo sobre las predicciones out-of-fold de desarrollo**, las mismas que eligen el umbral. El test no se usa para ajustar nada. Las métricas son:

- **Brier:** MSE entre la probabilidad y el resultado 0/1.
- **ECE:** gap medio ponderado entre la probabilidad media y la tasa observada, en 10 bins de igual ancho.

Resultados en test (1.126 transacciones, `calibration_metrics.csv`):

| Modelo | Brier | ECE | Average precision |
|---|---:|---:|---:|
| TinyModel sin calibrar | 0,1461 | 0,2966 | 0,9496 |
| TinyModel + Platt | **0,0216** | **0,0107** | 0,9496 |
| TinyModel + isotónica | 0,0224 | 0,0118 | 0,9410 |
| BigModel | 0,1575 | 0,3039 | 1,0000 |

- **Parámetros de Platt** (`platt_parameters.csv`): `a = 2,683`, `b = −5,082`. La isotónica queda con 20 escalones.
- `reliability_diagram.png` muestra la tasa observada contra la probabilidad media por bin. El TinyModel sin calibrar y el BigModel quedan muy por debajo de la diagonal: en test, el bin [0,7; 0,8) del BigModel tiene 0 fraudes en 76 transacciones, y el del TinyModel tiene 15,2% de fraude con un score medio de 0,743. Con Platt, los puntos se acercan a la diagonal. Los bins entre 0,2 y 0,9 tienen muy pocas filas (2 a 11 con Platt), así que son ruidosos.
- En test, el score medio del TinyModel es 0,417 y el del BigModel 0,422, pero la tasa real de fraude es 0,121. Con Platt, el score medio es 0,110.
- En desarrollo, el TinyModel sin calibrar tiene ECE 0,306 (out-of-fold). Los valores de Platt e isotónica en desarrollo son in-sample (`calibrator_in_sample = true`), por eso no se reportan como resultado.

**Decisiones con costos** (`calibration_decisions.csv`, test, con `cost_ratio = 20`: un fraude no detectado cuesta 20 revisiones manuales; costo = FP + 20·FN):

| Política | Umbral | TP | FP | TN | FN | Precision | Recall | Costo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TinyModel sin calibrar, umbral F1 | 0,858 | 110 | 4 | 986 | 26 | 0,965 | 0,809 | 524 |
| TinyModel sin calibrar, umbral de costo | 0,048 | 136 | 962 | 28 | 0 | 0,124 | 1,000 | 962 |
| TinyModel + Platt, umbral de costo | 0,048 | 130 | 84 | 906 | 6 | 0,607 | 0,956 | **204** |
| TinyModel + isotónica, umbral de costo | 0,048 | 132 | 124 | 866 | 4 | 0,516 | 0,971 | **204** |
| BigModel | 0,85 | 136 | 0 | 990 | 0 | 1,000 | 1,000 | 0 |

**¿Por qué sería apropiado calibrar en este caso?**

- **El TinyModel no aprende la probabilidad de fraude.** Se entrena con MSE para imitar los scores del BigModel, nunca contra `flagged_fraud`. En el mejor caso hereda la calibración del BigModel.
- **El BigModel tampoco está calibrado.** Separa perfectamente en 0,85, así que una transacción con score 0,8 tiene una frecuencia empírica de fraude de ~0, no de 0,8. Su ECE en test es 0,304. Es un ranking excelente (AP = 1), no una probabilidad.
- **La consigna pide una probabilidad** (0 = 0%, 1 = 100%), y hoy el número no significa eso: el score medio (0,417) es más del triple de la tasa real (0,121). El desbalance del 11,6% hace que un score que imita la forma del BigModel sobreestime mucho el fraude para la mayoría legítima. Además, la sigmoide satura en los extremos (3,7% de las salidas), donde el gradiente casi no corrige los errores.
- **Las decisiones por costo esperado necesitan probabilidades calibradas.** Conviene marcar si `p · C_fraude > (1 − p) · C_revisión`, o sea si `p > p* = C_revisión / (C_revisión + C_fraude) = 1/21 ≈ 0,048` para la razón 20. Esa fórmula solo sirve si `p` está calibrada:
  - con el score crudo, `p*` marca 1.098 de 1.126 transacciones y cuesta 962;
  - con Platt o isotónica, el mismo umbral teórico cuesta 204, **2,6 veces menos que el umbral F1** (524), sin buscar el umbral a mano;
  - el umbral 0,048 sobre Platt equivale a un score crudo de ≈ 0,685 (derivado de `a` y `b`), y el umbral F1 de 0,858 equivale a una probabilidad calibrada de ≈ 0,44. El número crudo no dice nada de eso.
- **Platt es monótona:** conserva el ranking y la average precision (0,9496 en ambos) y solo cambia el significado del número. Aplicada al logit, `σ(a·(w·x + c) + b) = σ((a·w)·x + (a·c + b))`: equivale a reescalar los pesos y el bias de la misma neurona. El TinyModel calibrado sigue siendo un perceptrón con los mismos 9 parámetros. La isotónica es más flexible pero crea empates, y baja un poco la AP (0,941).
- **Sin fuga de información:** el calibrador se ajusta sobre predicciones out-of-fold, como el umbral, y el test solo se usa para la evaluación final. Los pesos del TinyModel nunca ven `flagged_fraud`: la destilación sigue siendo contra el BigModel, y la etiqueta solo se usa, igual que antes, para el umbral y ahora para la calibración.
- **Producción:** la calibración deriva con el tiempo (cambia la tasa de fraude o el comportamiento). Se puede reajustar `a` y `b` con etiquetas recientes sin reentrenar el TinyModel.

### Ejercicios 2 y 3: robustez al ruido

**Protocolo.** `noise` evalúa modelos guardados sobre `digits_test.csv` con dos tipos de ruido:

- **Gaussiano:** `x' = clamp(x + σ·z, 0, 1)`, con `z ~ N(0, 1)` generado por Box–Muller.
- **Sal y pimienta:** una fracción de píxeles pasa a 1 (tinta) o 0 (fondo) con igual probabilidad.

La semilla es fija (42) y cada fila tiene su propio stream. Así, **todos los modelos ven exactamente las mismas imágenes ruidosas**, y los mismos números aleatorios se reusan en todos los niveles (las curvas son suaves y los modelos comparables). La fila σ = 0 reproduce exactamente la accuracy de `evaluate`.

Se comparan tres modelos:

- `exercise3`: el seleccionado, con augmentation;
- `exercise3_no_augmentation`: el paso t4 de la escalera, igual al anterior pero sin augmentation. `configs/exercise3_no_augmentation.toml` lo entrena; gana t4 con 96,63% de validación contra 95,90% de la línea de base, y en test da 96,80%;
- `exercise2`: el ganador del Ejercicio 2.

Accuracy en test (`noise_robustness.csv`, `noise_accuracy.png`):

| Ruido gaussiano σ | 0 | 0,1 | 0,2 | 0,3 | 0,5 | 0,7 | 1,0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exercise3` (con augmentation) | **98,64%** | 97,24% | 77,97% | 52,62% | 36,28% | 27,19% | 20,99% |
| `exercise3_no_augmentation` | 96,80% | 96,48% | **93,31%** | **79,26%** | **52,54%** | **35,92%** | **24,67%** |
| `exercise2` | 86,86% | 85,90% | 79,05% | 61,03% | 33,40% | 19,26% | 13,18% |

| Sal y pimienta (fracción) | 0 | 0,05 | 0,1 | 0,2 | 0,3 | 0,5 |
|---|---:|---:|---:|---:|---:|---:|
| `exercise3` (con augmentation) | **98,64%** | **97,40%** | 90,87% | 59,31% | 41,61% | 26,11% |
| `exercise3_no_augmentation` | 96,80% | 95,96% | **92,55%** | **77,77%** | **59,79%** | **34,40%** |
| `exercise2` | 86,86% | 85,42% | 81,22% | 64,56% | 44,73% | 20,38% |

**Clases que se degradan primero** (columnas `recall_*`):

- Caen primero los dígitos de trazo fino: con σ = 0,2, el modelo del Ejercicio 3 mantiene 99,2% en 0 y 2, 97,6% en 3 y 95,9% en 8, pero baja a 59,4% en 1, 49,4% en 7 y 46,8% en 9 (72,2% en 4). Con σ = 0,3, el recall del 1 es 2,1%.
- El ruido agrega "tinta" y empuja las predicciones hacia dígitos con mucha tinta: en `noisy_examples.png`, las predicciones erradas con σ alto son mayormente 2, 0, 8 y 5.

**¿Es posible afirmar que el modelo es robusto al ruido?** **No.**

- El modelo del Ejercicio 3 cae por debajo del 98% ya con σ = 0,1 (97,24%) y por debajo del 90% con σ = 0,2 (77,97%). Con sal y pimienta, baja de 98% con el 5% de píxeles corruptos y queda en 59,31% con el 20%.
- Es robusto solo a ruido muy leve. `noisy_examples.png` muestra que con σ = 0,2 los dígitos siguen siendo claros para una persona y el 9 del ejemplo ya se predice como 8 (el 1 y el 2 de los ejemplos ya estaban mal sin ruido).
- **Efecto del clamping:** el fondo es 0, así que el ruido negativo se recorta y el positivo no. El fondo gana en promedio `E[max(0, σz)] ≈ 0,4·σ` de brillo (y la tinta pierde lo mismo). El ruido no es de media cero en la imagen: agrega un velo gris que el modelo nunca vio en entrenamiento, donde el fondo es exactamente 0.
- **Con y sin augmentation.** La augmentation geométrica enseña invariancia a traslaciones y rotaciones, no a ruido de píxel, y se confirma con los números: **el modelo sin augmentation es bastante más robusto**. Con σ = 0,2 retiene 93,31% contra 77,97%, y con 20% de sal y pimienta 77,77% contra 59,31%. El modelo con augmentation solo gana sin ruido o con ruido muy leve (σ ≤ 0,1, fracción ≤ 0,05).
  - Una hipótesis, no medida: el modelo con augmentation entrenó 51 épocas contra 6, y tiene fronteras más confiadas (loss en test sin ruido 0,047 contra 0,117). Pequeñas perturbaciones fuera de la distribución de entrenamiento lo sacan de la clase correcta con más facilidad.
  - Para ganar robustez al ruido habría que entrenar con ruido (augmentation de píxel), no con más transformaciones geométricas.
- El modelo del Ejercicio 2 arranca con un techo del 86,86% incluso con σ = 0, porque nunca vio un 8 (recall del 8 = 0% en todos los niveles).

### Ejercicios 2 y 3: interpretabilidad (métodos de atribución)

`attribution` explica el logit `z_c` de una clase `c` con tres métodos:

| Método | Fórmula | Qué muestra |
|---|---|---|
| Saliency | `\|∂z_c / ∂x\|` | Qué píxeles, al cambiar, mueven más el logit (con o sin tinta). |
| Gradiente × input | `x ⊙ ∂z_c / ∂x` | Aporte lineal de la tinta presente; con signo. |
| Integrated Gradients (IG) | `IG_i = x_i · (1/m) Σ_k ∂z_c/∂x_i((k + 0,5)/m · x)`, baseline negro `x' = 0`, `m = 50` pasos (regla del punto medio) | Aporte de cada píxel sobre el camino desde la imagen negra; cumple completitud: `Σ_i IG_i = z_c(x) − z_c(0)`. |

**El gradiente respecto de la entrada reusa el backprop del entrenamiento.** Después del forward, en lugar de la delta `(P − Y)/B` de cross-entropy, se siembra un vector one-hot en el logit de la clase `c`. Se propaga con los mismos productos `D·W` y factores `f'(A)`, un paso más abajo que en el entrenamiento: en la capa 0, `D·W₀ = ∂z_c/∂x` (`backward_input_batch` en `backprop.rs`). No se toca el backward del entrenamiento. Un test compara este gradiente contra diferencias finitas centrales (tanh y sigmoide, error < 1e-6).

**Mapas medios por clase** (`attribution_class_means.png`, 100 dígitos de test bien clasificados por clase; filas: imagen media, saliency, gradiente × input, IG; rojo = evidencia a favor, azul = en contra):

- **1:** evidencia positiva casi solo en el trazo vertical. Es la clase con menos evidencia negativa (suma de IG negativa de −0,85 contra +12,09 positiva).
- **7:** positiva en la barra horizontal superior y el comienzo del trazo diagonal.
- **0:** el anillo completo es positivo, con azul en parte del borde interior (abajo a la derecha): la tinta que se mete hacia el agujero resta evidencia. Es la clase con más evidencia negativa (−4,57).
- **3, 9, 8:** la evidencia positiva se concentra en los tramos que los distinguen (los extremos y la curva media del 3, el lazo superior del 9, el cruce central del 8). Los tramos compartidos con otros dígitos mezclan rojo y azul; en el 9, la parte baja del trazo es azul.
- **Gradiente × input e IG** dan mapas muy parecidos. **Saliency** es más difusa y ruidosa, y la mayoría de su masa cae **fuera de la tinta**: entre el 55% (clases 2 y 5) y el 75% (clase 1) de la saliency media está en píxeles donde la imagen media vale menos de 0,05. Son píxeles donde "agregar tinta" cambiaría la decisión.

**Análisis de errores** (`attribution_examples.png`: 5 dígitos correctos y 5 mal clasificados; columnas: imagen, saliency, gradiente × input e IG para la clase predicha, e IG para la clase verdadera):

- Fila 2, un 1 predicho como 2: el IG hacia el 2 es positivo en el "pie" horizontal del 1 (que se parece a la base de un 2) y negativo en el trazo vertical.
- Fila 16, un 2 muy inclinado predicho como 7: el trazo superior horizontal es la evidencia del 7.
- Fila 349, un 6 abierto predicho como 5: la barra superior aporta al 5 (rojo en IG hacia el 5) y resta al 6 (azul en IG hacia el 6).
- En los errores, comparar IG(predicha) con IG(verdadera) muestra qué trazo "votó" por la clase equivocada.

**Completitud** (`attribution_completeness.csv`, las 1.020 computaciones de IG de la corrida):

- En el modelo real (ReLU, 50 pasos), el gap relativo `|Σ IG − (z_c(x) − z_c(0))| / |z_c(x) − z_c(0)|` tiene media 0,37% y máximo 4,03%. Supera el 1% en 56 casos y el 2% en 4.
- Con ReLU el gradiente es constante por tramos a lo largo del camino y la regla del punto medio no es exacta. Más pasos achican el gap.
- La garantía está en el test unitario: con una red tanh suave y 300 pasos, la completitud se cumple con error relativo < 1e-3.

**Limitaciones:**

- Con baseline negro, un píxel sin tinta tiene `x_i = 0`, así que su IG y su gradiente × input son **exactamente cero**. Estos métodos no pueden mostrar "evidencia por ausencia de tinta" (por ejemplo, que el centro vacío del 0 es lo que lo distingue del 8); la saliency sí la muestra.
- La saliency es ruidosa: refleja la sensibilidad local de una red ReLU, que cambia de región lineal píxel a píxel.
- Las atribuciones explican **el modelo, no los datos**: dicen qué usa la red para decidir, no qué causa el dígito. Un patrón espurio aprendido aparecería igual de "importante".
- `exercise2 attribution` corre igual. Para el 8 no hay mapa medio porque el modelo del Ejercicio 2 no clasifica bien ningún 8.
