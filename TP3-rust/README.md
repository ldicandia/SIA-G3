# TP3 - Perceptrons in Rust

Implementación educativa de perceptrones y del ejercicio de *knowledge distillation* para detección de fraude. El proyecto privilegia un core explícito y legible: no usa frameworks de machine learning y deja visibles el forward pass, la regla de actualización y backpropagation.

Todo el código, las interfaces, la configuración y los artefactos generados están en inglés. Este README está en español para responder directamente la consigna.

## Requisitos y datos

- Rust 1.82 o posterior.
- El archivo `fraud_dataset.csv` provisto por la materia.
- El dataset no se copia ni se versiona en este repositorio.

El loader exige el esquema documentado de once columnas. Nueve se usan como features; `big_model_fraud_probability` es el target de entrenamiento y `flagged_fraud` queda estrictamente reservado para la evaluación y la selección del umbral.

## Ejecución

Desde este directorio:

```bash
cargo run --release -- run-all \
  --data "/path/to/fraud_dataset.csv" \
  --config configs/default.toml \
  --output output
```

También se pueden ejecutar las etapas por separado:

```bash
cargo run --release -- inspect --data "/path/to/fraud_dataset.csv" --output output
cargo run --release -- compare --data "/path/to/fraud_dataset.csv" --output output
cargo run --release -- generalize --data "/path/to/fraud_dataset.csv" --output output
```

### Cómo entrenar

`compare` entrena los perceptrones lineal y sigmoide usando todas las muestras para estudiar su capacidad de aprendizaje. `generalize` entrena el TinyModel sigmoide con el split train/validation/test y realiza la selección de learning rate y cantidad de épocas. `run-all` ejecuta ambos entrenamientos, además de la exploración del dataset.

Para imprimir la pérdida de cada época mientras se entrena, compilar con el feature `training-logs`:

```bash
cargo run --release --features training-logs -- generalize \
  --data "/path/to/fraud_dataset.csv" \
  --config configs/default.toml \
  --output output
```

La salida tiene este formato:

```text
model=single-layer learning_rate=0.01 epoch=1 train_loss=0.0182345678 validation_loss=0.0187654321
model=single-layer learning_rate=0.01 epoch=2 train_loss=0.0141234567 validation_loss=0.0146543210
```

El feature sólo controla la impresión. El entrenamiento, las métricas y los artefactos son idénticos con o sin él.

La pérdida a lo largo de las épocas es el diagnóstico principal del entrenamiento. Siempre se generan:

- `learning_curves.png`: MSE por época para comparar el perceptrón lineal con el sigmoide.
- `generalization_learning_curve.png`: MSE de train y validation por época para detectar convergencia y separación entre ambos conjuntos.
- `learning_history.csv`: los mismos valores numéricos para análisis adicional.

Las validaciones sugeridas por la consigna viven fuera del pipeline principal:

```bash
cargo run --release --example validate_all
```

Ese ejemplo valida AND, `y=x`, `y=tanh(x)`, XOR con `[2,2,1]` y `[2,3,2,1]`, y demuestra que un perceptrón escalón simple no resuelve XOR.

## Diseño

El crate separa responsabilidades:

- `matrix`, `loss` y `model` contienen el core matemático.
- `training` implementa SGD online, early stopping y backpropagation.
- `data` carga y valida el CSV y ajusta el `StandardScaler`.
- `split` genera una división reproducible estratificada por bandas del target del BigModel.
- `metrics` contiene métricas puras de regresión, clasificación y selección de umbral.
- `experiment` orquesta las corridas y genera CSV y PNG.

`DenseMatrix` guarda los valores contiguos en orden row-major y presta cada muestra como `&[f64]`. El entrenamiento no clona filas: reutiliza el vector de índices, los buffers de activaciones y los deltas. La matriz normalizada se materializa una sola vez por etapa y se comparte entre todas las configuraciones candidatas.

`Loss` es un trait para que el objetivo pueda reemplazarse sin modificar los modelos. Esta versión implementa solamente `MeanSquaredError`, como se definió para el alcance del trabajo.

## Exploración del dataset

La corrida verificada contiene:

- 7.500 transacciones y nueve features.
- Cero valores faltantes y cero filas duplicadas exactas.
- 869 fraudes confirmados, equivalentes al 11,59% de las muestras.
- Features con escalas muy distintas y varias distribuciones sesgadas, por lo que la estandarización es necesaria.
- Probabilidades del BigModel en `[0,1]`; el label real no se incorpora a las entradas.

`inspect` produce `dataset_summary.csv`, `data_profile.csv`, `feature_distributions.png` y `target_distributions.png` para que estas propiedades puedan verificarse en cualquier nueva copia del dataset.

## Comparación de aprendizaje

Esta etapa usa todas las muestras, ajusta el scaler con todas ellas y entrena ambos modelos con MSE. Cada activación prueba la misma grilla de learning rates y conserva su mejor resultado; así no se confunde capacidad del modelo con un learning rate desfavorable.

| Modelo | MSE | RMSE | R² | Rango de salida |
|---|---:|---:|---:|---:|
| Lineal | 0,02611 | 0,16158 | 0,71461 | `[-0,391; 2,093]` |
| Sigmoide | 0,01087 | 0,10425 | 0,88120 | `[0,002; 1,000]` |

Conclusiones:

- El perceptrón lineal presenta mayor underfitting: se estabiliza con un error claramente superior y genera valores inválidos como probabilidades.
- Ambos modelos alcanzan un plateau, evidencia de la capacidad limitada de una única neurona. El sigmoide aprovecha mejor la estructura no lineal, aunque también conserva error residual.
- Se selecciona el perceptrón sigmoide para generalización porque logra menor error y respeta por construcción el rango probabilístico.

Los datos exactos quedan en `learning_trials.csv`, `learning_summary.csv` y `learning_history.csv`; los gráficos asociados son `learning_curves.png` y `prediction_comparison.png`.

## Estudio de generalización

Se usa un split determinista 70/15/15, con semilla 42 y estratificación por bandas de la probabilidad del BigModel. El scaler se ajusta únicamente con train durante la búsqueda.

No se elige un “training set afortunado”. Train ajusta los pesos, validation elige learning rate, época y umbral, y test se consulta una sola vez. Luego de elegir la configuración, los pesos se vuelven a entrenar con train+validation durante la cantidad de épocas seleccionada y se evalúan sobre test.

Las métricas contra el BigModel son MSE, RMSE y R²: miden fidelidad de destilación desde ángulos complementarios. Contra `flagged_fraud` se informan precision, recall, accuracy y matriz de confusión. Precision indica qué proporción de las transacciones marcadas era realmente fraudulenta; recall indica qué proporción de los fraudes reales fue detectada; accuracy indica qué proporción de todas las decisiones fue correcta.

La corrida verificada seleccionó:

- Perceptrón simple con activación sigmoide y MSE.
- Learning rate `0.01`.
- Seis épocas para el refit con train+validation.
- Umbral `0.90511`, elegido minimizando la cantidad total de errores en validation, o equivalentemente maximizando accuracy.

Resultados sobre las 1.124 muestras de test:

| Métrica | Valor |
|---|---:|
| MSE / RMSE | 0,01087 / 0,10426 |
| R² | 0,88121 |
| Precision / Recall | 0,91736 / 0,86047 |
| Specificity / Accuracy | 0,98995 / 0,97509 |
| TP / FP / TN / FN | 111 / 10 / 985 / 18 |

El umbral recomendado no debe confundirse con `0.5` ni copiarse del BigModel: fue determinado para las salidas del TinyModel. Sin costos de negocio explícitos, se elige el umbral que produce menos clasificaciones incorrectas en validation. `threshold_sweep.csv` conserva precision, recall y accuracy para que CompanyX pueda inspeccionar otros puntos operativos.

Por decisión de alcance, no se serializan los pesos ni el scaler. `selected_model.csv` identifica la mejor configuración y `test_metrics.csv` contiene la evaluación final.

## Tests y calidad

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Los tests se concentran en unidades pequeñas y críticas: MSE, derivadas de activación, matriz, scaler, split, métricas, desempate del umbral y actualizaciones que deben reducir la pérdida. No se duplica el pipeline completo en tests end-to-end.

## Fuera de alcance

ReLU, feature engineering, calibración probabilística y serialización del modelo quedan deliberadamente fuera de esta primera versión. La modularidad de `Activation` y `Loss` permite agregarlos sin reescribir el entrenamiento.
