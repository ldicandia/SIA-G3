# TP4 : Aprendizaje no supervisado

Resolución del enunciado [SIATP42026.pdf](docs/SIATP42026.pdf) con los datos suministrados y las técnicas de las [clases](clases/). PCA usa scikit-learn; Kohonen, Oja y Hopfield están implementados con NumPy.

Las [referencias teóricas](docs/referencias-teoricas.md) indican las páginas de clase y las convenciones de implementación. Los resultados están en [Ejercicio 1](docs/hallazgos/ej1.md), [Ejercicio 2](docs/hallazgos/ej2.md) y la [presentación](docs/presentacion/TP4-presentacion.pptx).

## Instalación

Python 3.11 o superior, desde `TP4/` (macOS, Linux o WSL):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Para decodificar los QR originales, las consultas y las salidas de Hopfield:

```bash
.venv/bin/pip install -r requirements-qr.txt
```

## Ejecución

Cada comando escribe tablas, gráficos, métricas y la configuración utilizada en `runs/<experiment>/`. Las rutas de datos de las configuraciones se resuelven desde `TP4/`. `runs/` no se versiona.

```bash
.venv/bin/python -m tp4 run configs/eda.json
.venv/bin/python -m tp4 run configs/pca.json
.venv/bin/python -m tp4 run configs/pca_scaling_sweep.json
.venv/bin/python -m tp4 run configs/som_sweep.json
.venv/bin/python -m tp4 run configs/som_design_sweep.json
.venv/bin/python -m tp4 run configs/som.json
.venv/bin/python -m tp4 run configs/oja.json
.venv/bin/python -m tp4 run configs/campaigns.json
.venv/bin/python -m tp4 run configs/hopfield_letters.json
.venv/bin/python -m tp4 run configs/hopfield_qr.json
```

Las configuraciones fijan semilla, datos e hiperparámetros. `plots: false` permite omitir imágenes; `--out-root DIR` cambia la carpeta de salida. Las claves desconocidas se rechazan. Las corridas se preparan en una carpeta temporal y reemplazan una corrida anterior solo al completarse correctamente.

- **EDA/PCA:** descriptivos, distribuciones, boxplots, correlaciones, selección de variables, estandarización, varianza explicada, cargas y biplot.
- **Kohonen:** ganadora por distancia euclídea, vecindad escalón, aprendizaje decreciente, impactos, matriz U y mapas de variables. Los barridos cambian un hiperparámetro por vez respecto de una base común; el informe justifica la elección final.
- **Oja:** índice de una neurona sobre las mismas variables estandarizadas y comparación con PC1.
- **Campañas:** perfiles de PCA/Kohonen/Oja, decisiones con supuestos comerciales explícitos y comparación con seguidores.
- **Hopfield:** pesos de Hebb, estados bipolares, diagonal nula, recuperación, energía, ruido, ortogonalidad, capacidad y estados espurios en letras y QR.

## Repeticiones y barras de error

Kohonen, Oja y los experimentos de ruido de Hopfield usan `params.n_repeats: 30`, con semillas 42 a 71. Cada configuración de los barridos de Kohonen usa esas mismas semillas. Las tablas conservan los resultados por semilla y sus medias y desvíos estándar muestrales; las barras y bandas muestran **media ± un desvío estándar entre semillas**.

En Kohonen y Oja cada semilla cambia el entrenamiento. En Hopfield los pesos son deterministas: cambian las consultas ruidosas y el orden de actualización. Las imágenes de grilla y las trazas individuales corresponden a la semilla 42, indicada como ejemplo. EDA, PCA, las reglas comerciales y los puntos fijos de Hebb se calculan sobre datos fijos; repetirlos con otras semillas daría el mismo resultado.

## Pruebas y presentación

```bash
.venv/bin/python -m pytest -q
```

Las pruebas cubren modelos, preprocesamiento y ejecución de los experimentos; generan sus resultados en carpetas temporales.

Para reconstruir el PowerPoint después de ejecutar los experimentos:

```bash
.venv/bin/pip install python-pptx
.venv/bin/python docs/presentacion/build_deck.py
```

El generador toma las métricas y figuras de `runs/`. Las figuras del informe se guardan en `docs/hallazgos/figuras/`.
