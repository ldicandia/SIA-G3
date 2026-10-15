# SIA TP4 — Métodos de Aprendizaje No Supervisado

Implementación en Python del TP4 de "Sistemas de Inteligencia Artificial" (ITBA): análisis
exploratorio y PCA de `datos/creadores.csv`, un mapa de Kohonen escrito a mano, la regla de Oja
y una red de Hopfield. Cada análisis se corre con **un solo comando** a partir de un archivo de
configuración con semilla fija, y deja sus resultados y gráficos en `runs/<experimento>/`.

Esta primera etapa deja lista la infraestructura (CLI, configs, directorios de corrida, tests)
y el **análisis exploratorio (EDA)** del dataset.

## Instalación

Python corre sobre un entorno virtual de **WSL**, nunca sobre uno de Windows (los venvs de
Windows ya se rompieron antes en este repo). Abrí una shell de WSL dentro del directorio `TP4/`
y ejecutá:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Si estás en Windows, anteponé a cada comando de este README `wsl -e bash -lc '...'`, por ejemplo:

```
wsl -e bash -lc '.venv/bin/python -m pytest -q'
```

Todos los comandos usan `.venv/bin/python` para que se puedan copiar y pegar sin activar el
entorno. No uses el venv de Windows.

## Correr un experimento

```bash
.venv/bin/python -m tp4 run configs/eda.json
```

El comando lee la config, valida todas sus claves (las desconocidas se rechazan), corre el
experimento y publica `runs/eda/`. Opcionalmente `--out-root DIR` cambia el directorio que
recibe `runs/`. Si la config o los datos son inválidos, el comando imprime `error: ...` por
stderr, sale con código 2 y **no toca** la corrida anterior.

Las configs son JSON (o TOML, por extensión `.toml`) con estas claves comunes:

| Clave | Significado |
|-------|-------------|
| `experiment` | nombre de la corrida; define el directorio `runs/<experiment>/` (`[a-z0-9_-]`, hasta 64 caracteres) |
| `kind` | tipo de experimento (por ahora `eda`) |
| `seed` | semilla entera obligatoria; misma semilla, mismos números |
| `backend` | `numpy` (por defecto) o `cupy` (opcional, ver "Backend CUDA opcional") |
| `plots` | `true`/`false`: escribir o no los PNG |

Para `kind: "eda"` además: `data` (ruta al CSV, relativa a `TP4/`) y `params.outlier_iqr_k`
(multiplicador de las cercas de Tukey, 1.5 por defecto).

Una corrida reemplaza por completo a otra con el mismo `experiment`, pero solo si esa carpeta
contiene un `run.json` de tp4; una carpeta ajena se rechaza sin tocarla.

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Artefactos de `runs/eda/`

| Archivo | Qué muestra |
|---------|-------------|
| `run.json` | procedencia: config validada, semilla, backend, versiones de paquetes, git sha, lista de artefactos |
| `summary.json` | métricas escalares (filas, columnas, outliers, correlación máxima) |
| `columns.csv` | una fila por columna del CSV: tipo, faltantes, únicos, media, desvío, cuartiles, RIC, cercas, cantidad de outliers y asimetría |
| `outliers.csv` | cada outlier de Tukey: `column,creator_id,name,value,lower_fence,upper_fence,side` |
| `correlation.csv` | matriz de correlación de Pearson entre las columnas numéricas |
| `distributions.png` | histograma de cada variable con las cercas de Tukey punteadas |
| `boxplots_standardized.png` | boxplots de las variables estandarizadas (los outliers coinciden con `outliers.csv`) |
| `scale_ranges.png` | desvío estándar de cada variable en escala logarítmica: muestra la brecha de escalas que motiva estandarizar |
| `correlation_heatmap.png` | mapa de calor de la correlación entre variables |

Los outliers solo se **marcan y reportan**: ningún creador se elimina ni se modifica.

`runs/` se regenera a partir de las configs y no se commitea (está en `.gitignore`).

## Barridos en paralelo

`kind: "sweep"` corre otro experimento (`base`) sobre una grilla de valores de config (`grid`),
en procesos de CPU paralelos (`workers`, con el método `spawn`). Claves:

| Clave | Significado |
|-------|-------------|
| `base` | config a barrer (ruta relativa a `TP4/`); no puede ser otro `sweep` |
| `grid` | `{"clave.con.puntos": [valores...]}`; se corre el producto cartesiano, en el orden en que escribas las claves |
| `workers` | cantidad de procesos (entero ≥ 1, por defecto 1) |
| `child_plots` | `true`/`false`: escribir los PNG de cada corrida hija (por defecto `false`) |

Cada punto de la grilla es una corrida completa en `runs/<experimento>/<hijo>/` (con su
`run.json` y `summary.json`), y `runs/<experimento>/sweep.csv` tiene una fila por hijo con las
claves de la grilla y las métricas escalares. El orden de las filas y el contenido no dependen
de la cantidad de workers. Ejemplo, PCA estandarizada contra PCA sin estandarizar:

```bash
.venv/bin/python -m tp4 run configs/pca_scaling_sweep.json
```

Una grilla vacía, un valor repetido, más de 512 hijos o una `base` inválida se rechazan antes
de correr nada.

## Mapa de Kohonen (`kind: "som"`)

```bash
.venv/bin/python -m tp4 run configs/som.json
```

El mapa autoorganizado está escrito **a mano en numpy** (`tp4/models/som.py`); no se usa ninguna
librería de SOM ni de clustering, y un test lo verifica mirando los imports de `tp4/`. Entrena
sobre las **25 variables seleccionadas y estandarizadas**, tal como se decidió en
[`docs/hallazgos/ej1.md`](docs/hallazgos/ej1.md) (no sobre componentes principales). Todos los
hiperparámetros salen de `params` en `configs/som.json`:

| Parámetro | Significado | Valores permitidos |
|-----------|-------------|--------------------|
| `rows` | filas de la grilla | entero >= 1 (`rows * cols` entre 2 y 400) |
| `cols` | columnas de la grilla | `null` (= `rows`, grilla cuadrada) o entero >= 1 |
| `topology` | forma de las celdas | `"hex"` (6 vecinos) o `"rect"` (4 vecinos) |
| `neighborhood` | influencia del ganador sobre sus vecinas | `"gaussian"` (`exp(-d²/2R²)`) o `"step"` (1 si `d <= R`, 0 si no) |
| `init` | inicialización de los pesos | `"sample"` (creadores al azar) o `"random"` (uniforme por variable) |
| `epochs` | pasadas sobre los creadores | entero entre 1 y 5000 |
| `eta0` | tasa de aprendizaje inicial | `0 < eta_min <= eta0 <= 1` |
| `eta_min` | piso de la tasa de aprendizaje | `0 < eta_min <= eta0` |
| `eta_decay` | cómo baja la tasa de `eta0` a `eta_min` | `"exponential"`, `"linear"`, `"inverse"` (familia 1/i de la teoría) o `"constant"` |
| `radius0` | radio de vecindad inicial (unidades de grilla) | número > 0 |
| `radius_min` | piso del radio (el ganador siempre se actualiza) | `0 < radius_min <= radius0` |
| `radius_decay` | cómo baja el radio de `radius0` a `radius_min` | igual que `eta_decay` |
| `n_regions` | cantidad de regiones del mapa (ver más abajo) | `null` (automática) o entero entre 1 y `rows * cols` |

Una config inválida (valores fuera de rango, opciones desconocidas, parámetros que no existen)
sale con `error: ...` y código 2 antes de crear `runs/som/`.

| Artefacto en `runs/som/` | Contenido |
|--------------------------|-----------|
| `bmus.csv` | una fila por creador: neurona ganadora, fila y columna, distancia al ganador, segunda ganadora y si es adyacente |
| `neurons.csv` | una fila por neurona: posición, cantidad de impactos (`hits`) y creadores que cayeron ahí |
| `weights.csv` | vector de pesos de cada neurona (unidades estandarizadas) |
| `history.csv` | por época: `eta`, `radius`, error de cuantización y error topográfico |
| `hit_map.png` | mapa de impactos con el nombre de cada creador dentro de su neurona |
| `training_curve.png` | errores y parámetros (`eta`, radio) a lo largo del entrenamiento |
| `u_matrix.png` | matriz U: distancia media de cada neurona a sus vecinas (con la cantidad de impactos y las fronteras de las regiones) |
| `u_edges.csv` | todos los pares de neuronas vecinas con su distancia, los creadores de cada lado y las 3 variables que más difieren, de la pared más fuerte a la más débil |
| `prototypes.csv` | prototipo de cada neurona en unidades originales (`media + w * desvío`), más `topic_art_pct_rec` = 100 - fútbol - maquillaje - perros |
| `component_planes.png` | un plano de componentes por variable seleccionada (unidades originales) y uno para el arte reconstruido |
| `pc_planes.png` | PC1 y PC2 de cada prototipo (mismo signo que `runs/pca`) |
| `region_scan.csv` | para cada k de 2 a 8: razón de distancias frontera/interior, mínimo de creadores por región y regiones sin creadores |
| `regions.csv` | una fila por región: neuronas, creadores, mezcla de tópicos, 3 variables más altas y más bajas, promedio de PC1 y PC2 |
| `region_profiles.csv` | perfil largo región x variable: media z y media en unidades originales de los creadores de la región |
| `region_profiles.png` | mapa de calor del perfil de cada región (media z) |
| `region_map.png` | mapa con las regiones coloreadas, sus bordes y los nombres de los creadores |
| `biplot_regions.png` | PC1 vs PC2 de cada creador coloreado por región del mapa |
| `summary.json`, `run.json` | métricas finales y registro de la corrida |

Métricas de calidad (en `summary.json`):

- **QE (error de cuantización):** distancia euclídea media entre cada creador y los pesos de su neurona ganadora.
- **TE (error topográfico):** fracción de creadores cuya primera y segunda ganadora no son vecinas en la grilla.
- **Preservación de distancias:** correlación de Pearson entre la distancia en el espacio de entrada de cada par de creadores y la distancia en la grilla entre sus neuronas ganadoras (`null` si no está definida).

Con la misma semilla, `summary.json`, `bmus.csv`, `weights.csv` y `history.csv` salen idénticos byte a byte.

### Regiones del mapa

Una **región** es un grupo de neuronas contiguas: se aglomeran los prototipos con el método de
Ward (costo `|A||B| / (|A| + |B|) * ||media_A - media_B||²`), fusionando solo regiones que
comparten al menos un par de neuronas vecinas. Así cada región es literalmente una zona de la
grilla, está numerada de 1 a k por su neurona de menor índice y las mismas pesas dan siempre las
mismas etiquetas. Los empates de costo van al par con menor (neurona mínima de A, neurona mínima de B).

`params.n_regions` fija k. Con `null` se elige k así (regla registrada en `region_scan.csv`): entre
k = 2..8 con al menos 3 creadores en cada región, el de mayor razón `u_border_ratio` (distancia
media entre neuronas vecinas de regiones distintas / distancia media dentro de las regiones); los
empates van al k menor y, si ningún k califica, se usa k = 2. Las regiones sin creadores (solo
posibles con un `n_regions` explícito) tienen `n_creators` 0 y celdas vacías en los perfiles.

En `summary.json`: `n_regions`, `n_regions_auto`, `region_creator_counts`, `u_border_ratio` y
`pc1_region_eta2` / `pc2_region_eta2` (razón de correlación eta² = SS_entre / SS_total de los
puntajes de cada creador en PC1 y PC2 agrupados por región: cuánto de cada componente explican las regiones).
Ninguna de estas lecturas usa el generador aleatorio, así que no cambian el entrenamiento.

## Barrido de Kohonen y configuración final

`configs/som.json` **no se edita a mano**: guarda los valores que eligieron dos barridos
registrados. La elección la hace una única regla determinista
(`tp4/analysis/som_selection.py`, que es el único lugar donde viven los umbrales):

> Se descartan los mapas con más de 0.5 de neuronas vacías; entre los restantes se conservan los
> de error topográfico a lo sumo 0.02 por encima del mejor y los de preservación de distancias a
> lo sumo 0.05 por debajo de la mejor; de esos se elige el de menor error de cuantización
> (empates: menos neuronas, luego menos épocas, luego el primero del barrido).

Los umbrales son inclusivos (un valor justo en el límite pasa). Si ningún mapa cumple la
ocupación, la regla usa todos y lo deja registrado (`occupancy_fallback`). La preservación de
distancias se suma al QE y al TE porque con 50 creadores el TE casi no distingue mapas buenos.

Se barre en dos etapas, en este orden:

1. **Diseño** (`configs/som_design_sweep.json`): topología x vecindad x inicialización
   (8 hijos) sobre la config de referencia congelada `configs/som_design_base.json`
   (6x6, eta0 0.5, radio 3, 200 épocas). Su selección fija `topology`, `neighborhood` e `init`
   de `configs/som.json`. Nota: el error topográfico considera vecinas a las neuronas a distancia
   de grilla <= 1 (4 vecinas en `rect`, 6 en `hex`), así que `rect` se juzga con más rigor.
2. **Hiperparámetros numéricos** (`configs/som_sweep.json`): tamaño de la grilla cuadrada
   (`params.rows`), `eta0`, `radius0` y `epochs` (216 hijos) sobre `configs/som.json` ya con el
   diseño elegido. Su selección fija `rows`, `eta0`, `radius0` y `epochs`.

| Artefacto en `runs/som_sweep/` y `runs/som_design_sweep/` | Contenido |
|-----------------------------------------------------------|-----------|
| `sweep.csv` | una fila por hijo con QE, TE, preservación de distancias y fracción de neuronas vacías |
| `selection.csv` | las mismas métricas más las etapas de la regla, el ranking de candidatos y la columna `selected` |
| `selected_params.json` | hijo elegido y valor de cada clave de la grilla |
| `sweep_metrics.png` | cada métrica contra cada hiperparámetro (media y rango min-max), con el valor elegido marcado |
| `sweep_tradeoff.png` | QE contra preservación de distancias, con el mapa elegido destacado |
| `summary.json` | `selected_*`, cantidades que pasan cada etapa y los umbrales de la regla |

Para regenerar todo, en este orden (diseño, hiperparámetros, mapa final) y verificar que
`configs/som.json` coincide con lo elegido:

```bash
.venv/bin/python -m tp4 run configs/som_design_sweep.json
.venv/bin/python -m tp4 run configs/som_sweep.json
.venv/bin/python -m tp4 run configs/som.json
.venv/bin/python -m tp4.experiments.som_sweep_report check configs/som.json   runs/som_design_sweep runs/som_sweep --run runs/som
```

El comando de verificación imprime `MISMATCH`, `MISSING` o `METRIC` por cada diferencia y sale con
código 1; si todo coincide imprime `ok: ...` y sale con 0. Si querés cambiar la regla, cambiala en
`tp4/analysis/som_selection.py` y volvé a correr los barridos; nunca edites `configs/som.json`
hacia un mapa que la regla no eligió. (Python puede imprimir un `RuntimeWarning` de `runpy` al
usar `-m`; es inofensivo.)

## Backend CUDA opcional

Con `"backend": "cupy"` en la config, los cálculos de arrays que usan `ctx.backend.xp` (por ahora
la verificación por autovalores de la PCA) corren en la GPU. Es **opcional**: nada de este
repo ni ningún test necesita CUDA, y `numpy` es el backend por defecto. Para habilitarlo:

```bash
.venv/bin/pip install -r requirements-cuda.txt
```

La RTX 5070 (Blackwell) necesita un driver de NVIDIA que exponga CUDA 12.8 o superior en WSL.
Si `cupy` no está instalado, no hay dispositivo CUDA o algo falla, tp4 emite un `RuntimeWarning`,
sigue con `numpy` y deja el motivo en `run.json` (`backend.requested`, `backend.resolved`,
`backend.fallback_reason`); la corrida termina con código 0.

## Hallazgos

Las conclusiones del Ejercicio 1 (EDA, conjunto de variables, PCA y la decisión sobre la entrada
de Kohonen) están en [`docs/hallazgos/ej1.md`](docs/hallazgos/ej1.md). Para verificar que cada
gráfico y número que cita viene de una corrida regenerada:

```bash
.venv/bin/python -m tp4 run configs/eda.json
.venv/bin/python -m tp4 run configs/pca.json
.venv/bin/python -m tp4 run configs/pca_scaling_sweep.json
.venv/bin/python -m tp4.findings docs/hallazgos/ej1.md \
  --cite runs/pca/summary.json:pc1_explained_variance_pct \
  --cite runs/pca/summary.json:pc2_explained_variance_pct \
  --cite runs/pca/summary.json:cumulative_pc2_pct \
  --cite runs/pca/summary.json:pc1_top_feature \
  --cite runs/pca_scaling_sweep/standardize-false/summary.json:pc1_explained_variance_pct \
  --cite runs/eda/summary.json:n_selected_features
```

El comando imprime `MISSING artifact:` o `MISSING cite:` por cada archivo o valor que falte y
sale con código 1; si todo está, imprime `ok: ...` y sale con 0.

### Sección 1.2: mapa de Kohonen

La sección `## 1.2 Implementación: mapa de Kohonen` de la misma nota justifica la configuración del mapa, describe la
distribución de los creadores, la matriz U, los planos de componentes, el perfil de cada región, la comparación con el
biplot de PC1 y PC2 y un mapa de control sobre las 6 primeras componentes (`kind: "som_control"`, `configs/som_pca_control.json`,
que reusa los hiperparámetros y la semilla de `configs/som.json`). Para regenerar la evidencia de la 1.2, en este orden
(barrido de diseño, barrido, mapa, control), y verificarla:

```bash
.venv/bin/python -m tp4 run configs/som_design_sweep.json
.venv/bin/python -m tp4 run configs/som_sweep.json
.venv/bin/python -m tp4 run configs/som.json
.venv/bin/python -m tp4 run configs/som_pca_control.json
.venv/bin/python -m tp4.experiments.som_sweep_report check configs/som.json runs/som_design_sweep runs/som_sweep --run runs/som
.venv/bin/python -m tp4.findings docs/hallazgos/ej1.md \
  --cite runs/pca/summary.json:pc1_explained_variance_pct \
  --cite runs/pca/summary.json:pc2_explained_variance_pct \
  --cite runs/pca/summary.json:cumulative_pc2_pct \
  --cite runs/pca/summary.json:pc1_top_feature \
  --cite runs/pca_scaling_sweep/standardize-false/summary.json:pc1_explained_variance_pct \
  --cite runs/eda/summary.json:n_selected_features \
  --cite runs/som_sweep/summary.json:selected_child \
  --cite runs/som_sweep/summary.json:selected_quantization_error \
  --cite runs/som_sweep/summary.json:selected_topographic_error \
  --cite runs/som_sweep/summary.json:selected_distance_preservation \
  --cite runs/som_design_sweep/summary.json:selected_child \
  --cite runs/som/summary.json:n_regions \
  --cite runs/som/summary.json:u_border_ratio \
  --cite runs/som/summary.json:pc1_region_eta2 \
  --cite runs/som/summary.json:pc2_region_eta2 \
  --cite runs/som/summary.json:empty_neuron_frac \
  --cite runs/som_pca_control/summary.json:region_ari \
  --cite runs/som_pca_control/summary.json:bmu_distance_agreement
```

El mapa de control escribe `runs/som_pca_control/{control_comparison.csv, creator_regions.csv, region_contingency.csv,
control_region_maps.png}`. La semilla del control tiene que ser igual a la de la config base (si no, sale con código 2),
así su mapa sobre las variables es exactamente el de `runs/som`. El QE de los dos mapas no es comparable (distinto espacio);
la nota compara TE, preservación de distancias y el índice de Rand ajustado de las regiones (`region_ari`).
