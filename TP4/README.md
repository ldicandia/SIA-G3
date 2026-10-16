# SIA TP4 — Métodos de Aprendizaje No Supervisado

Implementación en Python del TP4 de "Sistemas de Inteligencia Artificial" (ITBA): análisis
exploratorio y PCA de `datos/creadores.csv`, un mapa de Kohonen escrito a mano, la regla de Oja
y una red de Hopfield. Cada análisis se corre con **un solo comando** a partir de un archivo de
configuración con semilla fija, y deja sus resultados y gráficos en `runs/<experimento>/`.

Qué se entrega:

- **Ejercicio 1** (campañas de marketing): EDA, PCA, mapa de Kohonen, regla de Oja y la
  elección de creadores por campaña. **Ejercicio 2** (memoria asociativa): red de Hopfield sobre
  letras de 5×5 y sobre los códigos QR de menú. Kohonen, Oja y Hopfield están escritos a mano en
  numpy; solo la PCA usa sklearn.
- **Una nota de hallazgos por ejercicio** en `docs/hallazgos/` ([`ej1.md`](docs/hallazgos/ej1.md) y
  [`ej2.md`](docs/hallazgos/ej2.md)), con las figuras commiteadas en `docs/hallazgos/figuras/` y un
  *Resumen para la presentación* al principio de cada una.
- **Regeneración y verificación con un comando**: ver
  [Regenerar y verificar todo](#regenerar-y-verificar-todo). Las pruebas de TEST-01 y la prueba en
  una máquina limpia están en [Tests](#tests).

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
| `kind` | tipo de experimento, uno de los 9 registrados: `eda` (abajo), `pca` ([PCA](#pca-kind-pca)), `sweep` ([Barridos en paralelo](#barridos-en-paralelo)), `som` y `som_control` ([Mapa de Kohonen](#mapa-de-kohonen-kind-som); el control es el mapa sobre las componentes principales), `oja` ([Regla de Oja](#regla-de-oja-kind-oja)), `campaigns` ([Campañas](#campañas-kind-campaigns)), `hopfield_letters` ([Letras 5×5](#letras-55-kind-hopfield_letters)) y `hopfield_qr` ([Códigos QR](#códigos-qr-kind-hopfield_qr)) |
| `seed` | semilla entera obligatoria; misma semilla, mismos números |
| `backend` | `numpy` (por defecto) o `cupy` (opcional, ver "Backend CUDA opcional") |
| `plots` | `true`/`false`: escribir o no los PNG |

Para `kind: "eda"` además: `data` (ruta al CSV, relativa a `TP4/`), `feature_set` (ruta al JSON
que declara las variables que se descartan y cuáles se reconstruyen, `configs/feature_set.json`),
`params.outlier_iqr_k` (multiplicador de las cercas de Tukey, 1.5 por defecto) y
`params.near_redundant_abs_r` (umbral de |r| desde el que un par de variables es casi redundante,
en `(0, 1]`, 0.95 por defecto).

Una corrida reemplaza por completo a otra con el mismo `experiment`, pero solo si esa carpeta
contiene un `run.json` de tp4; una carpeta ajena se rechaza sin tocarla.

## Tests

Suite completa (unos 7 minutos; ningún test necesita `runs/`: cada experimento escribe en un directorio temporal):

```bash
.venv/bin/python -m pytest -q
```

Las cuatro propiedades que exige TEST-01, con los tests que las prueban (`archivo::test`):

| Propiedad (TEST-01) | Tests |
|---------------------|-------|
| Oja se alinea con PC1 | `tests/test_oja_model.py::test_recovers_the_leading_eigenvector_and_norm_one` (datos sintéticos con autovector conocido); `tests/test_oja.py::test_summary_validation_numbers_and_orientation` y `tests/test_oja.py::test_weights_vs_pc1_table` (sobre los creadores, contra la PC1 de sklearn) |
| Hopfield recupera los patrones guardados | `tests/test_hopfield_model.py::test_selected_letters_are_fixed_points_under_async_and_sync`; `tests/test_hopfield_model.py::test_random_stored_patterns_are_fixed_points`; `tests/test_hopfield_letters.py::test_exhaustive_one_and_two_flip_queries_converge_to_target` |
| La energía asincrónica nunca sube | `tests/test_hopfield_model.py::test_async_energy_never_increases_per_neuron_update_or_per_iteration` |
| Kohonen: ganadora (BMU) y vecindad | `tests/test_som_model.py::test_bmu_matches_brute_force_euclidean`; `tests/test_som_model.py::test_bmu_tie_goes_to_lowest_index_and_other_is_second`; `tests/test_som_model.py::test_gaussian_is_one_at_winner_and_decreases_with_distance_and_shrinking_radius`; `tests/test_som_model.py::test_step_is_non_increasing_in_distance_and_radius`; `tests/test_som_model.py::test_winner_always_has_full_influence_at_any_positive_radius` |

Un solo comando que corre exactamente esos 12 tests (pytest sale con error si un nombre dejó de existir, así que el mapa no puede quedar desactualizado en silencio):

```bash
.venv/bin/python -m pytest -q \
  tests/test_oja_model.py::test_recovers_the_leading_eigenvector_and_norm_one \
  tests/test_oja.py::test_summary_validation_numbers_and_orientation \
  tests/test_oja.py::test_weights_vs_pc1_table \
  tests/test_hopfield_model.py::test_selected_letters_are_fixed_points_under_async_and_sync \
  tests/test_hopfield_model.py::test_random_stored_patterns_are_fixed_points \
  tests/test_hopfield_letters.py::test_exhaustive_one_and_two_flip_queries_converge_to_target \
  tests/test_hopfield_model.py::test_async_energy_never_increases_per_neuron_update_or_per_iteration \
  tests/test_som_model.py::test_bmu_matches_brute_force_euclidean \
  tests/test_som_model.py::test_bmu_tie_goes_to_lowest_index_and_other_is_second \
  tests/test_som_model.py::test_gaussian_is_one_at_winner_and_decreases_with_distance_and_shrinking_radius \
  tests/test_som_model.py::test_step_is_non_increasing_in_distance_and_radius \
  tests/test_som_model.py::test_winner_always_has_full_influence_at_any_positive_radius
```

### Prueba en una máquina limpia (`fresh-venv`)

```bash
.venv/bin/python -m tp4.delivery fresh-venv
```

Exporta el **HEAD commiteado** con `git archive` a un directorio temporal (sin `runs/`, sin `.venv`
y sin archivos sin trackear), crea ahí un venv nuevo con **solo `requirements.txt`** (sin
`zxing-cpp` ni CuPy), corre pytest dentro de la exportación y borra todo al terminar. Es lo que
recibe un compañero con un clone limpio. Corrélo **después de commitear**: si hay cambios sin
commitear imprime `WARNING: uncommitted changes are not tested` y no los prueba. Todo lo que
sigue a `--` va a pytest sin cambios (por defecto, `-q`), por ejemplo los 12 tests de arriba:
`fresh-venv -- -q <node ids>`. Sale con el código de pytest (se imprime `fresh-venv: pytest exit <código>`);
si falla un paso de preparación (`git archive`, extracción, venv o `pip install`, que necesita
acceso a PyPI) imprime `FAILED: <paso>` y sale con 3.

Duración esperada: unos 10 minutos con la instalación. Los únicos tests que se saltean (`skipped`)
son los de los extras opcionales: 3 de `zxing-cpp` (`tests/test_qr.py` y `tests/test_hopfield_qr.py`)
y 1 de CuPy (`tests/test_backend.py`).

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
| `correlation_heatmap.png` | mapa de calor de la correlación entre todas las variables numéricas |
| `correlation_heatmap_selected.png` | el mismo mapa de calor, solo con las variables seleccionadas por `feature_set` |
| `identities.csv` | las identidades documentadas entre columnas (por ejemplo precio en ARS = precio en USD × 1500): si se cumplen en los datos (`holds`), el residuo máximo, la tolerancia y qué columna descartada (`broken_by`) las rompe |
| `near_redundant.csv` | los pares de variables numéricas con \|r\| >= `near_redundant_abs_r`, con la marca `both_selected` si las dos quedaron en el conjunto seleccionado |
| `feature_set.json` | el conjunto resuelto: variables seleccionadas, descartadas con su motivo, reconstruidas con su fórmula y si el conjunto es válido (`valid`) |

Los outliers solo se **marcan y reportan**: ningún creador se elimina ni se modifica.

`runs/` se regenera a partir de las configs y no se commitea (está en `.gitignore`).

## PCA (`kind: "pca"`)

```bash
.venv/bin/python -m tp4 run configs/pca.json
```

PCA de sklearn sobre las variables seleccionadas (`configs/feature_set.json`). Claves propias:
`data` (ruta al CSV) y `feature_set` (ruta al JSON de variables). `params`:

| Parámetro | Significado | Valores permitidos |
|-----------|-------------|--------------------|
| `standardize` | estandarizar (media 0, desvío 1) o solo centrar | `true` (por defecto) o `false` |
| `n_components` | cantidad de componentes que se calculan | `null` (todas) o entero >= 1 (no puede superar `min(creadores, variables)`) |
| `variance_thresholds` | umbrales de varianza acumulada para contar componentes (`k_cum_80`, `k_cum_90`) | lista no vacía de números en `(0, 1]`, `[0.8, 0.9]` por defecto |
| `loadings_pcs` | componentes que muestra el mapa de calor de cargas | entero >= 1, 6 por defecto |
| `biplot_max_arrows` | cantidad máxima de flechas de variables en el biplot (las de mayor norma) | `null` (todas) o entero >= 1 |
| `top_n` | variables que se listan por componente en `summary.json` (`pc1_top5`, `pc2_top5`) | entero >= 1, 5 por defecto |

| Artefacto en `runs/pca/` | Contenido |
|--------------------------|-----------|
| `explained_variance.csv` | una fila por componente: autovalor, varianza explicada y varianza acumulada |
| `loadings.csv` | coeficientes de cada componente (autovectores unitarios), una fila por variable |
| `scores.csv` | una fila por creador (en el orden del CSV): identificación, tópico dominante y PC1..PCk |
| `scree.png` | varianza explicada por componente y acumulada, con las líneas de los umbrales |
| `loadings_heatmap.png` | variables contra las primeras `loadings_pcs` componentes |
| `biplot.png` | PC1 contra PC2 con cada creador rotulado, coloreado por tópico dominante, y las flechas de las cargas |
| `summary.json`, `run.json` | varianza de PC1 y PC2 y acumulada, `k_cum_*`, `kaiser_k`, variable principal de PC1 y PC2 con su carga, la verificación por autovalores y el registro de la corrida |

Convención de signo: cada componente se orienta de modo que su coeficiente de mayor valor
absoluto sea positivo, así el signo no cambia de una corrida a otra (Oja y `runs/som` usan la misma
convención).

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

## Regla de Oja (`kind: "oja"`)

```bash
.venv/bin/python -m tp4 run configs/oja.json
```

Una sola neurona de Oja escrita **a mano en numpy** (`tp4/models/oja.py`; sin librerías de redes ni
autovalores, y un test lo verifica mirando los imports y las llamadas de `tp4/`). Entrena online, un
creador por vez, sobre las mismas 25 variables estandarizadas que la PCA, con la regla
`y = w·x`, `Δw = η·y·(x − y·w)`, pesos iniciales uniformes en [0, 1) y una permutación de los
creadores por época. El vector final se normaliza a largo 1 y se orienta como PC1 (coeficiente de
mayor valor absoluto positivo); el **índice** de cada creador es su fila estandarizada proyectada
sobre ese vector. Claves propias: `data`, `feature_set` (la entrada de la PCA) y `som_config`
(la config `som` cuyas regiones se escriben en el ranking; tiene que usar los mismos `data` y
`feature_set`). `params`:

| Parámetro | Significado | Valores permitidos |
|-----------|-------------|--------------------|
| `epochs` | pasadas sobre los creadores | entero entre 1 y 5000 |
| `eta0` | tasa de aprendizaje inicial | `0 < eta_min <= eta0 <= 1` |
| `eta_min` | piso de la tasa de aprendizaje | `0 < eta_min <= eta0` |
| `eta_decay` | cómo baja la tasa de `eta0` a `eta_min` | `"exponential"`, `"linear"`, `"inverse"` o `"constant"` |
| `norm_tol` | banda alrededor de 1 donde se considera que la norma convergió | número con `0 < norm_tol < 1` |

Una config inválida sale con `error: ...` y código 2 antes de crear `runs/oja/`. Una corrida que
diverge (η demasiado grande) queda marcada en `summary.json` (`diverged`, `divergence_epoch`) y no
escribe ranking.

| Artefacto en `runs/oja/` | Contenido |
|--------------------------|-----------|
| `ranking.csv` | una fila por creador, de mayor a menor índice: tópicos (arte reconstruido), followers, `oja_index`, `pc1_score` y `som_region` |
| `history.csv` | por época: `eta`, norma de `w`, `delta_w` y coseno con PC1 |
| `weights_vs_pc1.csv` | los 25 pesos de Oja (orientados y tal como los entrena la regla) al lado de las cargas de PC1 |
| `convergence.png` | la norma de `w` tendiendo a 1 y el cambio por época |
| `cos_to_pc1.png` | coseno con PC1 por época |
| `weights_vs_pc1.png` | peso de Oja contra carga de PC1, variable por variable |
| `scores_vs_pc1.png` | índice de Oja contra puntaje de PC1 por creador |
| `summary.json`, `run.json` | `cosine_to_pc1`, `pearson_scores_pc1`, `final_w_norm`, `epochs_to_converge`, `orientation_matches_pc1`, `lambda1`, `course_eta_bound`, mejor y peor creador, y el registro de la corrida |

La región de cada creador se recalcula desde `configs/som.json` (misma semilla y misma cuenta que
`runs/som`; un test verifica que coinciden).

## Barrido de la tasa de aprendizaje de Oja

`configs/oja_sweep.json` es un `kind: "sweep"` sobre `configs/oja.json`: 9 valores de `params.eta0`
(de 1e-05 a 0,5) por 3 semillas (42, 7 y 11), 27 corridas. Igual que con Kohonen, `configs/oja.json`
**no se edita a mano**: su `eta0` es el que eligió esta regla (un test verifica que coinciden).
La regla es determinista y **no mira PC1**:

> Se descartan los `eta0` con alguna semilla divergente; luego los que dejan alguna semilla con la
> norma de `w` fuera de la banda `norm_tol` alrededor de 1; luego los cuyas semillas ordenan a los
> creadores distinto (desplazamiento máximo de ranking mayor que 0); entre los que quedan se elige
> el que converge en menos épocas (el máximo entre semillas) y, en empate, el `eta0` menor.

```bash
.venv/bin/python -m tp4 run configs/oja_sweep.json
.venv/bin/python -m tp4 run configs/oja.json
```

| Artefacto en `runs/oja_sweep/` | Contenido |
|--------------------------------|-----------|
| `sweep.csv` | una fila por hijo (`eta0` x semilla) con las métricas de la corrida |
| `selection.csv` | por `eta0`: semillas divergentes, épocas para converger, desvío de la norma, desplazamiento de ranking, coseno mínimo con PC1, etapas de la regla y la columna `selected` |
| `selected_params.json` | hijo elegido y valores de la grilla |
| `sweep_convergence.png` | la norma de `w` por época para cada `eta0` y cada semilla, con el elegido marcado |
| `sign_ambiguity.png` | pesos antes y después de orientar, por semilla: la red encuentra el eje, no el sentido |
| `summary.json` | `selected_eta0`, `selected_child`, `selected_epochs_to_converge`, `selection_rule`, cantidades que pasan cada etapa, `seeds_flipped` y `seeds_not_flipped` |

Con 1/(1,2·λ1) = `course_eta_bound` y la regla de "η0 <= 0,5" del curso, la regla online diverge
sobre estos datos: cada paso escala con ‖x‖² del creador, no con λ1.

## Campañas (`kind: "campaigns"`)

```bash
.venv/bin/python -m tp4 run configs/campaigns.json
```

No entrena ningún modelo nuevo: recalcula, desde las mismas configs, las tres evidencias de cada
creador (PC1 y PC2 de la PCA, la celda y la región del mapa de Kohonen y el índice de Oja; un test
verifica que son iguales a `runs/oja` y `runs/som`) y aplica una **regla transparente por cliente**
(filtro de tópico, filtro opcional de región del mapa y una suma ponderada de puntajes z calculados
sobre los 50 creadores). Los elegidos son los `n_picks` de mayor puntaje, y se comparan con dos
selecciones solo por seguidores (global y dentro del tópico). Esquema de `configs/campaigns.json`:

| Clave | Significado |
|-------|-------------|
| `oja_config` | ruta a la config `oja` (de ella salen `som_config`, `data` y `feature_set`) |
| `params.n_picks` | elegidos por cliente, entero entre 1 y 10 (3 por defecto) |
| `clients` | lista de 1 a 8 clientes, cada uno con las claves de abajo |
| `clients[].name` | identificador (`[a-z][a-z0-9_]{0,23}`, único, no puede ser `n`, `oja`, `followers_global`, `overlapping` ni `backend`): prefija sus claves de `summary.json` |
| `clients[].label` | nombre para las tablas y gráficos |
| `clients[].objective` | objetivo del cliente (texto) |
| `clients[].topic` | columna de tópico del filtro: `topic_football_pct`, `topic_makeup_pct`, `topic_dogs_pct` o `topic_art_pct` (nunca nombres de cuenta) |
| `clients[].min_share` | participación mínima del tópico, 0 a 100, inclusiva |
| `clients[].regions` | opcional: regiones del mapa permitidas (enteros desde 1, sin repetir; `[]` o ausente = todas). Un número mayor que `n_regions` sale con código 2 |
| `clients[].rank_by` | lista no vacía de `{column, weight}`: peso distinto de cero (negativo = menor es mejor) y columna sin repetir |

Columnas permitidas en `rank_by` (`RANKABLE_COLUMNS` en `tp4/analysis/campaigns.py`, una lista
cerrada): las 25 variables seleccionadas más `topic_art_pct` (reconstruido), `content_showcase_pct`,
y las derivadas `est_arg_reach`, `engagement_per_1000` y `cost_per_1000_arg_reach`. Una config
inválida sale con `error: ...` y código 2 antes de crear `runs/campaigns/`.

| Artefacto en `runs/campaigns/` | Contenido |
|--------------------------------|-----------|
| `evidence.csv` | una fila por creador: PC1/PC2, celda y región del mapa, índice y rango de Oja y las métricas |
| `picks.csv` | los elegidos por cliente con sus tres evidencias y los otros clientes que eligieron al mismo creador (`also_picked_for`) |
| `baselines.csv` | las selecciones solo por seguidores: las más seguidas de todo el catálogo y las más seguidas dentro del tópico del cliente |
| `comparison.csv` | método, seguidores global y seguidores en el tópico, lado a lado: participación del tópico, audiencia argentina, alcance argentino estimado, engagement, precio total y USD por mil de alcance argentino |
| `strategy_profiles.csv` | los candidatos de cada cliente perfilados por región del mapa (la evidencia de la pregunta de fútbol: alcance masivo, descubrimiento o profundidad educativa) |
| `comparison.png` | la comparación en barras, una fila por cliente |
| `summary.json`, `run.json` | por cliente (`<nombre>_picks`, `_rule`, `_n_candidates`, `_mean_topic_share`, `_cost_per_1000_arg_reach`, ...), `followers_global_*`, `n_overlapping_creators`, `oja_top_creator` y `oja_top_creator_picked_for` |

**Agregar un cliente nuevo:** sumar un objeto a `clients` en `configs/campaigns.json` y volver a
correr el comando de arriba; no hace falta reentrenar el mapa ni Oja. Las suposiciones de las
métricas (una publicación por creador al precio `price_usd`, costo agrupado por mil de alcance
argentino estimado) y qué cambiar si cambian los datos están en la sección "Mantenimiento del
sistema" de [`docs/hallazgos/ej1.md`](docs/hallazgos/ej1.md).

## Red de Hopfield (`tp4/models/hopfield.py`)

Una red de Hopfield discreta escrita **a mano en numpy** (sin librerías de redes; un test mira los imports y las
llamadas de todo `tp4/`). Los estados son vectores de N neuronas en {−1, +1}. La usan los experimentos
`hopfield_letters` y `hopfield_qr` (más abajo), y sirve para cualquier otro conjunto de patrones.

- **Reglas de aprendizaje** (`train(patrones, rule)`): `"hebb"` es la del curso, `W = (1/N)·Σ ξξᵀ` con **diagonal cero**; `"pseudoinverse"`
  es la proyección `W = K·(KᵀK)⁻¹·Kᵀ`, también con diagonal cero, que **no es la del curso**: existe para comparar cuando los patrones
  están muy correlacionados (necesita patrones linealmente independientes).
- **Actualizaciones** (`recall`): `"async"` visita las N neuronas en un orden aleatorio por barrido (una permutación por barrido de la semilla, compartida por
  todas las consultas de un lote) y `"sync"` actualiza todas a la vez, `S(t+1) = signo(W·S(t))`.
- **Regla de signo:** `h_i = Σ_j w_ij S_j`; si `h_i = 0` (en rigor `|h_i| ≤ 1e-9`) la neurona **conserva su estado**.
- **Energía:** `H(S) = −½·Σ w_ij S_i S_j` (`energy`), por iteración y, opcionalmente (`trace`), por cada actualización individual de neurona;
  en modo asincrónico nunca sube.
- **Criterios de corte:** un barrido sin cambios (`converged`), un ciclo de período 2 en modo sincrónico (`cycle`) o `max_iter` barridos
  (`max_iter`, hasta 1000). `iterations` cuenta solo los barridos que cambiaron el estado.
- **Consultas en lote:** `recall` recibe una matriz de consultas (B × N) y devuelve estados finales, iteraciones, estado de corte y la energía de cada paso;
  el resultado de una fila no depende de qué otras consultas la acompañen.

## Letras 5×5 (`kind: "hopfield_letters"`)

```bash
.venv/bin/python -m tp4 run configs/hopfield_letters.json
```

Ejercicio 2.2 del enunciado: guarda 4 letras de 5×5 en la red, elige cuáles por **ortogonalidad** (matriz de productos internos de las 26 letras y un
ranking de los C(26, 4) subconjuntos), mide la capacidad, la recuperación con ruido y muestra un estado espúreo. Clave propia:
`alphabet` (ruta a un JSON que mapea una letra mayúscula ASCII a una lista de strings de la misma longitud, con `#` = píxel negro = +1 y
`.` = píxel blanco = −1; todas las letras con la misma forma y sin repetir; el archivo `datos/letras_5x5.json` tiene las 26 letras, dibujadas a mano).
`params`:

| Parámetro | Significado | Valores permitidos |
|-----------|-------------|--------------------|
| `n_store` | cuántas letras se guardan | entero entre 2 y 6 (y como máximo la cantidad de letras; C(letras, `n_store`) ≤ 300000) |
| `letters` | letras a guardar a mano; `[]` = elegir las más ortogonales | `[]` o `n_store` letras distintas del alfabeto |
| `rule` | regla de aprendizaje | `"hebb"` o `"pseudoinverse"` |
| `updates` | modos de actualización a medir | lista no vacía sin repetidos de `"async"` y `"sync"` |
| `max_iter` | tope de barridos | entero entre 1 y 1000 |
| `noise_levels` | fracciones de píxeles invertidos | lista no vacía, estrictamente creciente, números en [0, 1] |
| `example_level` | nivel de los ejemplos de `recall_examples.png` | uno de los `noise_levels` |
| `trials` | consultas por letra y nivel | entero entre 1 y 1000 |
| `spurious_noise` | nivel de la consulta "muy ruidosa" del estado espúreo | número en (0, 1] |
| `capacity_max_p` | mayor cantidad de letras de la curva de capacidad | entero entre 1 y la cantidad de letras |
| `capacity_subsets` | subconjuntos aleatorios por p en la curva de capacidad | entero entre 1 y 5000 |

Una config o un alfabeto inválidos salen con `error: ...` y código 2 antes de crear `runs/hopfield_letters/`. Con la misma semilla
los CSV y `summary.json` salen idénticos byte a byte. La red corre en numpy por defecto; con `"backend": "cupy"` los barridos de recuperación corren en la GPU (ver "Backend CUDA opcional") y el backend queda registrado en `run.json`.

| Artefacto en `runs/hopfield_letters/` | Contenido |
|---------------------------------------|-----------|
| `dot_products.csv`, `dot_products.png` | matriz de productos internos de las 26 letras |
| `orthogonality.csv` | ranking de los subconjuntos por (mayor valor absoluto del producto interno, promedio, orden alfabético) |
| `alphabet.png` | las 26 letras con el conjunto elegido y el peor marcados |
| `stored.csv` | por letra guardada (conjunto elegido y peor): punto fijo, margen de estabilidad y energía |
| `capacity.csv`, `capacity.png` | fracción de letras estables y de conjuntos con todas estables, por p, y para el conjunto elegido y el peor |
| `recall.csv` | una fila por consulta: conjunto, modo, letra, nivel, píxeles invertidos, iteraciones, estado de corte, clase final y distancia de Hamming |
| `recall_rates.csv`, `recovery_vs_noise.png`, `recall_examples.png` | tasas de recuperación, espúrea, invertida y de ciclo por conjunto, modo y nivel, y ejemplos |
| `mixtures.csv` | las mezclas de tres letras con signo y si son puntos fijos |
| `spurious.png` | la consulta muy ruidosa, el estado espúreo y las letras guardadas, con superposiciones y energías |
| `energy.csv`, `energy_trace.csv`, `energy.png` | energía por iteración y por actualización de neurona (ejemplos y el caso espúreo) |
| `summary.json`, `run.json` | letras elegidas y peores, productos internos máximos, cotas de capacidad (0,138·N y 0,15·N), tasas de recuperación, el caso espúreo (letra, píxeles, superposiciones, energía, mezcla) y el registro de la corrida |

## Códigos QR (`kind: "hopfield_qr"`)

```bash
.venv/bin/python -m tp4 run configs/hopfield_qr.json
```

Ejercicio 2.1 del enunciado: los cinco códigos QR de menú de `datos/hopfield/*.png` (37×37, negro = +1, N = 1369 neuronas, sin zona de silencio) se guardan en la red
de Hopfield con **dos reglas** (la de Hebb del curso y, solo para comparar, la pseudoinversa), se deterioran con ruido, esquina rota y manchas, y se miden la recuperación
exacta, las iteraciones y la energía en cada paso; además se mide qué lee un decodificador de QR (opcional, ver la sección siguiente) antes y después de la red. Clave propia:
`images` (de 2 a 10 rutas `.png` distintas, en el orden en que se guardan; cada imagen tiene que medir exactamente `side` × `side`). `params`:

| Parámetro | Significado | Valores permitidos |
|-----------|-------------|--------------------|
| `side` | lado de cada código en módulos | entero entre 5 y 64 (37 para estos códigos) |
| `rules` | reglas de aprendizaje a comparar | lista no vacía sin repetidos de `"hebb"` y `"pseudoinverse"` (`["hebb"]` saca la comparación) |
| `update` | modo de actualización | `"async"` o `"sync"` (`trace_levels` no vacío exige `"async"`) |
| `max_iter` | tope de barridos | entero entre 1 y 1000 |
| `noise_levels` | fracciones de módulos invertidos | lista no vacía, estrictamente creciente, números en [0, 1], hasta 21 niveles |
| `trials` | consultas con ruido por código y nivel, y manchas por código y tamaño | entero entre 1 y 200 |
| `corners` | esquinas que se rompen | lista sin repetidos de `"top_left"`, `"top_right"`, `"bottom_left"`, `"bottom_right"` |
| `torn_corner_sizes` | tamaños del triángulo blanco de la esquina rota (`fila + columna < tamaño`) | lista estrictamente creciente de enteros entre 1 y `side` (puede ser `[]`) |
| `occlusion_sizes` | lados de la mancha negra cuadrada | lista estrictamente creciente de enteros entre 1 y `side` (puede ser `[]`) |
| `trace_levels` | niveles de ruido para los que se guarda la energía por actualización de neurona | subconjunto sin repetidos de `noise_levels` (puede ser `[]`) |
| `decode` | usar el decodificador de QR | `true` o `false` (con `false` ni siquiera se prueba que esté instalado) |
| `decode_scale` | factor de escala con el que se dibuja cada matriz para el decodificador | entero entre 1 y 16 |
| `decode_quiet_zone` | módulos blancos de margen que se agregan al dibujarla | entero entre 0 y 10 |

Tipos de deterioro (`damage` en `queries.csv`): `flip` (módulos invertidos al azar, `round(nivel·N)` distintos), `torn_corner` (un triángulo blanco en una esquina, sin azar) y `occlusion` (una
mancha negra cuadrada en una posición al azar con semilla fija). Las dos reglas ven exactamente las mismas consultas. Una consulta se cuenta como **recuperada** solo si el estado final es
**igual** al código de origen. Una config inválida, una imagen que no mide `side` × `side` o un grid de más de 50000 recuperaciones sale con `error: ...` y código 2 antes de crear `runs/hopfield_qr/`.
Con la misma semilla los CSV y `summary.json` salen idénticos byte a byte.

| Artefacto en `runs/hopfield_qr/` | Contenido |
|----------------------------------|-----------|
| `patterns.png` | los cinco códigos como matrices de ±1 |
| `overlaps.csv`, `overlaps.png` | superposición `m` entre cada par de códigos |
| `shared_modules.png` | módulos idénticos en los cinco códigos y módulos que difieren |
| `stored.csv` | por regla y código: si es punto fijo, módulos inestables, margen de estabilidad y energía |
| `hebb_attractor.png` | el "QR promedio" `signo(Σ ξ)` al que caen las consultas con Hebb y los módulos inestables de cada código |
| `queries.csv` | una fila por consulta y regla (columnas abajo) |
| `recovery_rates.csv` | por deterioro, nivel o tamaño y regla: tasas de recuperada, otro código, invertida, espúrea y del "QR promedio", iteraciones, superposición con el objetivo y tasas de decodificación |
| `damage_examples.png`, `recovery_vs_noise.png` | consultas dañadas y su estado final, y la recuperación exacta por tipo de deterioro |
| `iterations.png`, `energy.csv`, `energy_trace.csv`, `energy_per_step.png` | iteraciones frente al ruido, energía por iteración de cada consulta y energía por actualización de neurona (`trace_levels`) |
| `decode_vs_noise.png` | solo con decodificador: fracción que se lee como el menú original antes y después de la red |
| `summary.json`, `run.json` | `hebb_*` y `pseudoinverse_*` (puntos fijos, módulos inestables, recuperación por nivel, esquina y mancha, iteraciones), superposiciones, módulos compartidos, `capacity_0138`, `energy_never_increased` y las claves de decodificación (`decoder_available`, `decoder_name`, `decoder_version`, `decoder_reason`, `n_originals_decoded`, `reference_texts`, `*_decode_rates`, `mean_attractor_decoded_as`) |

Columnas de `queries.csv`: `query_id`, `damage`, `level`, `size`, `corner`, `top`, `left`, `pattern`, `trial`, `n_changed`, `damaged_frac`, `rule`, `iterations`, `status`, `final_class`
(`stored`, `inverted` o `spurious`), `final_pattern`, `recovered`, `overlap_target`, `hamming_target`, `nearest_pattern`, `is_mean_attractor`, `final_energy`, `query_decode_ok`,
`final_decode_ok` y `final_decoded_as`.

## Decodificador QR opcional

La decodificación (¿un lector de QR lee el código recuperado?) usa `zxing-cpp`, que es un **extra opcional**: ningún test ni ninguna corrida lo necesita. Para habilitarlo:

```bash
.venv/bin/pip install -r requirements-qr.txt
```

Para comprobar que los cinco códigos originales se cargan, se dibujan de nuevo píxel por píxel y se decodifican (cada uno imprime su URL):

```bash
.venv/bin/python -m tp4.analysis.qr datos/hopfield/*.png
```

Sin el paquete instalado, el comando imprime `decoder unavailable: ...` una vez y sale con código 1, y `python -m tp4 run configs/hopfield_qr.json` termina igual con código 0: en `summary.json`
quedan `decoder_available` = `false` y `decoder_reason` con el motivo, sin números de decodificación (con `params.decode` en `false` no se prueba el decodificador y `decoder_available` queda en `null`).
Un `error: ...` con código 2 indica una imagen ilegible o de tamaño incorrecto.

## Backend CUDA opcional

Con `"backend": "cupy"` en la config, los cálculos de arrays que usan `ctx.backend.xp` corren en la
GPU. Es **opcional**: nada de este repo ni ningún test necesita CUDA, y `numpy` es el backend por
defecto (con él, los CSV y `summary.json` salen idénticos byte a byte a los de antes del backend).

**Qué corre en la GPU:** la verificación por autovalores de la PCA, el bucle de entrenamiento de
Kohonen (`som`, `som_control` y la evidencia de `oja` y `campaigns`), el entrenamiento de Oja
(`oja`, `campaigns`) y los barridos de recuperación de Hopfield con sus energías por barrido
(`hopfield_letters`, `hopfield_qr`).

**Qué se queda en el host, y por qué:** todos los sorteos (la `Generator` de numpy, siempre en el
mismo orden), la construcción y validación de los pesos de Hopfield (una sola vez, chicos, y
necesitan chequeos exactos de simetría y diagonal cero), las métricas, el estado de cada consulta
(conjunto activo, iteraciones, estado de corte), los CSV y los gráficos. Cada época (Oja, Kohonen)
o cada barrido (Hopfield) trae los datos de vuelta al host una sola vez.

**Qué se garantiza:** con la misma semilla salen los mismos sorteos y las mismas permutaciones en
los dos backends. Los resultados discretos (índices, etiquetas, estados de Hopfield, regiones,
conteos) son idénticos; los de punto flotante coinciden hasta cerca de 1e-9 pero **no** son
idénticos bit a bit, porque las reducciones de la GPU suman en otro orden. En Kohonen, dos
prototipos casi empatados (la vecindad `step` con radio grande los colapsa en las primeras
épocas) pueden salir en orden distinto por ese redondeo; por eso, si las dos mejores distancias de
la GPU difieren menos de `DEVICE_TIE_RTOL` (1e-12 relativo), la neurona ganadora se decide de nuevo
con la suma de numpy sobre copias del host y coincide con la de una corrida en numpy.

**Instalación** (desde `TP4/`, en WSL):

```bash
.venv/bin/pip install -r requirements-cuda.txt
```

`requirements-cuda.txt` pide `cupy-cuda12x[ctk]`: el extra `[ctk]` trae las librerías de runtime de
CUDA (cuBLAS, NVRTC, cuSOLVER...) como wheels oficiales de NVIDIA desde PyPI. Sin él, `cupy` importa
y ve la GPU, pero falla en la primera operación (`libcublasLt.so.12 not found`) si el sistema no
tiene un CUDA Toolkit 12.x. La RTX 5070 (Blackwell, capacidad 12.0) necesita un driver de NVIDIA que
exponga CUDA 12.8 o superior en WSL (`nvidia-smi`). Si `cupy` no está instalado, no hay dispositivo CUDA
o el driver falla al arrancar, tp4 emite un `RuntimeWarning`, sigue con `numpy` y deja el motivo en
`run.json` (`backend.requested`, `backend.resolved`, `backend.fallback_reason`); la corrida termina con
código 0.

**Barridos:** cada worker de un barrido es un proceso aparte y abriría su propio contexto CUDA. Con
`cupy` dejá `workers: 1`, y mantené las configs commiteadas en `numpy`.

**Comparar numpy contra cupy** (corre las configs dos veces, una por backend y sin gráficos, compara
`summary.json` y todos los CSV, mide los tiempos y corre dos pruebas sintéticas: el desempate de
`argmin` y Kohonen desde prototipos duplicados):

```bash
.venv/bin/python -m tp4.analysis.backend_compare [--configs PATH ...] [--out-root DIR] [--json PATH] [--rtol R] [--atol A]
```

Por defecto compara `som`, `oja`, `hopfield_letters` y `hopfield_qr` con `rtol` 1e-6 y `atol` 1e-9
(los barridos no se admiten). Códigos de salida: 0 todo equivalente; 1 cupy no disponible (imprime
`backend unavailable: <motivo>` y no corre nada); 2 error de uso o de config (`error: ...`); 3 alguna
diferencia, prueba fallida o corrida de cupy rota.

Medido en la RTX 5070 bajo WSL (driver 610.88, `cupy` 14.2.0, runtime CUDA 12.9, `cupy-cuda12x[ctk]`; los
núcleos ya compilados, una corrida por backend):

| Config | numpy (s) | cupy (s) | cupy / numpy | Diferencia absoluta máxima | Veredicto |
|--------|-----------|----------|--------------|----------------------------|-----------|
| `som` | 0,53 | 7,53 | 0,07x | 0 | equivalente |
| `oja` | 0,48 | 7,09 | 0,07x | 0 | equivalente |
| `hopfield_letters` | 0,35 | 1,31 | 0,27x | 2,2e-16 | equivalente |
| `hopfield_qr` | 15,25 | 109,00 | 0,14x | 0 | equivalente |

Las dos pruebas sintéticas pasan (desempate de `argmin` y Kohonen desde prototipos duplicados, diferencia
máxima de pesos 2,2e-16). **La GPU es más lenta que numpy en todas las configs:** estos bucles son
secuenciales (una muestra o una neurona por vez) y chicos, así que el costo de lanzar núcleos y de
sincronizar con el host domina. El valor del backend es la equivalencia validada y el margen para
problemas más grandes, no una aceleración prometida; para estos datos conviene `numpy`.

## Regenerar y verificar todo

Todo lo que citan las notas de hallazgos se regenera y se verifica con `python -m tp4.delivery`
(código en `tp4/delivery.py`). Lee `docs/hallazgos/evidencia.json`, el manifiesto de la entrega:
el orden de las corridas (`run_order`), las configs que no son corridas (`not_run`, con el motivo)
y, por nota, los valores que se citan (`cites`), las precondiciones (`requires`) y las secciones
con citas propias (`sections`). El manifiesto solo tiene datos: lo único que se ejecuta son
procesos `python -m tp4 run <config>`.

**Requisito previo:** los números del decodificador de QR de la sección 2.1 necesitan el extra
opcional (`.venv/bin/pip install -r requirements-qr.txt`, ver "Decodificador QR opcional"). Sin él
todo se regenera igual, pero `check` imprime una línea `PRECONDITION ...` que nombra la causa y
sale con código 1.

| Comando | Qué hace |
|---------|----------|
| `.venv/bin/python -m tp4.delivery regenerate [--out-root DIR]` | corre las 12 configs de `run_order`, en ese orden y cada una en su propio proceso `python -m tp4 run` (por defecto escribe en `runs/`, que reemplaza), y después verifica que `configs/som.json` coincide con lo que eligieron los barridos de Kohonen. Imprime `[i/12] configs/...` y termina con `regenerate ok: 12 configs, som selection ok`; si una config falla, imprime `FAILED: <config> (exit N)` y se detiene |
| `.venv/bin/python -m tp4.delivery figures [--root DIR]` | copia a `docs/hallazgos/figuras/` cada figura que las notas embeben desde `DIR/runs` (por defecto `runs/`) y reescribe los links `../../runs/...png` de las notas a la forma de copia. Imprime `figures: N declared, C copied, R links rewritten, D removed` |
| `.venv/bin/python -m tp4.delivery check [--root DIR]` | revisa las notas contra `DIR/runs`: que cada config esté clasificada, que cada `runs/<experimento>` que nombran tenga su config, las precondiciones, los números citados (en toda la nota y en cada sección), que cada referencia a un título (`` `### Título` ``) exista, y que las figuras commiteadas sean idénticas byte a byte a las de las corridas. Termina con `delivery ok: <n> notes, <n> configs, <n> cites, <n> figures` o con `delivery FAILED: N problem(s)` |
| `.venv/bin/python -m tp4.delivery verify` | todo desde cero: regenera las 12 configs en un directorio temporal, corre `check` contra ellas y borra el directorio (no toca `runs/`). Termina con `verify ok: clean regeneration matches the notes`. **Es el comando que hay que correr antes de armar las diapositivas** |
| `.venv/bin/python -m tp4.delivery fresh-venv` | corre la suite completa desde una exportación limpia del HEAD en un venv nuevo; ver [Tests](#tests) |

Para agregar algo nuevo:

- **Una config:** clasificala en `docs/hallazgos/evidencia.json`. Si produce una corrida, va en
  `run_order` (después de las que lee); si es una definición o la base de un barrido, va en
  `not_run` con el motivo escrito. Sin eso, `check` imprime `UNCLASSIFIED`.
- **Una figura:** escribí el link de la imagen en la nota apuntando al PNG de la corrida, dos
  niveles arriba (`![texto](../../runs/<experimento>/<archivo>.png)`), y corré `figures`: reescribe
  el link y copia el archivo. Los PNG de `docs/hallazgos/figuras/` que ninguna nota usa se borran.
- **Un número citado:** agregá `runs/<experimento>/<archivo>.json:<clave>` a `cites` de la nota
  (o de una de sus `sections`, si tiene que aparecer en esa sección) en `evidencia.json`. El valor
  de esa clave tiene que estar escrito en el texto, con coma decimal.

Qué significa cada línea de error:

| Línea | Causa |
|-------|-------|
| `UNCLASSIFIED config: configs/x.json` | la config no está ni en `run_order` ni en `not_run` |
| `NO CONFIG for runs/x (nota)` | la nota (o una cita o precondición) nombra `runs/x`, que ninguna config de `run_order` produce |
| `PRECONDITION archivo:clave = valor, expected esperado: pista` | no se cumple una precondición de la nota (por ejemplo el decodificador de QR no está instalado); la pista dice cómo arreglarlo |
| `MISSING artifact: ...` / `MISSING cite: archivo:clave=valor` | la nota nombra un archivo que no existe, o un número citado no aparece escrito en el texto (o la clave no existe) |
| `MISSING section: ...` | el título de la sección que declara el manifiesto no está en la nota |
| `MISSING figure source` / `MISSING figure copy` | falta el PNG de la corrida o su copia en `docs/hallazgos/figuras/` (correr `figures`) |
| `STALE figure: copia differs from fuente` | la copia commiteada ya no es idéntica a la de la corrida (correr `figures`) |
| `ORPHAN figure copy: archivo` | hay un PNG en `docs/hallazgos/figuras/` que ninguna nota embebe (correr `figures` lo borra) |
| `BAD figure link: ...` | un link de imagen no tiene la forma de copia ni apunta a `../../runs/...png` (por ejemplo tiene `..` o un nombre raro) |
| `UNKNOWN figure reference: archivo` | el texto menciona una figura `figuras/...png` que ninguna imagen declara |
| `BROKEN section reference: ### Título` | el texto remite a un título entre comillas invertidas que no existe en la nota |

## Hallazgos

Las conclusiones de cada ejercicio están en una nota con un *Resumen para la presentación* al
principio (el esqueleto de las diapositivas: qué sección responde cada pregunta del enunciado,
una diapositiva por hallazgo con su afirmación, figura y número, y la historia que las une).
Las listas de valores citados no se repiten acá: están en
[`docs/hallazgos/evidencia.json`](docs/hallazgos/evidencia.json), y
`python -m tp4.delivery check` (o `verify`, desde cero) las corre contra las corridas. La opción
`--cite` de `python -m tp4.findings` sigue disponible para revisar a mano un valor puntual de una nota.

### Ejercicio 1: campañas de marketing

[`docs/hallazgos/ej1.md`](docs/hallazgos/ej1.md) tiene cuatro secciones:

- **1.1 Exploración y representación (EDA + PCA):** los datos y su calidad, el conjunto de 25
  variables seleccionado, distribuciones, outliers, escalas y correlaciones, la varianza explicada
  por la PCA, si conviene estandarizar (barrido), la interpretación de PC1 y PC2 y la decisión sobre
  la entrada de Kohonen.
- **1.2 Implementación: mapa de Kohonen:** justifica la configuración del mapa (los dos barridos),
  describe la distribución de los creadores, la matriz U, los planos de componentes, el perfil de
  cada región y la comparación con el biplot de PC1 y PC2. Incluye un mapa de control sobre las 6
  primeras componentes (`kind: "som_control"`, `configs/som_pca_control.json`, que reusa los
  hiperparámetros y la semilla de `configs/som.json`; la semilla tiene que ser igual, si no sale con
  código 2). Su QE no es comparable con el del mapa final (distinto espacio): la nota compara el
  error topográfico, la preservación de distancias y el índice de Rand ajustado de las regiones
  (`region_ari`). Escribe `runs/som_pca_control/{control_comparison.csv, creator_regions.csv,
  region_contingency.csv, control_region_maps.png}`.
- **1.3 Índice (regla de Oja):** la regla implementada, la convergencia de la norma, la elección de
  la tasa de aprendizaje con el barrido, la comparación con PC1 (pesos y puntajes), qué mide el
  índice y por qué no conviene elegir siempre al creador con el índice más alto.
- **1.4 Decisión para las campañas:** el método de selección, tres elegidos por cliente con su
  posición en el biplot, su región y celda del mapa y su índice de Oja, la respuesta a la pregunta de
  fútbol (alcance masivo, descubrimiento o profundidad educativa) con los perfiles por región, la
  comparación con la selección solo por seguidores y el mantenimiento del sistema (campaña nueva,
  creadores o variables nuevas y actualización mensual).

### Ejercicio 2: memoria asociativa (Hopfield)

[`docs/hallazgos/ej2.md`](docs/hallazgos/ej2.md) va primero con la sección **2.2 Recomendación:
letras 5×5** (el enunciado recomienda probar el caso simple) y después **2.1 Recuperación de
códigos QR deteriorados**:

- **2.2:** la red implementada, por qué se guardan G, R, T y V (ortogonalidad), la capacidad frente
  a 0,138·N y 0,15·N, la recuperación con ruido y un estado espúreo.
- **2.1:** los cinco códigos como matrices de 37×37, por qué se parecen tanto, que la regla de Hebb
  del curso no recupera ninguno, la comparación con la regla de la pseudoinversa (que no es la del
  curso), la recuperación con ruido, esquina rota y manchas, las iteraciones y la energía en cada
  paso, lo que lee un decodificador de QR y las limitaciones del caso de uso. Los números del
  decodificador necesitan `requirements-qr.txt` (ver "Regenerar y verificar todo").
