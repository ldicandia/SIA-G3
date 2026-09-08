# SIA TP2 — Algoritmos Genéticos

Un motor de algoritmos genéticos escrito a mano — sin ninguna biblioteca de algoritmos
genéticos cerca, según la prohibición explícita del enunciado — que aproxima una imagen
objetivo con triángulos translúcidos sobre un lienzo en blanco. Construido para el TP2 de
"Sistemas de Inteligencia Artificial" del ITBA, Ejercicio 2 ("un compresor de imágenes un
tanto peculiar"). Dada una imagen objetivo y un presupuesto de triángulos, el motor evoluciona
una población de conjuntos de triángulos, generación tras generación, hasta que el resultado
renderizado se parece al original. La matriz completa de operadores que exige la cátedra —
los nueve métodos de selección, los cuatro de cruza, los cuatro alcances de mutación y las
tres estrategias de supervivencia — se elige desde un único archivo de configuración JSON,
sin necesidad de tocar código para pasar de uno a otro.

El entregable no es "un renderizador que casualmente funciona": es una comparación medida y
defendible. Cada elección de operador documentada más abajo está respaldada por una matriz de
experimentos y un conjunto de figuras comparativas construidas para defender esa elección
frente a la cátedra, no simplemente por una aproximación de imagen que anda.

## Instalación

Python corre sobre un entorno virtual de **WSL**, nunca sobre uno del lado de Windows: un venv
de Windows está roto para las dependencias de este proyecto (la misma restricción con la que
TP1, en este mismo repo, se topó primero). Abrí una shell de WSL/Linux dentro del directorio
`TP2/` clonado, o anteponé a cada comando de abajo:

```
wsl.exe bash -lc 'cd /path/to/TP2 && <command>'
```

(reemplazá `/path/to/TP2` por donde hayas clonado este repo — no copies una ruta de la máquina
de otra persona). Todos los comandos de este README asumen que estás dentro de esa shell, en
el directorio `TP2/`, y usan la forma `.venv/bin/python` de manera consistente en vez de una
shell con el entorno activado — las dos son equivalentes, pero este README se queda con una
sola para que cada comando de abajo se pueda copiar y pegar textualmente.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` fija cinco paquetes: **Pillow** (`==12.3.0`, clavado a una versión exacta
porque varios de los requisitos indispensables de este proyecto son afirmaciones de
reproducibilidad byte a byte que se desviarían entre releases de Pillow), **numpy**
(`>=1.26,<3`; verificado contra `2.5.2`), **pygame-ce** (`>=2.5,<3`; verificado contra `2.5.8`
— usado solo por el visor en vivo opcional), **matplotlib** (`>=3.8,<4`; verificado contra
`3.11.1` — usado solo por los gráficos de los experimentos) y **pytest** (`>=8,<10`; verificado
contra `9.1.1`). Atención al nombre del paquete: es `pygame-ce`, no `pygame` — los dos instalan
un módulo llamado literalmente `pygame`, así que tener ambos instalados a la vez es un entorno
roto; instalá únicamente lo que nombra `requirements.txt`.

## Ejecutar el motor

```bash
.venv/bin/python -m tp2 --image assets/flag_ar.png --triangles 30 --config configs/baseline.json --seed 42 --out runs/demo
```

`--image` siempre es obligatorio. `--triangles` (el presupuesto de triángulos del cromosoma) y
`--config` (el archivo JSON de hiperparámetros — ver [Configuración](#configuración-hiperparámetros-json)
más abajo) son las dos banderas que en la práctica vas a querer setear siempre de forma
explícita; `configs/baseline.json` es el punto de partida que viene incluido y un default
razonable para una primera corrida. Una corrida exitosa escribe cuatro artefactos en un
directorio nuevo dentro de `runs/`:

- `best.png` — el mejor individuo encontrado, renderizado como PNG.
- `triangles.json` — la enumeración de los triángulos de ese individuo (posición, color y flag
  de activo por triángulo).
- `run.json` — la configuración efectiva realmente usada (incluida la semilla resuelta), las
  versiones de las bibliotecas, el commit de git y el motivo por el que la corrida terminó.
- `metrics.csv` — una fila por generación: fitness, error, número de generación, cantidad de
  renders y tiempo transcurrido.

**Omitir `--seed`** sigue produciendo una corrida totalmente reproducible: se sortea una semilla
al azar y se archiva textualmente en `run.json`, así que volver a pasar ese mismo valor con
`--seed` replica exactamente la misma corrida. **Omitir `--out`** no evita que se escriba la
salida: simplemente usa por defecto un directorio nuevo, con timestamp, creado bajo `runs/`.
**Volver a correr sobre el MISMO directorio `--out` sin `--force`** no pisa en silencio lo que
ya está ahí: lanza un error nombrando la ruta en conflicto. Pasar `--force` es la forma
explícita de optar por sobrescribir un directorio de corrida existente.

## Ejecutar el visor

```bash
.venv/bin/python -m tp2 --image assets/flag_ar.png --triangles 30 --config configs/baseline.json --seed 42 --out runs/demo_viewer --viewer
```

El mismo comando de arriba, con `--viewer` agregado al final — el motor en sí nunca importa
`pygame` y corre idéntico en ambos casos; `--viewer` solo agrega una visualización en vivo por
encima. Esto abre una ventana que muestra al mejor individuo evolucionando generación a
generación, con el número de generación, el fitness y la cantidad de renders superpuestos.
`--viewer-scale` (por defecto 4) controla el factor de escala en píxeles de la ventana, y
`--viewer-every` (por defecto 1) controla cada cuánto se redibuja la pantalla (cada N
generaciones). Cerrar la ventana del visor termina la corrida limpiamente en vez de hacerla
explotar: los artefactos de la corrida igual se escriben, y el `stop_reason` de `run.json`
registra `viewer_closed`, así que es distinguible de una corrida que alcanzó su propia
condición de corte de forma natural.

## Ejecutar los experimentos

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tp2.experiments.runner --spec configs/experiments/main_matrix.json --out runs/matrix --jobs 8
```

**Fijá en 1 la cantidad de hilos de BLAS/OpenMP, como arriba — esto no es opcional.** El runner
bifurca (fork) un pool de procesos, y OpenMP no es seguro ante fork: cada worker reinicializa un
pool de hilos completo para arreglos demasiado chicos como para sacarle provecho, y el overhead
de sincronización domina todo lo demás. Medido sobre el runner real (14 corridas, 16 núcleos):
512 s con `--jobs 4` y los hilos sin fijar, contra 6,1 s con `--jobs 4` y los hilos fijados —
unas **85x**. Contra la intuición, subir `--jobs` dejando los hilos sin fijar lo empeora,
porque la sobresuscripción crece con el tamaño del pool. Con los hilos fijados, más jobs ayudan
como es de esperar.

El spec de matriz que viene incluido son 22 celdas (7 métodos de selección, esos mismos 7
métodos aplicados solo a la selección de padres contra un reemplazo `elite`, 6 combinaciones de
supervivencia/ratio `K:N`, y 2 controles de honestidad de la cruza) por 5 semillas cada una =
110 corridas independientes, distribuidas en un pool de procesos dimensionado por `--jobs` (por
defecto `min(cpu_count, 16)`). Con los hilos fijados y `--jobs 8`, la matriz completa tarda unos
5 minutos en una máquina de 16 núcleos; sin fijarlos es efectivamente inejecutable.

Los operadores de mutación viven en un **segundo spec, separado**, porque sus celdas
sobrescriben `mutation` y no alguna de las dimensiones que barre `main_matrix.json`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tp2.experiments.runner --spec configs/experiments/mutation_matrix.json \
  --out runs/matrix_mutation --jobs 8
```

Ese spec son 8 celdas × 5 semillas = 40 corridas (unos 2 minutos, con los hilos fijados, en 16
núcleos): los cuatro alcances de la cátedra con el `Pm = 0.9` propio del baseline, tres de ellos
repetidos con `Pm` reescalado para que todas las celdas muten los mismos **0,9 genes esperados
por hijo**, y un schedule `non_uniform` de Michalewicz. Las celdas con tasa igualada no son
decoración opcional: sobre un cromosoma de 30 triángulos (330 loci) el mismo `Pm = 0.9`
significa 0,9 genes mutados para `gene` y 297 para `multigen_uniform`, así que comparar los
cuatro alcances con igual `Pm` compara cuatro radios de búsqueda distintos en vez de cuatro
operadores.

Una vez corridas ambas matrices, convertilas en figuras con un segundo comando, mucho más chico:

```bash
.venv/bin/python scripts/generate_plots.py
```

Esto lee los archivos `metrics.csv` / `run.json` que `runs/matrix/`, `runs/matrix_mutation/` y
`runs/hillclimber/` ya contienen y escribe **once** figuras comparativas bajo `plots/` — nunca
vuelve a correr el GA, así que se puede reejecutar en cualquier momento (después de retocar el
estilo de un gráfico, por ejemplo) a un costo efectivamente nulo. Si `runs/matrix_mutation/` no
existe, las dos figuras de mutación se saltean con un aviso por stderr en vez de hacer fallar
toda la generación.

Los títulos de las figuras son las cadenas de `FIGURE_CLAIMS` — oraciones en inglés que nombran
la afirmación de la que cada figura es evidencia, que es el epígrafe correcto al lado del código
y el equivocado en una diapositiva en español. Las variantes de la presentación para las tres
figuras más nuevas se construyen aparte, a partir de las mismas funciones y los mismos datos,
con epígrafes y nombres de operadores en español:

```bash
.venv/bin/python scripts/make_deck_figures.py
```

Aterrizan en `plots/deck/` y son las que incrusta `docs/Presentacion.pptx`.

## Ejecutar el baseline hill-climber

```bash
.venv/bin/python -m tp2.baselines.hillclimber --image assets/flag_ar.png --triangles 30 --config configs/baseline.json --seed 1 --out runs/hillclimber
```

Esto es un hill climber estocástico `(1+1)` — población de tamaño uno, mutar y aceptar si mejora,
sin cruza, sin presión de selección sobre una población — usado únicamente como punto de
comparación honesto, a igual presupuesto de renders, en la presentación. **Nunca** es "el GA
baseline": no es un algoritmo genético en absoluto, y llamarlo así tergiversa la comparación
para la que existe.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Corre la suite completa desde el venv de WSL. La suite incluye un test lento de convergencia
(una compuerta sembrada de 3 corridas que verifica convergencia real por encima de un umbral de
fitness); excluilo con `-m "not slow"` para una pasada rápida durante el desarrollo normal.

## Imágenes objetivo incluidas

Bajo `assets/` vienen tres imágenes objetivo: `flag_ar.png` (una bandera), `silhouette.png` y
`pictogram.png` — todas deliberadamente simples, siguiendo el consejo del propio enunciado de
empezar con banderas, siluetas, pictogramas y formas simples por el estilo antes que con
fotografías. Están generadas por `scripts/make_assets.py` a partir de primitivas de dibujo
versionadas en el repo, no tomadas de ninguna fuente externa.

## Configuración (hiperparámetros JSON)

El archivo de configuración se pasa con `--config path/to/file.json`. Cuando una bandera de CLI
y una clave JSON nombran lo mismo (`--population`/`population`, `--canvas`/`canvas`), gana la
bandera de CLI si están las dos presentes. `--triangles` no tiene equivalente en JSON: el
presupuesto de triángulos (`triangle_budget`) es exclusivo de la CLI.

**Claves escalares de nivel superior:**

| Clave | Tipo | Rango | Significado |
|---|---|---|---|
| `population` | int | 1–10000 | Tamaño de la población `N` |
| `children` | int | 1–20000 | Cantidad de hijos producidos por generación, `K` |
| `canvas` | int | 8–1024 (píxeles) | Lado del render cuadrado |
| `recombination_probability` | float | 0.0–1.0 | Probabilidad de cruza por apareamiento `Pc` (CATEDRA.md); si un apareamiento no se recombina, los hijos son copias idénticas de los padres pero igual pasan por la mutación |
| `horizon` | int | ≥ 1 | Tope de cantidad de generaciones. Lo lee `stop.max_generations` como el tope que impone, y de forma independiente lo lee la mutación no uniforme de Michalewicz como el denominador de progreso de su propio schedule — usado por ambos incluso cuando la corrida termina por otra condición de corte |

**Objeto `stop`** — tiene que haber al menos una condición habilitada, o la validación de la
configuración rechaza la corrida de plano por estar configurada para no terminar nunca:

| Clave | Tipo | Significado |
|---|---|---|
| `max_generations` | bool | Habilita el tope de generaciones basado en `horizon`. **Si se omite por completo de la configuración, su valor por defecto es `false`** — una configuración que deja esta clave afuera no obtiene el tope de generaciones de regalo; hay que declararlo explícitamente como `true` para que esté activo. |
| `wall_clock_seconds` | number | Cortar una vez transcurridos estos segundos |
| `min_fitness` | number en (0, 1] | Cortar una vez que el mejor fitness alcanza este valor (STP-03, "solución aceptable") |
| `content_stagnation.window` / `.tolerance` | int / number | Cortar una vez que el mejor fitness haya variado menos que `tolerance` a lo largo de las últimas `window` generaciones (STP-04) |
| `structure_stagnation.window` / `.fraction` / `.tolerance` | int / number / number | Cortar una vez que al menos una `fraction` de la población haya permanecido "sin cambios" (dentro de `tolerance` de distancia genética) durante `window` generaciones consecutivas (STP-05) |

Cuando varias condiciones de corte se disparan en la misma generación, gana `max_generations`,
después `wall_clock`, después `min_fitness`, después `content_stagnation` y por último
`structure_stagnation` — un orden de prioridad fijo, no arbitrario.

**`parents` / `replacement`** — ambas son ranuras de selección y aceptan exactamente la misma
forma; el MISMO conjunto de métodos está disponible en cualquiera de las dos (`replacement` es
el selector de la nueva generación, y la cátedra permite que reutilice cualquiera de los métodos
de selección de padres):

| Clave | Aplica a | Significado |
|---|---|---|
| `method` | todas | Uno de los 9 nombres de selección registrados más abajo |
| `t0`, `tc`, `k` | solo `boltzmann` | Temperatura inicial, temperatura final, constante de decaimiento |
| `m` | solo `tournament_deterministic` | Tamaño del torneo |
| `threshold` | solo `tournament_probabilistic` | `Threshold ∈ [0.5, 1]` |
| `method_1`, `method_2`, `coefficient` | solo `blend`, reemplaza a `method` | Dos specs de selección anidados, mezclados por `coefficient ∈ [0,1]` |

**`crossover`:**

| Clave | Significado |
|---|---|
| `method` | Uno de los 4 nombres de cruza registrados más abajo |
| `boundary` | `"gene"` o `"triangle"` — dónde puede caer un corte: en un locus de gen individual, o solo sobre un límite de triángulo de 11 genes. Aplica a los cuatro métodos de cruza. |
| `p` | solo `uniform` — probabilidad de intercambio por locus, por defecto `0.5` |

**`mutation`:**

| Clave | Significado |
|---|---|
| `method` | Uno de los 4 nombres de mutación registrados más abajo |
| `probability` | Probabilidad de mutación `Pm` |
| `schedule` | `"uniform"` o `"non_uniform"` (Michalewicz) — aceptado por los cuatro métodos de mutación |
| `b` | Parámetro de forma de Michalewicz — solo tiene sentido bajo `schedule: "non_uniform"` |
| `m` | solo `multigen_limited` — cota superior de la cantidad de genes sorteados por mutación |
| `sigma` | Objeto opcional que sobrescribe el tamaño de paso por defecto según el tipo de gen, con cualquiera de las claves `coordinate` / `color` / `alpha`; una clave omitida de `sigma` conserva su valor por defecto incorporado en vez de tratarse como cero |

**`survival`:**

| Clave | Significado |
|---|---|
| `method` | Uno de los 3 nombres de supervivencia registrados más abajo |
| `gap` | solo `generational_gap` — `G ∈ [0,1]`, sin sentido para los otros dos métodos |

### Nombres de operadores registrados

Los 20 nombres de abajo (9 de selección + 4 de cruza + 4 de mutación + 3 de supervivencia) se
chequean por completitud contra el registro vivo: acá está documentado todo nombre que el motor
realmente acepta, y nada de lo que está acá es un nombre que el motor no acepte.

**Selección (9):**

- `elite` — Elite: ordenados por fitness, el individuo en el puesto `i` se copia
  `n(i) = ceil((K-i)/N)` veces; los mejores individuos se toman *varias* veces siempre que
  `K > N` (no es un simple "ordenar y tomar los primeros K").
- `random` — Muestreo Aleatorio: elige de forma uniforme al azar, ignorando el fitness por
  completo.
- `blend` — Compone dos métodos de selección mediante un coeficiente `A` (`coefficient`); un
  extra opcional aprobado, más allá del conjunto que exige el enunciado (SEL-09).
- `roulette` — Ruleta: `K` sorteos independientes contra la ruleta de fitness relativo acumulado.
- `universal` — Universal (SUS): la misma ruleta acumulada que `roulette`, pero los `K` sorteos
  se estratifican a partir de un único offset aleatorio (`r_j = (r+j)/K`) en vez de ser `K`
  sorteos independientes.
- `ranking` — Ranking: pseudo-fitness `f'(i) = (N-rank(i))/N` según el puesto por fitness, y
  después ruleta sobre ese pseudo-fitness.
- `boltzmann` — Boltzmann (Entrópica): pseudo-fitness `ExpVal(i,g,T)` (un valor esperado contra
  el promedio poblacional de la generación), y después ruleta; la temperatura sigue
  `T(t) = Tc + (T0-Tc)·e^(-kt)`.
- `tournament_deterministic` — Torneo Determinístico: se eligen `M` individuos al azar, se
  conserva el mejor, se repite `K` veces.
- `tournament_probabilistic` — Torneo Probabilístico: se eligen 2 individuos al azar, se conserva
  el más apto con probabilidad `Threshold ∈ [0.5,1]` (si no, se conserva el menos apto), se
  repite `K` veces.

**Cruza (4):** las cuatro aceptan la clave compartida `boundary` — `"triangle"` corta solo en
límites de triángulo de 11 genes, `"gene"` corta en loci de genes individuales.

- `one_point` — Un Punto: un corte aleatorio; todo lo que va del corte en adelante se intercambia
  entre los padres.
- `two_point` — Dos Puntos: dos cortes aleatorios; se intercambia el segmento que queda entre
  ellos.
- `ring` — Anular: una posición de inicio y una longitud aleatorias, que envuelven el final del
  cromosoma, definen el segmento intercambiado.
- `uniform` — Uniforme: cada locus se intercambia de forma independiente con probabilidad `p`
  (por defecto 0.5) — según CATEDRA.md, la única de las cuatro cruzas que no preserva la
  correlación posicional entre alelos.

**Mutación (4):** las cuatro aceptan la clave compartida `schedule: uniform | non_uniform`
(Michalewicz).

- `gene` — Gen: se altera un único gen, con probabilidad `Pm`.
- `multigen_limited` — Multigen Limitada: se sortea una cantidad aleatoria de genes en `[1, m]`
  para mutar, cada uno con probabilidad `Pm`.
- `multigen_uniform` — Multigen Uniforme: cada gen tiene, de forma independiente, probabilidad
  `Pm` de mutar.
- `complete` — Completa: con probabilidad `Pm`, mutan todos los genes del individuo.

**Supervivencia (3):**

- `additive` — Supervivencia Aditiva: selecciona `N` individuos de la unión de los `N` padres y
  los `K` hijos.
- `exclusive` — Supervivencia Exclusiva: si `K > N`, selecciona `N` de entre los `K` hijos
  únicamente; si `K <= N`, toma los `K` hijos completos más `N-K` seleccionados de los padres
  (el límite de la rama es estrictamente `K > N`, no `K >= N`).
- `generational_gap` — Brecha Generacional `G`: la nueva generación son `(1-G)*N` individuos
  arrastrados de la generación anterior más `G*N` provenientes de los hijos.

9 (selección) + 4 (cruza) + 4 (mutación) + 3 (supervivencia) = 20 nombres, chequeados contra
`SELECTION.names() | CROSSOVER.names() | MUTATION.names() | SURVIVAL.names()` — no simplemente
afirmados de memoria.
