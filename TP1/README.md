# Grid World — TP1

Juego Grid World para el TP1 de *Sistemas de Inteligencia Artificial* del ITBA, Ejercicio 2 (Lado A). Una cuadrícula de $N \times M$ contiene obstáculos negros, autos numerados y banderas numeradas; el objetivo es mover un auto a la vez hasta que todos los autos queden estacionados sobre su bandera correspondiente.

El proyecto incluye el motor de reglas, una ventana de Pygame para juego manual, algoritmos de búsqueda no informados e informados, visualizador paso a paso y herramientas de benchmarking.

## Requisitos

- Python 3.10–3.13 (desarrollado y probado en 3.12)
- [pygame](https://www.pygame.org/) 2.6.1

## Instalación

Desde el directorio `TP1`:

```sh
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Ejecución de Algoritmos de Búsqueda

Grid World soporta algoritmos de búsqueda **no informados** e **informados**, con visualizadores interactivos en Pygame, generación automatizada de gráficos comparativos/benchmarks y ejecución en modo headless.

### Algoritmos Soportados

| Tipo | Algoritmo | Identificador CLI (`--algo`) | Descripción |
|------|-----------|------------------------------|-------------|
| **No Informado** | Breadth-First Search (BFS) | `bfs` | Expande primero los nodos menos profundos; garantiza costo óptimo. |
| **No Informado** | Depth-First Search (DFS) | `dfs` | Expande primero los nodos más profundos; no óptimo, rápida exploración. |
| **No Informado** | Iterative Deepening DFS (IDDFS) | `iddfs` | Combina la eficiencia de memoria de DFS con la optimalidad en profundidad de BFS. |
| **Informado** | Greedy Best-First Search (Avara) | `greedy` | Expande los nodos más cercanos a la meta según $h(n)$; rápido, no óptimo. |
| **Informado** | A* Search | `astar` | Minimiza $f(n) = g(n) + h(n)$; óptimo y completo con heurísticas admisibles. |

### Heurísticas Disponibles (para Greedy y A\*)

| Heurística | Identificador CLI (`--heuristic`) | Descripción | ¿Admisible? |
|------------|-----------------------------------|-------------|:-----------:|
| **Suma de Distancias Manhattan** | `manhattan` *(o `heuristic_a`)* | Suma de las distancias Manhattan ($L_1$) de cada auto sin estacionar a su bandera. *(Por defecto)* | ✅ Sí |
| **Máxima Distancia Manhattan** | `max_manhattan` *(o `heuristic_b`)* | Máxima distancia Manhattan entre todos los autos sin estacionar a sus banderas. | ✅ Sí |
| **Suma de Distancias Euclidianas** | `euclidean_distance` | Suma de las distancias euclidianas directas ($L_2$) de cada auto sin estacionar a su bandera. | ✅ Sí |

---

### 1. Visualización y Replay de Búsqueda (Paso a Paso en Pygame)

Utiliza `scripts/replay_search.py` para ejecutar cualquier algoritmo sobre un nivel y reproducir visualmente tanto la exploración de nodos como el camino de solución paso a paso en la interfaz gráfica de Pygame:

```sh
python scripts/replay_search.py --algo <bfs|dfs|iddfs|greedy|astar> [ruta_nivel] [--heuristic <manhattan|max_manhattan|euclidean_distance>] [--delay <ms>]
```

#### Ejemplos de ejecución:

- **Breadth-First Search (BFS):**
  ```sh
  python scripts/replay_search.py --algo bfs levels/01-warmup.json
  python scripts/replay_search.py --algo bfs levels/02-classic.json
  ```

- **Depth-First Search (DFS):**
  ```sh
  python scripts/replay_search.py --algo dfs levels/01-warmup.json
  python scripts/replay_search.py --algo dfs levels/01-warmup.json --delay 100
  ```

- **Iterative Deepening DFS (IDDFS):**
  ```sh
  python scripts/replay_search.py --algo iddfs levels/01-warmup.json
  ```

- **Greedy Best-First Search (Búsqueda Avara):**
  ```sh
  # Con Heurística Manhattan
  python scripts/replay_search.py --algo greedy levels/01-warmup.json --heuristic manhattan

  # Con Heurística Máxima Manhattan
  python scripts/replay_search.py --algo greedy levels/02-classic.json --heuristic max_manhattan

  # Con Heurística Euclidiana
  python scripts/replay_search.py --algo greedy levels/01-warmup.json --heuristic euclidean_distance
  ```

- **A\* Search:**
  ```sh
  # Heurística por defecto (Suma de distancias Manhattan)
  python scripts/replay_search.py --algo astar levels/01-warmup.json

  # Con Heurística Máxima Manhattan
  python scripts/replay_search.py --algo astar levels/01-warmup.json --heuristic max_manhattan

  # Con Heurística Euclidiana en nivel Classic
  python scripts/replay_search.py --algo astar levels/02-classic.json --heuristic euclidean_distance
  ```

**Controles en la ventana de Replay:**
- `Espacio`: Pausar / reanudar la animación.
- `Esc`: Cerrar la ventana.

---

### 2. Búsqueda Óptima en Vivo dentro del Juego (A\*)

Mientras juegas cualquier nivel de forma manual (`python -m gridworld [ruta_nivel]`):
- Presiona **`S`** para iniciar la búsqueda incremental con A* desde el estado actual del tablero.
- Los estados explorados se muestran como puntos azules mientras el HUD reporta en tiempo real los nodos expandidos y en frontera.
- Una vez encontrada la meta, el camino óptimo se anima y reproduce en amarillo.
- Presiona **`Espacio`** para pausar/reanudar y **`-` / `+`** para ajustar la velocidad de animación.

---

### 3. Generación de Gráficos y Benchmarks Comparativos

Para ejecutar benchmarks estadísticos y generar gráficos comparativos de alta resolución listos para presentación:

```sh
python scripts/generate_plots.py [ruta_nivel] [--runs N] [--out DIR] [--compare-all] [--skip-iddfs]
```

#### Ejemplos:

- **Benchmark de los 5 algoritmos en el nivel Warmup (5 corridas por algoritmo):**
  ```sh
  python scripts/generate_plots.py levels/01-warmup.json --compare-all
  ```

- **Benchmark en nivel Classic (omitiendo IDDFS por límite de profundidad):**
  ```sh
  python scripts/generate_plots.py levels/02-classic.json --skip-iddfs --runs 5
  ```

- **Comparativa específica de heurísticas (Greedy y A\* con distintas heurísticas):**
  ```sh
  python scripts/plot_informed_comparison.py
  ```

#### Gráficos exportados (en el directorio `plots/`):
- `*_desinformados.png`: Panel de 4 gráficos comparando BFS vs DFS vs IDDFS (Nodos expandidos, Frontera final, Tiempo de ejecución con barras de desvío estándar, Costo de solución).
- `*_frontera_max_vs_final.png`: Gráfico de barras agrupadas comparando Frontera Final vs Frontera Máxima (pico).
- `*_heuristicas_astar.png`: Panel de 4 gráficos comparando A* según la heurística (Manhattan vs Max Manhattan vs Euclidiana).
- `*_heuristicas_greedy.png`: Panel de 4 gráficos comparando Greedy según la heurística.
- `*_todos_los_algoritmos.png`: Comparativa integral de los 5 algoritmos.

---

### 4. Exportación de GIFs Animados

Para generar animaciones en formato GIF de la fase de exploración y la solución:

```sh
python scripts/generate_gifs.py [ruta_nivel] [--algo <algo>] [--heuristic <heuristic>] [--out-dir DIR] [--fps FPS]
```

#### Ejemplos:

```sh
python scripts/generate_gifs.py levels/01-warmup.json --algo astar --heuristic manhattan
python scripts/generate_gifs.py levels/02-classic.json --algo bfs --out-dir gifs/classic
```

---

### 5. Verificación Headless y Búsqueda Aleatoria

Para probar el motor de búsqueda en modo headless (sin inicializar Pygame):

```sh
python scripts/check_headless.py
```

Para ejecutar la demo de búsqueda aleatoria:

```sh
python scripts/replay_random_search.py levels/01-warmup.json
```

---

## Controles del Juego

| Tecla | Acción |
|-------|--------|
| `1`–`9` | Seleccionar un auto por su número (en juego), o elegir nivel (en pantalla de selección) |
| Flechas (`↑`, `↓`, `←`, `→`) | Mover el auto seleccionado una celda |
| `U` | Deshacer el último movimiento (*Undo*) |
| `R` | Reiniciar el nivel a su posición inicial (*Reset*) |
| `S` | Iniciar búsqueda óptima con A* desde la posición actual |
| `Espacio` | Pausar o reanudar la animación de búsqueda |
| `-` / `+` | Cambiar la velocidad de la animación |
| `Esc` | Salir del juego o volver atrás |

## Reglas del juego

- Exactamente un auto seleccionado se mueve una celda ortogonal por turno.
- Un movimiento es rechazado sin modificar el tablero si la celda destino:
  está fuera del tablero, es un obstáculo negro, está ocupada por otro auto o es una bandera de otro auto.
- Un auto que se mueve a su propia bandera numerada queda **estacionado** y no puede volver a seleccionarse ni moverse.
- El nivel se gana cuando todos los autos están estacionados en su bandera correspondiente; la pantalla de victoria muestra el total de movimientos realizados.
- Dado que los autos estacionados bloquean celdas y las banderas ajenas no se pueden atravesar, el orden de estacionamiento puede dejar el tablero en un estado sin solución. Cuando un auto sin estacionar ya no puede alcanzar su bandera, aparece una advertencia en pantalla sugiriendo usar `U` (deshacer) o `R` (reiniciar).

## Archivos de niveles

Los niveles son archivos JSON ubicados en `TP1/levels/`:

- `01-warmup.json` — Tablero 5×5, 2 autos.
- `02-classic.json` — Tablero 7×7, 3 autos.
- `03-gridlock.json` — Tablero 9×9, múltiples autos y obstáculos.

El formato detallado campo por campo, convención de coordenadas y validaciones están documentados en [`levels/SCHEMA.md`](levels/SCHEMA.md).

## Ejecución de tests

Desde el directorio `TP1`:

```sh
pytest
```
