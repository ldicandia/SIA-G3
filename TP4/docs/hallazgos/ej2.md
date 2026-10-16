# Ejercicio 2 — Memoria asociativa (Hopfield): hallazgos

Los números y gráficos de esta nota salen de corridas de `python -m tp4 run <config>` (los comandos
están al final de cada sección, en "Cómo regenerar"). Los valores citados se copian de
`runs/hopfield_letters/summary.json`, `runs/hopfield_qr/summary.json` y de los CSV de cada corrida, y
`python -m tp4.findings` verifica que cada archivo referenciado exista y que los números clave aparezcan
en el texto. Los decimales van con coma. La red de Hopfield está escrita a mano en numpy
(`tp4/models/hopfield.py`), sin ninguna librería de redes neuronales.

El enunciado recomienda probar primero el caso simple antes de pasar a las imágenes, así que la sección 2.2
(letras de 5×5) va **antes** que la 2.1 (códigos QR de 37×37). Además el argumento de los QR se apoya en lo que
mostramos con las letras: la ortogonalidad de los patrones y la capacidad de la red. Los títulos conservan la
numeración del enunciado (2.1 y 2.2) para poder ubicarlos después.

## 2.2 Recomendación: letras 5×5

Todo lo de esta sección sale de `runs/hopfield_letters/`, regenerado con los comandos de "Cómo regenerar (2.2)".
Hay decisiones que son criterio nuestro y el grupo todavía tiene que aceptar: están marcadas en "Supuestos (2.2)".

### Red de Hopfield implementada

**Escribimos la red de Hopfield a mano en numpy: pesos de Hebb con diagonal cero, actualización asincrónica o sincrónica, energía y tres criterios de corte.**

- Los patrones son vectores `ξ` de N neuronas con valores en {−1, +1}; para letras de 5×5, **N = 25** (`n_neurons` en
  `runs/hopfield_letters/summary.json`).
- **Pesos de Hebb:** `w_ij = (1/N)·Σ_μ ξ_i^μ ξ_j^μ` para `i ≠ j` y `w_ii = 0` (la matriz es simétrica).
- **Campo local y regla de actualización:** `h_i = Σ_j w_ij S_j` y `S_i = signo(h_i)`; si `h_i = 0` la neurona **conserva su estado**
  (así la regla no inventa un signo cuando no hay información).
- **Dos formas de actualizar:** *asincrónica* (se visitan las N neuronas en un orden aleatorio por barrido, una por vez, con la semilla de la config)
  y *sincrónica* (`S(t+1) = signo(W·S(t))`, todas juntas).
- **Energía:** `H(S) = −½·Σ_ij w_ij S_i S_j`.
- **Criterios de corte:** (1) un barrido sin ningún cambio (convergió), (2) en modo sincrónico, un ciclo de período 2 (`S(t+1) = S(t−1)`),
  (3) `max_iter` barridos (100 en esta corrida).
- **Qué garantiza pytest** (`tests/test_hopfield_model.py`, 16 tests): las letras G, R, T y V guardadas y patrones aleatorios guardados son puntos fijos (con las dos reglas de actualización),
  la energía **no sube** en ninguna actualización individual en modo asincrónico, el ejemplo resuelto del curso da el resultado de las diapositivas,
  una consulta resulta igual sola o dentro de un lote, y ningún módulo de `tp4/` importa una librería de redes.
- **Por qué asincrónico es el modo por defecto:** converge siempre (la energía baja en cada cambio y está acotada), mientras que el sincrónico puede
  oscilar entre dos estados. En las corridas de ruido de abajo, el asincrónico tuvo `n_async_cycles` = 0 ciclos y el sincrónico
  `n_sync_cycles` = 144 (de las 1600 consultas de cada modo sobre G, R, T y V; el conjunto peor no cicla nunca).

### Ortogonalidad: qué 4 letras guardar

**De 26 letras dibujadas a mano, las 4 más ortogonales son G, R, T y V: el producto interno más grande entre dos de ellas es 1, contra 21 en el peor conjunto.**

![Alfabeto de 5×5 con el conjunto elegido y el peor](../../runs/hopfield_letters/alphabet.png)

![Matriz de productos internos entre las 26 letras](../../runs/hopfield_letters/dot_products.png)

- La regla (`selection_rule` en `runs/hopfield_letters/summary.json`, y se fijó antes de mirar los resultados): *menor máximo |producto interno|
  fuera de la diagonal, luego menor promedio |producto interno|, luego orden alfabético.* Se evalúan los **14950** subconjuntos de 4 letras
  (`n_subsets` = 14950, es decir C(26, 4)) con la matriz de `runs/hopfield_letters/dot_products.csv`; el ranking completo está en `runs/hopfield_letters/orthogonality.csv`.
- Dos patrones de 25 píxeles tienen producto interno 25 consigo mismos y un valor impar entre −25 y 25 con otro; el valor mínimo posible de |producto interno|
  entre dos letras distintas es 1, y el conjunto elegido lo alcanza en los seis pares.

**Letras almacenadas:** G, R, T y V, con `selected_max_abs_dot` = 1 y `selected_mean_abs_dot` = 1,0 (los seis pares valen ±1).

- **Contraste con el peor conjunto** (`worst_letters` = H, M, N, W): `worst_max_abs_dot` = 21 y promedio 18,3333. Las cuatro tienen los dos palos
  verticales de los costados en común (columnas 1 y 5 completas o casi), así que sus patrones se parecen mucho entre sí.
- **Por qué importa la ortogonalidad:** al evaluar `h_i` sobre un patrón guardado `ξ^ν`, el término de interés vale `ξ_i^ν` pero aparece además un
  término de interferencia (*crosstalk*, diapositiva 33 del curso) que suma los productos internos con los demás patrones. Con productos internos
  chicos esa interferencia no alcanza para cambiar un signo; con productos grandes sí.
- **Margen de estabilidad** (`min_i ξ_i·h_i`, positivo si todos los píxeles son estables; `runs/hopfield_letters/stored.csv`):

| Conjunto | Letras | Márgenes | Puntos fijos |
|----------|--------|----------|--------------|
| Elegido | G, R, T, V | 0,72; 0,72; 0,72; 0,72 (`selected_stability_margins`) | `selected_n_fixed_points` = 4 de 4 |
| Peor | H, M, N, W | −1,2; −1,36; 0,16; −1,36 | `worst_n_fixed_points` = 1 de 4 (solo N) |

- Con el peor conjunto, tres de las cuatro letras guardadas **ni siquiera son estables**: la red ya cambia píxeles de una H, una M o una W exactas.

### Capacidad de almacenamiento

**Con N = 25 la cota teórica de la red de Hebb es de 3,45 a 3,75 patrones; guardamos 4, por encima de las dos cotas, y funciona solo porque estas 4 letras son casi ortogonales.**

![Capacidad empírica: fracción de letras estables y de conjuntos con todas estables](../../runs/hopfield_letters/capacity.png)

- Cotas del curso: `capacity_0138` = 3,45 (0,138·N) y `capacity_015` = 3,75 (0,15·N). `n_store` = 4 queda por encima de las dos.
- Curva empírica (`runs/hopfield_letters/capacity.csv`): para cada p se toman 200 subconjuntos aleatorios de p letras de nuestro alfabeto y
  se cuenta qué fracción de las letras guardadas son puntos fijos y qué fracción de los conjuntos las tiene **todas** estables.

| p letras guardadas | Fracción de letras estables | Fracción de conjuntos con todas estables |
|--------------------|-----------------------------|-------------------------------------------|
| 1 | 1,0 | 1,0 |
| 2 | 1,0 | 1,0 |
| 3 | 0,9483 | 0,91 |
| 4 | 0,7525 | 0,465 |
| 5 | 0,572 | 0,14 |
| 6 | 0,4275 | 0,03 |
| 7 | 0,3064 | 0,0 |
| 8 | 0,2669 | 0,0 |
| 9 | 0,1967 | 0,0 |
| 10 | 0,1485 | 0,0 |
| 4 (elegidas G, R, T, V) | 1,0 | 1,0 |
| 4 (peores H, M, N, W) | 0,25 | 0,0 |

- Con 4 letras **al azar**, en el 46,5% de los conjuntos todas las letras son estables; con las 4 elegidas por ortogonalidad, siempre. Con las peores, solo una de cuatro.
- **Conclusión:** la capacidad que importa depende de cuán correlacionados estén los patrones, no solo de cuántos sean. Con nuestro alfabeto la memoria
  sostiene 4 letras únicamente porque las elegimos casi ortogonales. Esta curva depende de nuestro alfabeto (ver Supuestos).

### Recuperación de letras con ruido

**Las 4 letras se recuperan sin errores hasta 3 píxeles invertidos de 25 (nivel 0,12) y la recuperación cae a menos de la mitad desde 9 píxeles; el conjunto peor nunca pasa de 0,25.**

![Recuperación frente al nivel de ruido, conjunto elegido y peor, asincrónico y sincrónico](../../runs/hopfield_letters/recovery_vs_noise.png)

![Ejemplos de recuperación a nivel 0,2](../../runs/hopfield_letters/recall_examples.png)

Conjunto G, R, T, V, 50 consultas por letra y nivel (200 por celda), cada una con ese número de píxeles invertidos al azar
(`runs/hopfield_letters/recall_rates.csv`; "recuperada" = el estado final es exactamente la letra de origen):

| Nivel | Píxeles | Async: recuperada | Async: espúrea | Async: invertida | Sync: recuperada | Sync: espúrea | Sync: ciclos |
|-------|---------|-------------------|----------------|------------------|------------------|---------------|--------------|
| 0,04 | 1 | 1,0 | 0,0 | 0,0 | 1,0 | 0,0 | 0,0 |
| 0,08 | 2 | 1,0 | 0,0 | 0,0 | 1,0 | 0,0 | 0,0 |
| 0,12 | 3 | 1,0 | 0,0 | 0,0 | 1,0 | 0,0 | 0,0 |
| 0,16 | 4 | 0,98 | 0,02 | 0,0 | 1,0 | 0,0 | 0,0 |
| 0,2 | 5 | 0,915 | 0,085 | 0,0 | 0,95 | 0,05 | 0,015 |
| 0,28 | 7 | 0,645 | 0,315 | 0,02 | 0,77 | 0,225 | 0,12 |
| 0,36 | 9 | 0,255 | 0,48 | 0,16 | 0,315 | 0,545 | 0,305 |
| 0,44 | 11 | 0,045 | 0,34 | 0,355 | 0,0 | 0,455 | 0,28 |

- `selected_max_level_full_recovery` = 0,12: hasta ese nivel el modo asincrónico recupera el 100% de las consultas.
- **1 y 2 píxeles invertidos se recuperan siempre.** No solo en estas 200 consultas por celda: un test exhaustivo
  (`tests/test_hopfield_letters.py`) prueba **todas** las consultas de 1 y 2 píxeles invertidos de G, R, T y V con 5 semillas del orden asincrónico y todas
  vuelven a su letra.
- A partir de 0,28 aparecen también consultas que terminan en **otra letra guardada** (0,02 a nivel 0,28; 0,105 a 0,36; 0,26 a 0,44) y en la **letra invertida**
  (estado `−ξ`, que también es un punto fijo de la red, porque la energía no cambia si se invierten todos los signos): con 11 píxeles invertidos de 25 el 35,5% de las consultas
  ya termina en el negativo de alguna letra.
- **Conjunto peor** (H, M, N, W; `worst_async_recovery`): queda clavado en 0,25 en todos los niveles hasta 0,36 y baja a 0,24 en 0,44. Es el 25% de las consultas que
  parten de N, la única letra estable: todas las consultas de H, M y W (0,75 es "otra letra") terminan en N (o en su negativo), porque sus propios patrones ni siquiera son puntos fijos.
  Esto se ve en `runs/hopfield_letters/recall.csv` (columna `final_letter`).
- **Asincrónico contra sincrónico:** el sincrónico recupera algo más en los niveles medios (0,77 contra 0,645 en 0,28), pero se queda sin recuperar nada en 0,44 y
  entra en ciclos de período 2 en hasta el 30,5% de las consultas (`n_sync_cycles` = 144 en total); el asincrónico nunca cicla. Elegimos el asincrónico como modo principal
  porque tiene garantía de convergencia aunque una tasa de recuperación algo menor en ruido medio.

### Un patrón muy ruidoso y un estado espúreo

**Una G con 10 de 25 píxeles invertidos termina en un estado estable que no es ninguna letra guardada ni su negativo: una mezcla de tres letras.**

![Consulta ruidosa, estado espúreo y las 4 letras guardadas](../../runs/hopfield_letters/spurious.png)

**Estado espúreo:** partiendo de una G con `spurious_n_flips` = 10 píxeles invertidos (nivel 0,4, "muy ruidoso"), la red converge a un estado cuya letra guardada más cercana es la G (`spurious_nearest_letter` = G; la V empata con la misma superposición y el desempate elige la primera) pero a distancia de Hamming `spurious_hamming_to_nearest` = 6 de ella, con superposiciones con (G, R, T, V) de 0,52; -0,2; -0,44 y 0,52 (`spurious_overlaps`) y energía `spurious_energy` = -7,68, **más alta** que la de cada letra guardada (-10,56 en las cuatro, `stored_energies`).

- **Es espúreo y no invertido:** no coincide con ninguna de las 4 letras ni con ninguna de sus negativas (se clasifica por igualdad exacta de los 25 píxeles), y es un **punto fijo**:
  la corrida lo confirma porque ninguna neurona cambia en el último barrido. Es un mínimo local de energía más alto que el de las letras.
- **Es una mezcla:** el estado es el signo de `+G −T +V` (`spurious_mixture`), una combinación de tres letras guardadas con signos (un estado de mezcla), que es el tipo de espúreo esperable
  cuando los patrones no son del todo ortogonales. De hecho **las 16 mezclas de tres letras con signo son puntos fijos** (`n_stable_mixtures` = 16 de 16, `runs/hopfield_letters/mixtures.csv`).
- **Con qué frecuencia pasa:** con 10 píxeles invertidos, el 39,5% de las consultas del conjunto elegido (200 consultas, asincrónico) terminan en un estado espúreo
  (`spurious_rate` = 0,395) y otro 24% en una letra invertida (`inverted_rate_at_spurious_noise` = 0,24). Es el ejemplo de la primera consulta espúrea que converge, en el orden
  (letra, intento) con la semilla fija: otra semilla mostraría otro igual de válido.
- **Energía:** el estado espúreo y las letras tienen energías distintas, y la curva de energía por actualización individual no sube nunca
  (`energy_never_increased` = true):

![Energía H por actualización individual de neurona](../../runs/hopfield_letters/energy.png)

### Supuestos (2.2)

- **El alfabeto es nuestro:** las 26 letras de 5×5 están dibujadas a mano en `datos/letras_5x5.json` (la J se copió del enunciado), no las dio el curso. Los productos internos, el
  conjunto elegido y la curva de capacidad dependen de esos dibujos.
- **Regla de ortogonalidad:** "más ortogonal" significa menor máximo |producto interno| entre pares, luego menor promedio, luego orden alfabético. Es criterio nuestro y se fijó antes de ver los resultados.
- **Definición de espúreo:** un punto fijo que no es ninguna letra guardada **ni** su negativa; las letras invertidas se cuentan aparte. El ejemplo mostrado es la primera consulta espúrea que termina convergida.
- **"Muy ruidoso" = 10 de 25 píxeles (nivel 0,4):** es una elección nuestra (el enunciado no fija un número); con ese nivel el 39,5% de las consultas termina en un estado espúreo.
- **Orden asincrónico:** una permutación aleatoria de las 25 neuronas por barrido, compartida por todas las consultas de un lote, con semilla fija; los resultados no dependen de cómo se agrupen las consultas.
- **Iteraciones:** se cuentan solo los barridos que cambiaron el estado; el último barrido sin cambios es el chequeo de convergencia y no suma.
- **Capacidad empírica:** 200 subconjuntos aleatorios por p (hasta p = 10) y la misma semilla; depende del alfabeto.
- **Solo numpy:** ni la red ni el análisis usan librerías de redes neuronales; la regla de Hebb es la del curso.
- Los gráficos se enlazan desde `runs/` (se regeneran, no se versionan). Copiarlos a `docs/` queda para la fase de documentación.

### Cómo regenerar (2.2)

```bash
.venv/bin/python -m tp4 run configs/hopfield_letters.json
```

Verificación de la evidencia (los 21 valores de 2.2):

```bash
.venv/bin/python -m tp4.findings docs/hallazgos/ej2.md \
  --cite runs/hopfield_letters/summary.json:selected_letters \
  --cite runs/hopfield_letters/summary.json:selected_max_abs_dot \
  --cite runs/hopfield_letters/summary.json:selected_mean_abs_dot \
  --cite runs/hopfield_letters/summary.json:worst_letters \
  --cite runs/hopfield_letters/summary.json:worst_max_abs_dot \
  --cite runs/hopfield_letters/summary.json:n_subsets \
  --cite runs/hopfield_letters/summary.json:capacity_0138 \
  --cite runs/hopfield_letters/summary.json:capacity_015 \
  --cite runs/hopfield_letters/summary.json:selected_n_fixed_points \
  --cite runs/hopfield_letters/summary.json:worst_n_fixed_points \
  --cite runs/hopfield_letters/summary.json:selected_stability_margins \
  --cite runs/hopfield_letters/summary.json:selected_max_level_full_recovery \
  --cite runs/hopfield_letters/summary.json:n_sync_cycles \
  --cite runs/hopfield_letters/summary.json:spurious_n_flips \
  --cite runs/hopfield_letters/summary.json:spurious_rate \
  --cite runs/hopfield_letters/summary.json:spurious_letter \
  --cite runs/hopfield_letters/summary.json:spurious_nearest_letter \
  --cite runs/hopfield_letters/summary.json:spurious_hamming_to_nearest \
  --cite runs/hopfield_letters/summary.json:spurious_overlaps \
  --cite runs/hopfield_letters/summary.json:spurious_energy \
  --cite runs/hopfield_letters/summary.json:n_stable_mixtures
```
