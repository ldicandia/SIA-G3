# Ejercicio 1b: Denoising Autoencoder

Análisis del Ejercicio 1b del TP5 (Sistemas de Inteligencia Artificial, ITBA). La consigna pide (1b.1) entrenar un autoencoder con la instancia distorsionada como entrada y el patrón original como salida esperada, y explicar por qué se eligió la arquitectura y el ruido de entrenamiento, y (1b.2) mostrar ejemplos de patrones distorsionados y analizar hasta qué nivel de ruido el autoencoder logra eliminarlo. Todos los números de este documento salen de la salida de `cargo run --release -- ej1b` y de `cargo run --release -- dae-sweep`, y el test `tests/dae_doc_consistency.rs` verifica que coincidan con lo que produce el código.

Para regenerar todo: `cargo run --release -- dae-sweep` (estudio de arquitecturas y ruido de entrenamiento, tablas y curvas), `cargo run --release -- ej1b` (entrenamiento de la configuración elegida, tabla por nivel, gráficos y ejemplos) y `cargo run --release -- noise` (letras distorsionadas por nivel, sin entrenar nada).

## Arquitectura elegida y justificación

La configuración elegida es `noise-fix0.20-s1`. Su arquitectura es `35-64-32-8-32-64-35`: un encoder de dos capas ocultas de 64 y 32 neuronas, un espacio latente de 8 dimensiones y un decoder con la distribución invertida. Las capas ocultas usan activación tanh, la capa latente es la identidad y la salida es sigmoide. Se entrena con entropía cruzada binaria (BCE) y Adam con tasa 0.005 (beta1 0.9, beta2 0.999, epsilon 1e-8), 3000 épocas por lote completo, con 8 copias de los 32 patrones por época (256 filas por época), semilla 1 y un ruido de entrenamiento fijo de 0.20: en cada copia y en cada época cada pixel se invierte con probabilidad 0.20. La línea que imprime `dae-sweep` al aplicar la regla de selección es:

```
SELECCION: noise-fix0.20-s1 (variante noise-fix0.20, puntaje mediano 1.6790, 3/3 semillas elegibles)
```

**Por qué un denoising autoencoder se entrena así.** En la teoría de la materia (DAE, Denoising Autoencoder) la red recibe la muestra contaminada con ruido, caracterizado numéricamente (por ejemplo sal y pimienta) o con una distribución de probabilidad, y la salida que se le pide es el dato original sin alterar. Como la estructura encoder/decoder intenta preservar el contenido de información más relevante del conjunto de datos, lo que no es parte del patrón (el ruido) no puede atravesarla, y una muestra nueva contaminada se reemplaza por la salida del decoder. Acá el ruido es la versión para imágenes binarias de sal y pimienta: cada pixel se invierte de forma independiente con probabilidad `p` (el nivel).

**Por qué el ruido se sortea de nuevo en cada época.** Si se entrenara con una única copia ruidosa de cada patrón, la red podría memorizar esa copia; con ruido nuevo en cada época y 8 copias por época con máscaras distintas, el único regularizador posible es el patrón limpio que se repite detrás de miles de versiones distintas, y eso es lo que el encoder aprende a conservar. Las curvas de pérdida son por eso ruidosas y no decrecen de forma suave: la pérdida se mide sobre un lote distinto en cada época.

**Por qué importa el rango del ruido de entrenamiento.** Lo que la red ve durante el entrenamiento define para qué niveles aprende a corregir. Un ruido de entrenamiento más bajo que el que se evalúa deja a la red sin experiencia con entradas tan dañadas; uno muy alto la obliga a trabajar con entradas casi sin información y puede perjudicar los niveles bajos. El estudio de ruido de entrenamiento (siguiente sección) lo mide: el calendario fijo en 0.20 obtiene el menor puntaje (1.6790), por debajo de uno fijo en 0.10 (1.9611) y de los uniformes 0.0-0.20 (1.8328), 0.0-0.30 (1.7702) y 0.0-0.40 (1.7911). Entre esos cinco, el que mejor corrige en el rango de niveles con el que se puntúa (0.05 a 0.30) es el que entrena siempre con un ruido de 0.20, en el medio del rango. Esa elección no cambia el nivel máximo eliminado que se informa más abajo (0.10 con el criterio de este trabajo), pero baja los pixeles incorrectos a niveles 0.20 y 0.30.

**Por qué el tamaño latente se estudia y no se fija en 2.** En el Ej 1a la consigna impone una capa latente de 2 neuronas; en el 1b no, así que el tamaño del cuello es un factor del diseño. Las filas del estudio muestran que 2 dimensiones alcanzan para reconstruir los 32 patrones limpios (las 3 semillas de `lat-l2` son elegibles) pero cuesta eliminar ruido: puntaje 2.6049 con 2 dimensiones, 2.1183 con 4, 1.8328 con 8 y 1.8167 con 16. La mejora se aplana a partir de 8, y entre 8 y 16 la diferencia (0.0161) es menor que la dispersión entre semillas de una misma variante, así que se eligió 8 por ser el cuello más chico que ya está en la zona plana. Con la intuición de la teoría: un cuello más ancho conserva más información del patrón y por lo tanto deja más margen para separar un pixel invertido del contenido real.

**Por qué BCE con Adam.** Se reutilizó la receta que convergió en el Ej 1a (`docs/ej1a-analisis.md`): allí los optimizadores SGD y Momentum con tasas altas no llegaron a reconstruir los patrones (0 de 3 semillas con SGD) mientras que Adam sí llegó a reconstruirlos en la mayoría de las semillas, y la entropía cruzada con salida sigmoide es la pérdida natural para pixeles binarios. Volver a repetir los estudios de pérdida y de optimizador para el DAE habría multiplicado el barrido sin aportar a lo que pide el 1b, que es el efecto de la arquitectura y del ruido.

**Por qué varias semillas y una regla escrita como código.** Cada variante se entrena con 3 semillas (0, 1 y 2) y la selección es una función (`select_dae_variant`): una variante es elegible si en todas sus semillas los 32 patrones limpios vuelven con a lo sumo 1 pixel incorrecto; entre las elegibles gana la de menor mediana de puntaje; un empate lo gana la anterior en la grilla; dentro de la ganadora se toma la semilla de menor puntaje (en `noise-fix0.20`, la semilla 1). El puntaje es la media de los pixeles incorrectos de salida en los niveles 0.05, 0.10, 0.15, 0.20 y 0.30, medidos con un ruido de selección distinto del de entrenamiento y del de las figuras finales, para que ni el entrenamiento ni el reporte participen de la decisión.

## Estudio de arquitecturas y ruido de entrenamiento

El barrido (`cargo run --release -- dae-sweep`) entrena 13 variantes con 3 semillas cada una (39 corridas) en tres estudios, cambiando un solo factor respecto de una base de ocultas `[64, 32]`, latente 8 y ruido de entrenamiento uniforme entre 0.0 y 0.2 (todas con tanh, latente identidad, salida sigmoide, BCE, Adam 0.005, 3000 épocas y 8 copias por época):

1. **Tamaño latente** (`lat`): latente de 2, 4, 8 y 16 dimensiones con ocultas `[64, 32]`.
2. **Capacidad del encoder** (`hid`): ocultas `[24]`, `[32, 16]`, `[64, 32]` y `[48, 24, 12]` con latente 8 (el decoder refleja al encoder).
3. **Calendario de ruido de entrenamiento** (`noise`): fijo en 0.10, fijo en 0.20 y uniforme entre 0.0 y 0.2, 0.3 y 0.4, con ocultas `[64, 32]` y latente 8.

El puntaje de una corrida es la media de los pixeles incorrectos de la salida (contra el patrón limpio, binarizando en 0.5) en los niveles 0.05 a 0.30 sobre 100 distorsiones por patrón y por nivel del ruido de selección; menor es mejor. La condición de elegibilidad es la del Ej 1a aplicada a los patrones limpios (nivel 0.00). Las tablas siguientes son `output/ej1b/dae_comparison.md` (con los títulos bajados a nivel 3). Los resultados completos, una fila por corrida, están en `output/ej1b/dae_sweep.csv`, y las curvas de pérdida de entrenamiento de la primera semilla de cada variante, una imagen por estudio, en `output/ej1b/dae_curves_lat_loss.png`, `output/ej1b/dae_curves_hid_loss.png` y `output/ej1b/dae_curves_noise_loss.png`.

### Estudio: lat

| variante | semillas elegibles | puntaje (mediana) | px a nivel 0.10 | px a nivel 0.20 | px a nivel 0.30 | dentro de 1 px a nivel 0.10 (%) | perdida final (mediana) |
|---|---|---|---|---|---|---|---|
| lat-l2 | 3/3 | 2.6049 | 0.437 | 3.122 | 7.920 | 93.2 | 2.644774 |
| lat-l4 | 3/3 | 2.1183 | 0.227 | 2.423 | 6.968 | 95.7 | 1.594742 |
| lat-l8 | 3/3 | 1.8328 | 0.189 | 2.016 | 6.237 | 96.2 | 1.295606 |
| lat-l16 | 3/3 | 1.8167 | 0.175 | 1.992 | 6.220 | 96.5 | 1.283148 |

### Estudio: hid

| variante | semillas elegibles | puntaje (mediana) | px a nivel 0.10 | px a nivel 0.20 | px a nivel 0.30 | dentro de 1 px a nivel 0.10 (%) | perdida final (mediana) |
|---|---|---|---|---|---|---|---|
| hid-h24 | 3/3 | 2.3453 | 0.457 | 2.877 | 7.010 | 90.4 | 2.405667 |
| hid-h32x16 | 3/3 | 1.9991 | 0.277 | 2.278 | 6.540 | 94.2 | 1.793430 |
| hid-h64x32 | 3/3 | 1.8328 | 0.189 | 2.016 | 6.237 | 96.2 | 1.295606 |
| hid-h48x24x12 | 3/3 | 1.9776 | 0.204 | 2.217 | 6.663 | 96.1 | 1.571604 |

### Estudio: noise

| variante | semillas elegibles | puntaje (mediana) | px a nivel 0.10 | px a nivel 0.20 | px a nivel 0.30 | dentro de 1 px a nivel 0.10 (%) | perdida final (mediana) |
|---|---|---|---|---|---|---|---|
| noise-fix0.10 | 3/3 | 1.9611 | 0.198 | 2.215 | 6.570 | 96.1 | 0.640769 |
| noise-fix0.20 | 3/3 | 1.6790 | 0.176 | 1.732 | 5.876 | 96.4 | 4.506564 |
| noise-u0-0.20 | 3/3 | 1.8328 | 0.189 | 2.016 | 6.237 | 96.2 | 1.295606 |
| noise-u0-0.30 | 3/3 | 1.7702 | 0.179 | 1.884 | 6.095 | 96.2 | 3.898678 |
| noise-u0-0.40 | 3/3 | 1.7911 | 0.199 | 2.001 | 5.980 | 95.6 | 7.409959 |

**Lectura de las tablas.** Las 13 variantes (39 corridas) resultaron elegibles: con cualquier arquitectura y calendario estudiados, los 32 patrones limpios vuelven con a lo sumo 1 pixel incorrecto, así que la diferencia entre variantes está en cuánto ruido corrigen y no en si aprenden los patrones.

- **Tamaño latente.** Es el factor que más separa los resultados: el puntaje baja de 2.6049 (2 dimensiones) a 2.1183 (4) y a 1.8328 (8). La dispersión entre las 3 semillas de una misma variante es mucho menor que esos saltos (por ejemplo, las semillas de `lat-l2` puntúan entre 2.5351 y 2.6135 en `output/ej1b/dae_sweep.csv`), de modo que son diferencias reales. En cambio, de 8 a 16 el puntaje pasa de 1.8328 a 1.8167, dentro de la dispersión entre semillas (las de `lat-l16` van de 1.7927 a 1.8420 y las de `lat-l8` de 1.8211 a 1.8508): no hay evidencia de que 16 sea mejor que 8.
- **Capacidad del encoder.** `hid-h24` (una sola capa oculta de 24) es la peor (2.3453), después `hid-h32x16` (1.9991) y `hid-h48x24x12` (1.9776), y la mejor es `hid-h64x32` (1.8328), cuyas semillas (1.8211 a 1.8508) no se solapan con las de `hid-h48x24x12` (1.9666 a 1.9969): el encoder de tres capas más angostas no supera al de dos capas más anchas. `hid-h32x16` y `hid-h48x24x12` están dentro de la dispersión entre semillas entre sí.
- **Ruido de entrenamiento.** `noise-fix0.20` (1.6790) es la mejor y sus semillas (1.6724 a 1.6921) no se solapan con las de ninguna otra variante de este estudio. `noise-u0-0.30` (1.7702) y `noise-u0-0.40` (1.7911) no se distinguen entre sí por la dispersión entre semillas. `noise-fix0.10` (1.9611) queda por detrás de los calendarios que llegan a 0.20 o más: entrenar con menos ruido del que se evalúa en la parte alta del rango perjudica. En la columna de 0.30, el calendario uniforme 0.0-0.40 tiene los menos pixeles incorrectos entre los uniformes (5.980), pero a 0.10 tiene 0.199, de los valores más altos del estudio: al ensanchar el rango se gana en lo alto y se pierde en lo bajo.
- **Las pérdidas finales de variantes con distinto calendario de ruido no son comparables entre sí.** La pérdida se mide sobre el lote ruidoso de entrenamiento de cada variante y un lote con más ruido tiene un piso de pérdida más alto aunque la red corrija mejor: `noise-fix0.20` tiene 4.5066 de pérdida y el mejor puntaje, y `noise-fix0.10` tiene 0.6408 y un puntaje peor. La curva de `output/ej1b/dae_curves_noise_loss.png` muestra esos pisos ordenados por la cantidad de ruido del lote (fijo 0.10 lo más abajo, uniforme 0.0-0.40 lo más arriba), no por la calidad de la corrección. En cambio, en los estudios `lat` y `hid` el ruido de entrenamiento es el mismo y la pérdida sí ordena las variantes igual que el puntaje (`output/ej1b/dae_curves_lat_loss.png` y `output/ej1b/dae_curves_hid_loss.png`): la curva de `lat-l2` queda por encima de las demás y las de `lat-l8` y `lat-l16` casi se superponen.

## Niveles de ruido y distorsiones

El nivel `p` es la probabilidad de invertir cada pixel (`v` pasa a `1 - v`), de forma independiente para los 35 pixeles de cada letra. El rango válido es de 0.0 a 0.5. El límite es 0.5 porque con esa probabilidad el valor de cada pixel de entrada es 0 o 1 con igual probabilidad sea cual sea el patrón: la entrada deja de llevar información sobre el patrón y ningún método podría recuperarlo. Con nivel 0.0 la entrada es el patrón limpio. La cantidad esperada de pixeles incorrectos de una entrada ruidosa es 35 veces el nivel (1.75 a 0.05, 3.5 a 0.10, 7.0 a 0.20, 17.5 a 0.50), y la tabla por nivel de la sección siguiente lo confirma sobre 3200 muestras (columna `entrada_px`).

`cargo run --release -- noise --levels 0.05,0.1,0.2` muestra ocho letras (a, b, e, g, k, s, w, z) distorsionadas a cada nivel, con la cantidad de pixeles invertidos de cada una:

```
originales:
.....  #....  .....  .###.  #....  .....  .....  .....
.###.  #....  .....  #...#  #....  .....  .....  .....
....#  #....  .###.  #..##  #..#.  .####  #...#  #####
.##.#  ###..  #...#  .##.#  #.#..  #....  #...#  ...#.
#..##  #..#.  #####  ....#  ##...  .###.  #.#.#  ..#..
#..##  #..#.  #....  ....#  #.#..  ....#  #.#.#  .#...
.##.#  ###..  .####  .###.  #..#.  ####.  .#.#.  #####
nivel 0.05 | pixeles invertidos: a=3 b=3 e=0 g=2 k=2 s=2 w=3 z=1
.....  #.#..  .....  .###.  #...#  .....  .....  .#...
.###.  #....  .....  #...#  #....  .....  .....  .....
##..#  .....  .###.  #..##  #..#.  #####  #...#  #####
.##.#  ###..  #...#  .##.#  #.#..  #....  #...#  ...#.
#..##  #....  #####  ...##  ###..  .#.#.  #..##  ..#..
#..##  #..#.  #....  .#..#  #.#..  ....#  #.###  .#...
.##..  ###..  .####  .###.  #..#.  ####.  .#.#.  #####
nivel 0.10 | pixeles invertidos: a=5 b=3 e=5 g=3 k=3 s=3 w=3 z=3
#....  #....  ##...  .###.  #...#  .....  ..#..  ..#..
.###.  #....  .....  #...#  #....  ....#  .....  ....#
....#  .....  .####  #..#.  #....  .####  #...#  #####
###.#  ###..  #...#  .##.#  #.#.#  #....  #....  ...#.
#..#.  ####.  ##.#.  #...#  ##...  .###.  #.#.#  .##..
#.#.#  #..#.  #....  .....  #.#..  #.#.#  #.###  .#...
.##.#  ###..  .####  .###.  #..#.  ####.  .#.#.  #####
nivel 0.20 | pixeles invertidos: a=7 b=8 e=6 g=6 k=6 s=5 w=4 z=5
```

Las mismas letras, con todos los niveles evaluados, están en `output/ej1b/noise_examples.png` (una fila `original` y una fila por nivel, con la cantidad de pixeles invertidos como pie de cada letra). A 0.05 las letras se leen casi sin esfuerzo; a 0.20 ya cuesta reconocer algunas a simple vista (la `b` con 8 pixeles invertidos); a partir de 0.30 cuesta reconocer la mayoría de las entradas.

## Eliminación de ruido por nivel

**Cómo se evalúa.** Para cada nivel se sortean 100 distorsiones nuevas de cada uno de los 32 patrones (3200 entradas por nivel) con un generador sembrado que no es el del entrenamiento ni el de la selección (semilla de prueba fija). La red reconstruye cada entrada, se binariza la salida en 0.5 y se cuentan los pixeles que difieren del patrón limpio; la misma cuenta se hace sobre la entrada ruidosa, para saber cuánto empeoraría el patrón sin autoencoder. El criterio de éxito es el del Ej 1a: a lo sumo 1 pixel incorrecto. Esto es lo que imprime `cargo run --release -- ej1b` (la configuración elegida entrenada y evaluada en los 8 niveles por defecto):

```
nivel  entrada_px  salida_px  max_px  dentro_1px_%
 0.00       0.000      0.000       0         100.0
 0.05       1.759      0.041       8          99.2
 0.10       3.476      0.182      17          96.4
 0.15       5.266      0.620      17          89.8
 0.20       6.914      1.579      20          76.8
 0.30      10.516      5.928      31          37.2
 0.40      14.033     10.964      31           9.3
 0.50      17.563     14.982      33           1.3
```

`entrada_px` y `salida_px` son los pixeles incorrectos medios de la entrada ruidosa y de la salida del autoencoder, `max_px` es el peor caso de la salida entre las 3200 muestras y `dentro_1px_%` es el porcentaje de muestras cuya salida queda con a lo sumo 1 pixel incorrecto. La línea de titular es:

```
NIVEL MAXIMO ELIMINADO: 0.10 (a ese nivel y a todos los menores al menos el 90 % de las distorsiones no vistas queda con a lo sumo 1 pixel incorrecto)
```

Los números se guardan en `output/ej1b/dae_levels.csv` y se grafican en `output/ej1b/dae_error_vs_noise.png` (pixeles incorrectos medios de la entrada y de la salida contra el nivel) y en `output/ej1b/dae_within_vs_noise.png` (porcentaje de muestras con a lo sumo 1 pixel incorrecto).

**Qué se ve en la tabla y en los gráficos.**

- **Nivel 0.00.** Se comporta como un autoencoder común: las 32 letras limpias vuelven con 0 pixeles incorrectos (100.0 %), aunque la configuración elegida entrena siempre con ruido de 0.20, es decir, casi nunca vio una entrada sin pixeles invertidos.
- **Niveles bajos (0.05 y 0.10).** El autoencoder elimina casi todo el ruido: de 1.759 pixeles incorrectos en la entrada a 0.041 en la salida a 0.05 (99.2 % de las muestras dentro de 1 pixel), y de 3.476 a 0.182 a 0.10 (96.4 %). En `output/ej1b/dae_error_vs_noise.png` la curva de la salida casi coincide con el eje horizontal hasta 0.10 mientras la de la entrada crece en línea recta. Aun así quedan casos extremos: el peor caso a 0.10 es una muestra con 17 pixeles incorrectos, una máscara especialmente destructiva.
- **Niveles medios (0.15 a 0.30).** La corrección empieza a fallar: el porcentaje dentro de 1 pixel baja de 89.8 % a 0.15 a 76.8 % a 0.20 y a 37.2 % a 0.30, y los pixeles incorrectos medios de la salida suben de 0.620 a 1.579 y a 5.928, todavía por debajo de los de la entrada (5.266, 6.914 y 10.516) pero ya en una proporción mucho menor: a 0.30 la salida conserva más de la mitad de los pixeles incorrectos de la entrada. La curva de la salida cambia de pendiente cerca de 0.20.
- **Niveles altos (0.40 y 0.50).** La salida queda cerca de la entrada: 10.964 contra 14.033 pixeles incorrectos a 0.40 y 14.982 contra 17.563 a 0.50. Solo el 9.3 % de las muestras queda dentro de 1 pixel a 0.40 y el 1.3 % a 0.50. A 0.50 ningún método puede recuperar el patrón porque la entrada no lo contiene; el 1.3 % que queda dentro de la tolerancia es azar (la red reconstruye una letra cualquiera que a veces coincide con la original), no eliminación de ruido. El porcentaje de la entrada sin autoencoder dentro de 1 pixel es 0.0 % a partir de 0.30 (columna `input_frac_within_1px` de `output/ej1b/dae_levels.csv`), de modo que la salida del autoencoder siempre mejora a la entrada, pero eso deja de ser útil en los niveles altos.
- **El titular y su umbral.** El nivel máximo eliminado es 0.10 porque a 0.15 el 89.8 % de las muestras queda dentro de 1 pixel, apenas debajo del umbral de 90 % que fijamos como convención. El umbral y el nivel no son una propiedad de la red: con el criterio de 85 % el titular sería 0.15, y entre 0.10 y 0.20 la curva cae de forma continua. Lo informativo es el perfil completo, no el número de la línea.

## Ejemplos visuales

Para cada nivel hay una imagen `dae_examples_p{nivel}.png` con tres filas sobre las mismas 8 letras (a, b, e, g, k, s, w, z): la fila `original`, la fila `entrada ruidosa` (exactamente las mismas distorsiones que muestra `output/ej1b/noise_examples.png`) y la fila `salida reconstruida` (la salida del autoencoder binarizada en 0.5), con los pixeles incorrectos contra el patrón limpio como pie de cada letra. Estas son 8 muestras elegidas de una vez, y sirven para ver qué hace la red; la afirmación estadística es la de la tabla anterior, que usa 3200 muestras por nivel.

- `output/ej1b/dae_examples_p0.00.png`: las tres filas coinciden (0 pixeles incorrectos en cada letra).
- `output/ej1b/dae_examples_p0.05.png`: las entradas tienen entre 0 y 3 pixeles invertidos y las 8 salidas son idénticas al original (0 px).
- `output/ej1b/dae_examples_p0.10.png`: las entradas tienen 3 a 5 pixeles invertidos y las 8 salidas siguen siendo exactas.
- `output/ej1b/dae_examples_p0.15.png`: entradas con 0 a 6 pixeles incorrectos (la `w` con 6) y las 8 salidas exactas. Este es el último nivel de los ejemplos en que las 8 letras vuelven exactas.
- `output/ej1b/dae_examples_p0.20.png`: es el primer nivel en que una letra vuelve mal: 7 de las 8 letras son exactas, pero la `b` (con 8 pixeles invertidos de entrada) sale como una forma incompleta con 6 pixeles incorrectos.
- `output/ej1b/dae_examples_p0.30.png`: fallan varias letras: `a` (17 pixeles incorrectos) sale como una forma distinta de la original, y `b` (9), `e` (7), `s` (7) y `z` (2) salen con errores; `g` queda con 1 pixel incorrecto (dentro de la tolerancia) y `k` y `w` exactas.
- `output/ej1b/dae_examples_p0.40.png`: ninguna de las 8 letras queda dentro de 1 pixel (entre 4 y 18 pixeles incorrectos). Casi todas las salidas son formas limpias con aspecto de letra pero que no son la letra original (la excepción es la `a`, que sale como un bloque irregular): la red proyecta la entrada hacia un patrón aprendido y se equivoca de patrón.
- `output/ej1b/dae_examples_p0.50.png`: las salidas (9 a 22 pixeles incorrectos) son formas de letra que no guardan relación con la letra de partida, como corresponde a una entrada sin información sobre el patrón.

La lectura conjunta de la tabla y de las imágenes: hasta 0.15 la eliminación del ruido es esencialmente total en las letras mostradas y supera el 89 % en la tabla; a 0.20 aparecen los primeros fallos (76.8 % dentro de 1 pixel), a 0.30 los fallos son frecuentes y a partir de 0.40 la red deja de poder recuperar la letra.

## Alcance y limitaciones

- Solo se estudió ruido simétrico de inversión de pixeles, la versión binaria de sal y pimienta. Otros tipos de ruido (por ejemplo, pixeles que solo se apagan, o corrimientos de la letra) no se midieron y los resultados no se pueden extrapolar a ellos.
- El conjunto son los 32 patrones de `font.h`: la red es un modelo de esos 32 patrones, y por eso, ante un ruido que la engaña, devuelve otro de ellos y no una letra "intermedia". Los resultados no dicen nada sobre otras fuentes o letras.
- Las distorsiones de evaluación se sortean con una semilla independiente de la del entrenamiento y de la selección, pero eso no las hace distintas máscara por máscara: con 35 pixeles solo existen 36 máscaras de 0 o 1 pixeles invertidos por patrón, de modo que a niveles muy bajos algunas máscaras de la evaluación coinciden necesariamente con máscaras que la red vio al entrenar. A niveles de 0.10 en adelante, casi todas las máscaras tienen varios pixeles invertidos y la coincidencia no es relevante.
- El criterio de a lo sumo 1 pixel incorrecto y el umbral de 90 % de muestras que definen el titular `NIVEL MAXIMO ELIMINADO` son convenciones de este trabajo (la tolerancia viene del Ej 1a, el umbral lo elegimos nosotros); con otro umbral el nivel cambia, y el perfil completo de la tabla es la información que no depende de él.
- Los resultados corresponden a redes elegidas entre tres semillas por una regla escrita como código; otra semilla daría números algo distintos (en el estudio, las semillas de una misma variante difieren en el puntaje en centésimas), y las diferencias menores a esa dispersión no se deben interpretar como mejoras.
- El estudio de arquitecturas es de un factor por vez alrededor de una base y no explora todas las combinaciones (por ejemplo, el calendario de ruido se estudió solo con ocultas `[64, 32]` y latente 8), ni repite los estudios de optimizador, pérdida y activación del Ej 1a.
