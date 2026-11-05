# Ejercicio 2: autoencoder variacional sobre emojis

Análisis del Ejercicio 2 del TP5 (Sistemas de Inteligencia Artificial, ITBA). La consigna pide (2a) elegir o construir un dataset de emojis, (2b) implementar un autoencoder variacional (VAE) para resolver el problema de la representación en el espacio latente y justificar la configuración elegida, y (2c) generar un emoji nuevo muestreando el espacio latente y juzgar si pertenece al dataset. Todos los números de este documento salen de `cargo run --release -- ej2`, que entrena el VAE y el autoencoder de contraste una sola vez y escribe las figuras en `output/ej2/`; una prueba (`tests/vae_doc_consistency.rs`) recalcula cada línea citada y falla si el documento difiere del código.

Para regenerar todo: `cargo run --release -- ej2` (el ejercicio completo, con veredicto de aceptación) y, por separado, `cargo run --release -- emojis` (dataset), `cargo run --release -- vae` (entrenamiento y una muestra del prior), `cargo run --release -- vae-sample` (búsqueda del emoji nuevo), `cargo run --release -- vae-manifold` (grilla del plano latente) y `cargo run --release -- vae-baseline` (contraste con el autoencoder simple y aceptación). Todas las corridas son deterministas por semilla y los comandos individuales imprimen los mismos números que `ej2` para la misma configuración.

## Dataset de emojis

**Respuesta a 2a.** La consigna permite elegir o construir el dataset; se construyó por código. Todas las imágenes comparten el mismo contorno de cara y combinan 6 estilos de ojos (`dots`, `happy`, `wide`, `wink`, `closed`, `x`) con 6 estilos de boca (`smile`, `grin`, `flat`, `frown`, `open`, `small`): 6 x 6 = 36 combinaciones. Cada imagen se dibuja con primitivas geométricas (anillos, cápsulas, discos) en una grilla de 24 x 24 píxeles con supersampling de 4 x 4, de modo que los valores son grises en [0, 1] (1 = tinta) y se aplana a un vector de 576 valores. No requiere descargar nada ni agrega ninguna dependencia, y es determinista. La corrida imprime:

```
emojis: 36 imagenes de 24x24 (576 valores en [0,1], 1 = tinta)
separacion: minima 7 px, mediana 7.5 px, radio de cercania 8 px
```

La distancia entre dos emojis es la cantidad de píxeles en que difieren tras binarizar en 0.5 (distancia de Hamming, la misma regla que usa el reporte del Ej 1). La línea `separacion:` resume la distancia de cada emoji a su vecino más cercano: el par más parecido difiere en 7 píxeles, la mediana es 7.5 y el radio de cercanía se define como la mediana redondeada hacia arriba (8 px). Estos números definen únicamente cuándo una imagen decodificada es reconocible (a lo sumo el radio de cercanía de un emoji del dataset); no definen cuándo es nueva, porque la distancia global a un emoji es del mismo orden que el ruido de reconstrucción del propio decodificador (ver más adelante).

El dataset es el producto completo de 6 estilos de ojos por 6 estilos de boca: no existe ninguna combinación de ojos y boca que esté ausente, de modo que un emoji nuevo no puede ser una combinación inédita de estilos conocidos.

La consecuencia es que la novedad solo puede ser una región (los ojos o la boca) que no sea ninguno de los 6 estilos de esa región, y por eso el Ej 2 juzga la novedad por región y no por distancia global a un emoji. Que toda combinación de un estilo de ojos con un estilo de boca ya es una fila del dataset lo comprueba la prueba `every_combination_of_known_styles_is_already_a_dataset_emoji` (`tests/vae_regions.rs`).

La figura `output/ej2/emojis.png` muestra el dataset como un montaje de 6 x 6 con una fila por estilo de ojos y una columna por estilo de boca: se ven las 36 caras con el mismo trazo y los grises de los bordes producidos por el supersampling. Hay que decirlo con claridad: son caras estilo emoji dibujadas en código, no las ilustraciones de los emojis de Unicode de ninguna plataforma.

## Arquitectura y esquema variacional

**Respuesta a 2b.** La configuración congelada es:

```
configuracion: vae-h128x32-l2-s0 | arquitectura: encoder 576-128-32-4 | decoder 2-32-128-576 | optimizador: Adam(lr=0.003) | reconstruccion: BCE + KL | epocas: 3000 | semilla de entrenamiento: 0
```

**El problema que resuelve el esquema variacional.** En la teoría de la materia (autoencoder generativo) se entrena un autoencoder, se deja de lado el encoder y se eligen valores de las variables latentes z para que el decoder genere muestras nuevas. El problema es lo que la teoría llama la conquista del espacio latente: un autoencoder común deja el espacio latente sin estructura, con regiones de códigos separadas por zonas donde el decoder produce cualquier cosa, y nada garantiza que un punto elegido produzca una muestra válida. La solución es darle al espacio latente una estructura estadística: en lugar de mirar la salida para un único punto z, mirar las salidas para un conjunto de puntos.

**Codificador estocástico.** Se asume un prior p(z) = N(0, I) y se aproxima la posterior con q(z|x) = N(mu(x), Sigma(x)) con Sigma diagonal. El encoder tiene por lo tanto 2 x 2 = 4 salidas: la media y el logaritmo de la varianza de cada una de las 2 dimensiones latentes (de ahí el `4` de `encoder 576-128-32-4`). Se predice el logaritmo de la varianza y se la obtiene con exp(.), parametrización de la teoría que mantiene la varianza positiva y evita problemas numéricos; la capa de salida del encoder es la identidad para que la media y el logaritmo de la varianza no estén acotados.

**Truco de reparametrización.** Muestrear z directamente de N(mu, Sigma) corta el camino del gradiente. Se sortea eps de N(0, I) y se calcula z = mu + eps * sigma con sigma = exp(0.5 * logvar): la aleatoriedad queda fuera de la red, la capa es diferenciable y el error de reconstrucción fluye hacia el encoder. En cada época se sortea un eps nuevo para los 36 emojis, de un flujo de números aleatorios independiente del de la inicialización de los pesos.

**Costo.** El costo es el negativo de la cota inferior variacional: el error de reconstrucción más la divergencia KL(q(z|x) || p(z)). Con las dos gaussianas la KL tiene forma cerrada, 0.5 * suma(exp(logvar) + mu^2 - 1 - logvar) sobre las dimensiones latentes, y se usa con peso 1. Las dos magnitudes se miden por imagen (suma sobre los píxeles o las dimensiones, media sobre los 36 emojis), en nats, de modo que son comparables y su suma es el costo que se minimiza.

**Por qué esta configuración.**

- **Latente de 2 dimensiones.** Hace que el espacio latente sea un plano que se puede decodificar en una grilla y mirar, como el autoencoder generativo de 2 variables de la teoría, y permite ver directamente qué estructura le dio el término KL.
- **Reconstrucción BCE con salida sigmoide.** Es la receta que convergió en los Ej 1a y 1b. El error cuadrático de la teoría es la otra instancia del mismo -E log p(x|z) (gaussiana en lugar de Bernoulli) y el código lo soporta; se eligió BCE porque los píxeles están en [0, 1] y para mantener una sola receta en todo el trabajo.
- **Capas ocultas tanh de 128 y 32 neuronas y Adam con tasa 0.003.** Mismas activaciones y optimizador que en los ejercicios anteriores, con una capa más ancha porque las imágenes tienen 576 píxeles en lugar de 35.
- **Entrenamiento por lote completo** (los 36 emojis en cada paso, con ruido nuevo en cada época), 3000 épocas.
- **Qué evidencia respalda la elección.** Que la configuración elegida cumple 11 comprobaciones explícitas de calidad generativa (la sección "Reconstrucción y espacio latente" las lista; entre ellas, la ventaja de plausibilidad sobre el autoencoder simple, las combinaciones distintas entre los híbridos y los dos controles negativos) y que un autoencoder idéntico sin el esquema variacional, entrenado con los mismos hiperparámetros, no es muestreable (contraste de la sección "Reconstrucción y espacio latente"). No se hizo un estudio de arquitecturas para el Ej 2 porque la consigna no lo pide; la configuración está congelada en el código (`chosen_vae_config`) y las pruebas la verifican con las semillas 0, 1 y 2.

## Evolución de la pérdida

La tabla muestra la época 1, una de cada 300 y la última, y se escribe completa en `output/ej2/vae_loss.csv` (una fila por época); la figura `output/ej2/vae_loss.png` muestra las tres curvas en escala logarítmica.

```
epoca  reconstruccion  kl  total
    1        400.9978      0.6028    401.6006
  300        102.7271      3.3819    106.1090
  600         94.2728      4.0108     98.2836
  900         91.6688      4.2866     95.9554
 1200         90.7487      4.1675     94.9163
 1500         90.5933      4.0924     94.6857
 1800         90.7652      4.0482     94.8134
 2100         90.3984      4.2204     94.6188
 2400         90.1190      4.1523     94.2714
 2700         89.2864      4.4991     93.7854
 3000         88.9131      4.3949     93.3080
PERDIDA: reconstruccion 401.00 -> 88.91 | KL 0.60 -> 4.39 | total 401.60 -> 93.31
```

Lectura:

- **Reconstrucción.** Cae de 401 a unos 130 en las primeras 30 épocas (576 x ln 2 es aproximadamente 399, que es lo que da la BCE al predecir 0.5 en todos los píxeles y es cercano al valor de la época 1) y después baja lentamente hasta 88.91. Desde la época 600 la mejora es pequeña (de 94.27 a 88.91) y la curva tiene un ruido visible porque cada época usa un eps distinto. No llega a 0 y no puede hacerlo: los objetivos son grises del supersampling (no 0 o 1 puros), y con BCE la pérdida de un objetivo gris tiene un mínimo positivo; además el ruido de muestreo de z impide una reconstrucción exacta.
- **KL.** Empieza en 0.60 y sube durante las primeras épocas hasta un pico de unos 14.7 nats en la época 10, mientras el decoder aprende a usar el código; luego baja hasta un mínimo de unos 1.8 nats hacia la época 64 y vuelve a subir lentamente hasta estabilizarse en torno a 4 nats (4.39 al final). Es el compromiso que impone el costo: el encoder gasta algo más de 4 nats por emoji para describirlo con 2 números, y el término KL le exige que esas descripciones queden cerca del prior N(0, I).
- **Total.** Sigue de cerca a la reconstrucción, porque el KL es unas 20 veces menor (4.39 contra 88.91).

## Reconstrucción y espacio latente

Reconstruir desde la media del encoder (sin ruido) da:

```
RECONSTRUCCION: media 5.19 px, maximo 16 px por emoji; 5/36 emojis sin pixeles incorrectos
```

La figura `output/ej2/vae_reconstruction.png` muestra los 36 originales (filas impares) y sus reconstrucciones con los píxeles incorrectos de cada una (filas pares). Las reconstrucciones se leen como el emoji original; los mayores errores (de 9 a 16 píxeles) aparecen en las bocas `grin` y `flat` de casi todos los estilos de ojos, mientras que la mayoría de las demás combinaciones queda entre 0 y 5 píxeles. La boca `grin` es una mancha negra con borde nítido que la red suaviza; la boca `flat` es una barra gris, y un gris cercano a 0.5 hace que pequeñas diferencias cambien muchos píxeles al binarizar.

**El plano latente decodificado.** La figura `output/ej2/vae_manifold.png` decodifica una grilla de 15 x 15 puntos sobre z en [-2.5, 2.5] x [-2.5, 2.5] (z1 crece hacia la derecha y z2 hacia arriba). La línea medida es:

```
MANIFOLD: 225 celdas (15x15) sobre z en [-2.5, 2.5]^2 | reconocibles 67.1 % (a lo sumo 8 px de un emoji del dataset) | salto medio entre celdas vecinas 7.2 px (maximo 32) | saltos de mas de 24 px: 7.1 %
```

Lo que se ve en la imagen: es un mapa de caras sin celdas rotas. El eje z1 (horizontal) lleva los ojos: a la izquierda ojos de arcos y guiños (`happy`, `wink`, `closed`), en el centro puntos (`dots`) y a la derecha cruces (`x`) y aros (`wide`). El eje z2 (vertical) lleva la boca: arriba `frown`, luego `flat`, `small` y `smile`, y abajo `open` y `grin`. Las transiciones son graduales en la mayor parte de la grilla (las regiones de un mismo estilo ocupan varias celdas vecinas) y los saltos bruscos se concentran en dos franjas: las columnas centrales, donde los ojos pasan a `x` y `wide` con celdas de mezcla en gris, y las filas del tercio inferior, donde la boca pasa de `smile` a `open` y a `grin` con la mancha negra de la boca abierta. El salto máximo entre vecinas es de 32 píxeles y solo el 7.1 % de los pares vecinos salta más de 24 píxeles.

**Contraste con un autoencoder sin esquema variacional.** Para mostrar que el esquema variacional es lo que vuelve muestreable el espacio latente, `ej2` entrena además un autoencoder simple con exactamente la misma arquitectura (latente de 2), optimizador, pérdida, épocas y semilla, pero sin capa estocástica ni término KL. Para ambos modelos se decodifican 1000 puntos z ~ N(0, I) con la misma semilla:

Las definiciones de "cara plausible" y "híbrido plausible" son las de la sección "Generación de un emoji nuevo"; cada modelo se juzga con los pisos de ruido de sus propias reconstrucciones. Las líneas medidas son:

```
VAE PRIOR: de 1000 puntos z ~ N(0, I): 74.4 % decodifican a un emoji reconocible (a lo sumo 8 px de un emoji del dataset), 98.2 % a una cara plausible (marco intacto y cada region en un estilo o entre dos), 17.6 % a un hibrido plausible (alguna region fuera del piso de ruido de todos los estilos) en 31 combinaciones distintas
AE simple PRIOR: de 1000 puntos z ~ N(0, I): 23.1 % decodifican a un emoji reconocible (a lo sumo 8 px de un emoji del dataset), 34.4 % a una cara plausible (marco intacto y cada region en un estilo o entre dos), 16.1 % a un hibrido plausible (alguna region fuera del piso de ruido de todos los estilos) en 13 combinaciones distintas
CONTRASTE: puntos z ~ N(0, I) que decodifican a un emoji reconocible: VAE 74.4 % contra autoencoder sin esquema variacional 23.1 % | a una cara plausible: VAE 98.2 % contra 34.4 % | a un hibrido plausible: VAE 17.6 % contra 16.1 % | salto medio entre celdas vecinas: VAE 7.2 px contra 27.5 px | reconstruccion media: VAE 5.19 px contra 4.56 px por emoji
```

Interpretación:

- El autoencoder simple reconstruye igual de bien o mejor (4.56 px contra 5.19 px por emoji): tener un buen autoencoder no alcanza para poder generar.
- Pero sus códigos no están distribuidos como N(0, I): al sortear del prior solo el 23.1 % de los puntos cae donde su decoder produce un emoji reconocible, contra el 74.4 % del VAE, y solo el 34.4 % decodifica a una cara plausible, contra el 98.2 % del VAE. El término KL del VAE es lo que acerca las posteriores al prior y así hace que muestrear N(0, I) tenga sentido.
- La proporción de sorteos aceptados como híbridos plausibles es prácticamente la misma en los dos modelos (17.6 % del VAE contra 16.1 % del autoencoder simple), de modo que por sí sola no separa al VAE del autoencoder simple; con otra semilla de entrenamiento del VAE puede incluso ser menor (con la semilla 2 es 8.1 %, por debajo del 16.1 % del autoencoder simple). Es la misma debilidad del criterio anterior, cuya proporción de nuevos y plausibles también era casi igual en los dos modelos.
- Lo que separa a los dos modelos es la plausibilidad: los puntos z ~ N(0, I) caen en su mayoría fuera de la región de códigos que aprendió el autoencoder simple y decodifican a marcos rotos (34.4 % de caras plausibles contra 98.2 %). Por eso los híbridos del autoencoder simple no son evidencia de un buen generador: son una fracción pequeña de un conjunto de sorteos en el que dos tercios ni siquiera son una cara.
- La grilla del autoencoder simple (`output/ej2/plain_manifold.png`, hecha sobre la caja de sus propios códigos con 10 % de margen) tiene una región de caras razonables arriba y a la izquierda, pero hacia la esquina inferior derecha aparece una zona de imágenes con el fondo negro (invertidas) y celdas casi ilegibles, con bordes duros entre regiones; el salto medio entre celdas vecinas es 27.5 px contra 7.2 px del VAE, que no tiene ninguna celda negra.

**Comprobaciones de aceptación.** El resultado se verifica con 11 comprobaciones que el código evalúa y la corrida imprime:

```
ACEPTACION OK: reconstruccion media (px por emoji) = 5.194 (umbral <= 12.000)
ACEPTACION OK: KL final (nats) = 4.395 (umbral >= 1.000)
ACEPTACION OK: celdas reconocibles de la grilla = 0.671 (umbral >= 0.500)
ACEPTACION OK: salto medio entre celdas vecinas (px) = 7.181 (umbral <= 12.000)
ACEPTACION OK: sorteos del prior reconocibles = 0.744 (umbral >= 0.600)
ACEPTACION OK: sorteos del prior hibridos y plausibles = 0.176 (umbral >= 0.050)
ACEPTACION OK: ventaja sobre el autoencoder simple (sorteos reconocibles) = 0.513 (umbral >= 0.300)
ACEPTACION OK: ventaja sobre el autoencoder simple (sorteos plausibles) = 0.638 (umbral >= 0.300)
ACEPTACION OK: combinaciones distintas entre los hibridos del prior = 31.000 (umbral >= 8.000)
ACEPTACION OK: control negativo: reconstrucciones del entrenamiento aceptadas como nuevas = 0.000 (umbral <= 0.000)
ACEPTACION OK: control negativo: muestras de la posterior aceptadas como nuevas = 0.044 (umbral <= 0.100)
```

Los umbrales son convenciones de este trabajo, fijadas con margen respecto de lo que logra la configuración y sin ajustarlos para aprobar; las pruebas (`tests/vae_acceptance.rs`) repiten las comprobaciones con las semillas de entrenamiento 0, 1 y 2. Las dos comprobaciones de control negativo y la de plausibilidad son las que se agregaron para cerrar el hallazgo de la revisión sobre el emoji nuevo, y la comprobación de la tasa de sorteos híbridos reemplazó a la anterior, que el autoencoder simple también cumplía.

## Generación de un emoji nuevo

**Respuesta a 2c.** La consigna pide generar un emoji nuevo muestreando el espacio latente y juzgar si pertenece al dataset. Lo difícil es definir "nuevo" de una manera que no se pueda cumplir por accidente. Esta sección cuenta primero qué falló en el primer criterio, después define el criterio actual, lo verifica con un control negativo y termina con lo que los autores ven en la figura.

**Por qué se descartó el primer criterio.** La primera versión declaraba nuevo a un emoji decodificado a entre 7 y 16 píxeles de distancia global (Hamming tras binarizar) del emoji más cercano del dataset. Esa banda es del mismo tamaño que el ruido de reconstrucción del propio decodificador: al reconstruir un emoji real del entrenamiento, el VAE falla en promedio 5.19 px y hasta 16 px (línea `RECONSTRUCCION:` de la sección anterior). Por eso reconstrucciones de emojis reales caían dentro de la banda y pasaban por nuevas; el control negativo de más abajo cuenta cuántas (línea `CONTROL NEGATIVO:`). La muestra que se presentaba entonces era una copia degradada de `happy-flat`: la revisión de la figura mostró que la barra de la boca plana quedaba partida en fragmentos porque el decodificador la dibuja más tenue, y esos 8 píxeles de diferencia eran borrosidad y no un diseño distinto. La afirmación de que esa muestra no copiaba ningún emoji del dataset era incorrecta y se retiró.

**Regiones y pisos de ruido.** Como el dataset es el producto completo de ojos por boca, la novedad se mide por región. Las regiones se derivan de los datos: la región de ojos es el conjunto de píxeles que cambian con el estilo de ojos, la de boca los que cambian con el estilo de boca, y el marco los píxeles iguales en los 36 emojis. El piso de ruido de una región es la mayor cantidad de píxeles de esa región que el VAE falla al reconstruir, desde la media de la posterior, los 36 emojis del entrenamiento. La línea medida es:

```
REGIONES: ojos 46 px, boca 62 px, marco 468 px | piso de ruido de reconstruccion (maximo sobre los 36 emojis): ojos 3 px, boca 14 px, marco 2 px | separacion minima entre estilos: ojos 7 px, boca 8 px
```

**Definiciones.** Con los números de esa línea:

- El estilo más cercano de una región es el de menor distancia (en los píxeles de esa región) entre los 6 estilos de la región; el prototipo de cada estilo son los píxeles de esa región en una fila del dataset.
- Una región es *fuera de estilo* si está más lejos que el piso de la región de cada uno de sus 6 estilos: más de 3 px de todos los estilos de ojos, más de 14 px de todos los estilos de boca.
- La cara es *plausible* si el marco está dentro de su piso (a lo sumo 2 px de diferencia con el marco del dataset) y cada región fuera de estilo queda entre sus dos estilos más cercanos: la distancia al más cercano más la distancia al segundo más cercano es a lo sumo 1.5 veces la distancia entre esos dos estilos.
- Una muestra se acepta como nueva (un *híbrido*) cuando al menos una región es fuera de estilo y la cara es plausible.
- El *margen* es cuántos píxeles por encima del piso está la región más fuera de estilo.

**Regla de la galería.** El flujo de N(0, I) tiene semilla fija (1514078721) y se examinan 1000 sorteos. Entre los candidatos (los sorteos aceptados como híbridos) se conserva uno por combinación de (emoji más cercano, estilo de ojos, estilo de boca), el de mayor margen, y se muestran como máximo 8 ordenados por margen; ante un empate gana el sorteo anterior. El emoji principal es entonces el de mayor margen entre combinaciones distintas, elegido por esta regla y no a ojo. La búsqueda dio:

```
EMOJI NUEVO: z=(1.3701, -1.2836) sorteo 9 de 1000 | emoji mas cercano: x-grin a 25 px (radio de cercania 8 px) | ojos: x a 14 px del estilo mas cercano (piso 3 px), boca: grin a 11 px (piso 14 px) | criterio: alguna region a mas de su piso de ruido de todos los estilos de esa region, con el marco intacto y cada region en un estilo o entre dos
```

```
GALERIA: 176 sorteos candidatos de 1000, 31 combinaciones distintas (emoji mas cercano, estilo de ojos, estilo de boca); se muestran 8 ordenadas por margen sobre el piso de ruido
  nuevo 1: sorteo 9 | emoji mas cercano x-grin a 25 px | ojos x a 14 px | boca grin a 11 px | margen 11 px
  nuevo 2: sorteo 357 | emoji mas cercano wide-frown a 18 px | ojos wide a 14 px | boca frown a 4 px | margen 11 px
  nuevo 3: sorteo 852 | emoji mas cercano wide-grin a 27 px | ojos wide a 13 px | boca grin a 14 px | margen 10 px
  nuevo 4: sorteo 903 | emoji mas cercano x-smile a 12 px | ojos x a 12 px | boca smile a 0 px | margen 9 px
  nuevo 5: sorteo 842 | emoji mas cercano wide-flat a 19 px | ojos wide a 11 px | boca flat a 8 px | margen 8 px
  nuevo 6: sorteo 172 | emoji mas cercano wide-smile a 10 px | ojos wide a 10 px | boca smile a 0 px | margen 7 px
  nuevo 7: sorteo 619 | emoji mas cercano wide-small a 10 px | ojos wide a 10 px | boca small a 0 px | margen 7 px
  nuevo 8: sorteo 13 | emoji mas cercano dots-grin a 20 px | ojos dots a 2 px | boca grin a 18 px | margen 4 px
```

La imagen decodificada del emoji principal, binarizada en 0.5 (izquierda), junto a su emoji más cercano del dataset, `x-grin` (derecha). La diferencia de 25 px se reparte entre los ojos (14 px, las cruces de `x` quedan reemplazadas por manchas dispersas) y la boca (11 px, el borde superior de la boca abierta queda fragmentado):

```
........................  ........................
........########........  ........########........
......###......###......  ......###......###......
.....##..........##.....  .....##..........##.....
....##............##....  ....##............##....
...##..............##...  ...##..............##...
..##................##..  ..##................##..
..#.....#......#.....#..  ..#..................#..
.##...#..#....#..#...##.  .##...##.#....#.##...##.
.#.....##.#..#.##.....#.  .#.....##......##.....#.
.#....#.#......#.#....#.  .#.....###....###.....#.
.#....##.#....#.##....#.  .#....#..#....#..#....#.
.#....................#.  .#....................#.
.#.....#.....#..#.....#.  .#....############....#.
.#....############....#.  .#....############....#.
.##....##########....##.  .##....##########....##.
..#....##########....#..  ..#....##########....#..
..##....########....##..  ..##....########....##..
...##....######....##...  ...##....######....##...
....##............##....  ....##.....##.....##....
.....##..........##.....  .....##..........##.....
......###......###......  ......###......###......
........########........  ........########........
........................  ........................
```

**Control negativo.** Si el criterio aceptara cualquier reconstrucción del decodificador, no probaría nada. Por eso se lo aplica a imágenes que seguro no son nuevas: las reconstrucciones de los emojis del entrenamiento.

```
CONTROL NEGATIVO: reconstrucciones de los 36 emojis del entrenamiento aceptadas como nuevas: 0/36 (criterio anterior de banda de pixeles: 12/36) | muestras de la posterior q(z|x), 30 por emoji: 4.4 % aceptadas como nuevas
```

Se lee así: el criterio actual no acepta ninguna de las 36 reconstrucciones (media de la posterior) de emojis del entrenamiento, mientras que la banda de píxeles que reemplaza aceptaba 12 de esas 36. Ese cero es cierto por construcción: los pisos se miden como el máximo de error sobre esas mismas 36 reconstrucciones, así que ninguna puede superarlos. La evidencia independiente es la segunda mitad de la línea: de las reconstrucciones estocásticas (30 muestras de la posterior q(z|x) por cada emoji, con una semilla propia del control) el 4.4 % se acepta como nueva. Es una proporción pequeña pero no nula, y es unas cuatro veces menor que el 17.6 % de los sorteos del prior que se aceptan como híbridos, de modo que el criterio no acepta cualquier salida del decodificador; tampoco es perfecto, porque una pequeña parte del ruido de reconstrucción lo supera.

La figura `output/ej2/vae_new_emoji.png` muestra, arriba, los 8 emojis de la galería en orden de margen y, abajo, el emoji más cercano de cada uno con su distancia en píxeles. Los ocho son caras de la misma familia que el dataset (mismo contorno, dos ojos y una boca). En siete de ellos los ojos son anillos pequeños de textura moteada; en el octavo son los puntos negros de `dots`.

**Juicio visual de los autores.** Lo que vemos al abrir la figura, entrada por entrada:

- **Entradas 1 y 4** (cercanos `x-grin` y `x-smile`): son las dos claramente distintas de su emoji de abajo, porque las cruces del dataset aparecen reemplazadas por anillos moteados. Aun así, esos anillos moteados se parecen a los ojos `wide` del dataset, de modo que también pueden leerse como una versión ruidosa de ese estilo; lo que sí afirmamos es que, medidos en píxeles, están a más del piso de todos los estilos de ojos.
- **Entradas 2, 3, 5, 6 y 7** (cercanos `wide-frown`, `wide-grin`, `wide-flat`, `wide-smile` y `wide-small`): solo sutilmente distintas. A este tamaño de figura se confunden a primera vista con el emoji de abajo; la diferencia (de 10 a 14 px en los ojos) está en la textura del anillo, y las bocas coinciden o difieren poco. Se leen como una variante del estilo `wide`, no como un estilo nuevo evidente.
- **Entrada 8** (cercano `dots-grin`): los ojos son idénticos a los del dataset, y la diferencia es una boca cuya fila superior es gris y más angosta. Es la entrada más débil (margen de 4 px, la única cuya novedad viene solo de la boca) y un lector podría considerarla una copia degradada de `dots-grin`.

En resumen, solo 2 de las 8 entradas se distinguen a simple vista de su emoji más cercano, 5 son sutiles y 1 es dudosa. Casi toda la novedad está en los ojos: en 7 de las 8 filas de la galería los ojos están a 10 px o más del estilo más cercano (el piso es 3 px), y solo en la fila 8 viene de la boca. La región de la boca casi nunca califica porque su piso de ruido (14 px) es mayor que la separación mínima entre estilos de boca vecinos (8 px, ambos números de la línea `REGIONES:`): una boca que mezcla dos estilos vecinos queda a poca distancia de alguno de ellos y es difícil que supere el piso, de modo que casi nunca se detecta como fuera de estilo.

Lo que el criterio no puede mostrar: es una medida relativa al dataset y al ruido de este decodificador, no una medida de qué tan distinto lo percibe una persona; que una región supere el piso de ruido significa que no es el error habitual del VAE al reconstruir, no que sea un estilo de ojos o de boca inventado con valor propio. Este es el juicio de los autores y debe leerse junto con la imagen.

## Alcance y limitaciones

- El dataset es una construcción de 36 imágenes con dos factores de variación (ojos y boca) y es el producto completo de 6 por 6: como ninguna combinación de estilos conocidos está ausente, los emojis nuevos no son combinaciones de estilos conocidos sino regiones que son mezclas o variantes de los estilos de una región. Haber tratado como nuevas combinaciones que ya estaban en el dataset fue la causa del hallazgo de la revisión sobre el emoji nuevo.
- El piso de ruido de la región de boca (14 px) es tan grande como la separación entre estilos vecinos de boca (8 px), así que un híbrido de boca casi nunca es detectable y la novedad está sobre todo en los ojos (7 de las 8 entradas de la galería); los números salen de la línea `REGIONES:`.
- Los pisos de ruido se miden sobre los mismos 36 emojis con los que se entrena el VAE. Por eso el control con la reconstrucción media da cero por construcción, y la evidencia independiente es el control con reconstrucciones estocásticas, que acepta una proporción pequeña pero no nula (4.4 %).
- El criterio de novedad sigue siendo una aproximación definida respecto del dataset y del ruido de este decodificador: el juicio final de si un emoji es nuevo es visual y subjetivo, y de las 8 entradas de la galería solo 2 se distinguen a simple vista de su emoji más cercano.
- La proporción de sorteos aceptados como híbridos no separa al VAE del autoencoder simple (17.6 % contra 16.1 %, y con la semilla 2 es 8.1 %, menor que la del autoencoder simple); lo que los separa es la plausibilidad (98.2 % contra 34.4 %).
- El tamaño latente está fijado en 2 y no se estudió otra arquitectura, porque la consigna no lo pide; los resultados corresponden a una red elegida con la semilla de entrenamiento 0, y las pruebas repiten las comprobaciones de aceptación con las semillas 0, 1 y 2.
- Los criterios de reconocibilidad (radio de cercanía derivado de la separación), de plausibilidad y de novedad (pisos de ruido por región) son aproximaciones definidas respecto del propio dataset, con distancia de Hamming tras binarizar; los umbrales de aceptación son convenciones de este trabajo.
- En el plano latente persisten saltos bruscos entre estilos (máximo de 32 px entre celdas vecinas, 7.1 % de pares con saltos de más de 24 px), y las decodificaciones son algo más borrosas que los originales: el costo de reconstrucción no baja de unos 89 nats por imagen.
- No se usó recocido ni ponderación del término KL distinta de 1.
- El contraste con el autoencoder simple usa una sola arquitectura y una sola semilla; muestra la diferencia para esa configuración y no la generaliza a cualquier autoencoder.
