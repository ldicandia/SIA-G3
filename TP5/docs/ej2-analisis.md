# Ejercicio 2: autoencoder variacional sobre emojis

Análisis del Ejercicio 2 del TP5 (Sistemas de Inteligencia Artificial, ITBA). La consigna pide (2a) elegir o construir un dataset de emojis, (2b) implementar un autoencoder variacional (VAE) para resolver el problema de la representación en el espacio latente y justificar la configuración elegida, y (2c) generar un emoji nuevo muestreando el espacio latente y juzgar si pertenece al dataset. Todos los números de este documento salen de `cargo run --release -- ej2`, que entrena el VAE y el autoencoder de contraste una sola vez y escribe las figuras en `output/ej2/`; una prueba (`tests/vae_doc_consistency.rs`) recalcula cada línea citada y falla si el documento difiere del código.

Para regenerar todo: `cargo run --release -- ej2` (el ejercicio completo, con veredicto de aceptación) y, por separado, `cargo run --release -- emojis` (dataset), `cargo run --release -- vae` (entrenamiento y una muestra del prior), `cargo run --release -- vae-sample` (búsqueda del emoji nuevo), `cargo run --release -- vae-manifold` (grilla del plano latente) y `cargo run --release -- vae-baseline` (contraste con el autoencoder simple y aceptación). Todas las corridas son deterministas por semilla y los comandos individuales imprimen los mismos números que `ej2` para la misma configuración.

## Dataset de emojis

**Respuesta a 2a.** La consigna permite elegir o construir el dataset; se construyó por código. Todas las imágenes comparten el mismo contorno de cara y combinan 6 estilos de ojos (`dots`, `happy`, `wide`, `wink`, `closed`, `x`) con 6 estilos de boca (`smile`, `grin`, `flat`, `frown`, `open`, `small`): 6 x 6 = 36 combinaciones. Cada imagen se dibuja con primitivas geométricas (anillos, cápsulas, discos) en una grilla de 24 x 24 píxeles con supersampling de 4 x 4, de modo que los valores son grises en [0, 1] (1 = tinta) y se aplana a un vector de 576 valores. No requiere descargar nada ni agrega ninguna dependencia, y es determinista. La corrida imprime:

```
emojis: 36 imagenes de 24x24 (576 valores en [0,1], 1 = tinta)
separacion: minima 7 px, mediana 7.5 px, radio de cercania 8 px
```

La distancia entre dos emojis es la cantidad de píxeles en que difieren tras binarizar en 0.5 (distancia de Hamming, la misma regla que usa el reporte del Ej 1). La línea `separacion:` resume la distancia de cada emoji a su vecino más cercano: el par más parecido difiere en 7 píxeles, la mediana es 7.5 y el radio de cercanía se define como la mediana redondeada hacia arriba (8 px). Estos números definen más adelante cuándo una imagen decodificada se considera nueva y reconocible.

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
- **Qué evidencia respalda la elección.** Que la configuración elegida cumple siete comprobaciones explícitas de calidad generativa (sección siguiente y la última de esta sección) y que un autoencoder idéntico sin el esquema variacional, entrenado con los mismos hiperparámetros, no es muestreable (contraste de la sección "Reconstrucción y espacio latente"). No se hizo un estudio de arquitecturas para el Ej 2 porque la consigna no lo pide; la configuración está congelada en el código (`chosen_vae_config`) y las pruebas la verifican con las semillas 0, 1 y 2.

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

```
VAE PRIOR: de 1000 puntos z ~ N(0, I): 74.4 % decodifican a un emoji reconocible (a lo sumo 8 px de un emoji del dataset), 36.1 % a un emoji nuevo (al menos 7 px de todos) y 31.9 % a un emoji nuevo y plausible (a lo sumo 16 px)
AE simple PRIOR: de 1000 puntos z ~ N(0, I): 23.1 % decodifican a un emoji reconocible (a lo sumo 8 px de un emoji del dataset), 82.2 % a un emoji nuevo (al menos 7 px de todos) y 30.7 % a un emoji nuevo y plausible (a lo sumo 16 px)
CONTRASTE: puntos z ~ N(0, I) que decodifican a un emoji reconocible: VAE 74.4 % contra autoencoder sin esquema variacional 23.1 % | salto medio entre celdas vecinas: VAE 7.2 px contra 27.5 px | reconstruccion media: VAE 5.19 px contra 4.56 px por emoji
```

Interpretación:

- El autoencoder simple reconstruye igual de bien o mejor (4.56 px contra 5.19 px por emoji): tener un buen autoencoder no alcanza para poder generar.
- Pero sus códigos no están distribuidos como N(0, I): al sortear del prior solo el 23.1 % de los puntos cae donde su decoder produce un emoji reconocible, contra el 74.4 % del VAE. El término KL del VAE es lo que acerca las posteriores al prior y así hace que muestrear N(0, I) tenga sentido.
- El 82.2 % de sorteos "nuevos" del autoencoder simple no es una virtud: casi todos son imágenes rotas, lejos de cualquier emoji. La comparación informativa es la de sorteos reconocibles; la de nuevos y plausibles es parecida en los dos modelos (30.7 % y 31.9 %) y por sí sola no los distingue.
- La grilla del autoencoder simple (`output/ej2/plain_manifold.png`, hecha sobre la caja de sus propios códigos con 10 % de margen) tiene una región de caras razonables arriba y a la izquierda, pero hacia la esquina inferior derecha aparece una zona de imágenes con el fondo negro (invertidas) y celdas casi ilegibles, con bordes duros entre regiones; el salto medio entre celdas vecinas es 27.5 px contra 7.2 px del VAE, que no tiene ninguna celda negra.

**Comprobaciones de aceptación.** El resultado se verifica con siete comprobaciones que el código evalúa y la corrida imprime:

```
ACEPTACION OK: reconstruccion media (px por emoji) = 5.194 (umbral <= 12.000)
ACEPTACION OK: KL final (nats) = 4.395 (umbral >= 1.000)
ACEPTACION OK: celdas reconocibles de la grilla = 0.671 (umbral >= 0.500)
ACEPTACION OK: salto medio entre celdas vecinas (px) = 7.181 (umbral <= 12.000)
ACEPTACION OK: sorteos del prior reconocibles = 0.744 (umbral >= 0.600)
ACEPTACION OK: sorteos del prior nuevos y plausibles = 0.319 (umbral >= 0.100)
ACEPTACION OK: ventaja sobre el autoencoder simple (sorteos reconocibles) = 0.513 (umbral >= 0.300)
```

Los umbrales son convenciones de este trabajo, fijadas con margen respecto de lo que logra la configuración y sin ajustarlos para aprobar; las pruebas (`tests/vae_acceptance.rs`) las repiten con las semillas de entrenamiento 0, 1 y 2.

## Generación de un emoji nuevo

**Respuesta a 2c.** Con los números de la línea `separacion:` (mínima 7 px, mediana 7.5 px, radio de cercanía 8 px) se definen tres criterios explícitos sobre la imagen decodificada, siempre con la distancia en píxeles al emoji más cercano del dataset:

- **Nuevo:** está a al menos 7 px (la separación mínima del dataset) de los 36 emojis, es decir, no es una copia de ninguno.
- **Reconocible:** está a lo sumo a 8 px (el radio de cercanía) de algún emoji del dataset.
- **Plausible:** está a lo sumo a 16 px (el doble del radio) del emoji más cercano, o sea que no es una imagen cualquiera.

La muestra presentada es el primer sorteo aceptado (nuevo y plausible) de un flujo de N(0, I) con semilla fija (1514078721), sin ranking ni selección a mano. La búsqueda y las estadísticas del prior dan:

```
EMOJI NUEVO: z=(-1.8367, 0.7242) tras 3 sorteos | emoji mas cercano: happy-flat a 8 px | criterio: a al menos 7 px de todo el dataset y a lo sumo 16 px del mas cercano
VAE PRIOR: de 1000 puntos z ~ N(0, I): 74.4 % decodifican a un emoji reconocible (a lo sumo 8 px de un emoji del dataset), 36.1 % a un emoji nuevo (al menos 7 px de todos) y 31.9 % a un emoji nuevo y plausible (a lo sumo 16 px)
```

Es decir que casi un tercio de los puntos del prior (31.9 %) decodifica a un emoji nuevo y plausible, y el primero aparece al tercer sorteo. La imagen decodificada, binarizada en 0.5 (izquierda), junto a su emoji más cercano del dataset, `happy-flat` (derecha):

```
........................  ........................
........########........  ........########........
......###......###......  ......###......###......
.....##..........##.....  .....##..........##.....
....##............##....  ....##............##....
...##..............##...  ...##..............##...
..##................##..  ..##................##..
..#..................#..  ..#..................#..
.##....##......##....##.  .##....##......##....##.
.#....##.##..##.##....#.  .#....##.##..##.##....#.
.#....................#.  .#....................#.
.#....................#.  .#....................#.
.#....................#.  .#....................#.
.#....................#.  .#....................#.
.#....................#.  .#....................#.
.##..................##.  .##..................##.
..#.....#....#.#.....#..  ..#.....########.....#..
..##....#..##.##....##..  ..##....########....##..
...##..............##...  ...##..............##...
....##............##....  ....##............##....
.....##..........##.....  .....##..........##.....
......###......###......  ......###......###......
........########........  ........########........
........................  ........................
```

La figura `output/ej2/vae_new_emoji.png` muestra, arriba, hasta ocho emojis nuevos tomados en orden del mismo flujo y, abajo, el emoji más cercano de cada uno, con la distancia en píxeles.

**Juicio visual de los autores.** Mirando la imagen, los ocho emojis nuevos se leen como caras de la misma familia: mismo contorno, dos ojos y una boca. El principal combina los ojos de arcos de `happy` con la boca plana de `flat`, y se diferencia de `happy-flat` en la boca: en la versión binarizada la barra horizontal queda partida en fragmentos porque en la imagen decodificada es más tenue (gris por debajo de 0.5 en parte de su largo), y esos píxeles son los 8 de diferencia; los ojos y el contorno coinciden. Entre los demás, el 3 es una mezcla de ojos `wink` con boca `flat` y el 8 es una cara `x` con boca `frown`; varios (2, 4, 5 y 6) están cerca de `dots-grin` (de 14 a 16 px) y se ven casi iguales a ese emoji, con ojos y bordes algo más difusos. Por lo tanto la novedad es real según el criterio numérico pero la diferencia visual con el dataset es sutil: lo nuevo son combinaciones o mezclas de estilos y grises de transición, no un dibujo de otra clase. Es el juicio de los autores y debe leerse junto con la imagen.

## Alcance y limitaciones

- El dataset es una construcción de 36 imágenes con dos factores de variación (ojos y boca): los emojis nuevos son combinaciones o mezclas de esos estilos, no tipos de dibujo nuevos.
- El tamaño latente está fijado en 2 y no se estudió otra arquitectura, porque la consigna no lo pide; los resultados corresponden a una red elegida con la semilla de entrenamiento 0, y las pruebas repiten las comprobaciones de aceptación con las semillas 0, 1 y 2.
- Los criterios de novedad, plausibilidad y reconocibilidad son aproximaciones definidas respecto del propio dataset (distancia de Hamming tras binarizar y umbrales derivados de la separación); el juicio final de pertenencia es visual, y los umbrales de aceptación son convenciones de este trabajo.
- En el plano latente persisten saltos bruscos entre estilos (máximo de 32 px entre celdas vecinas, 7.1 % de pares con saltos de más de 24 px), y las decodificaciones son algo más borrosas que los originales: el costo de reconstrucción no baja de unos 89 nats por imagen.
- No se usó recocido ni ponderación del término KL distinta de 1.
- El contraste con el autoencoder simple usa una sola arquitectura y una sola semilla; muestra la diferencia para esa configuración y no la generaliza a cualquier autoencoder.
