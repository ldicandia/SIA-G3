# Ejercicio 2: autoencoder variacional sobre bloques de mena de Minecraft

Análisis del Ejercicio 2 del TP5 (Sistemas de Inteligencia Artificial, ITBA). La consigna pide (2a) elegir o construir un dataset nuevo, (2b) implementar un autoencoder variacional (VAE) para resolver el problema de la representación en el espacio latente y justificar la configuración elegida, y (2c) generar una muestra nueva desde el espacio latente y juzgar si pertenece al dataset. Todos los números de este documento salen de `cargo run --release -- ej2`, que entrena el VAE y el autoencoder de contraste una sola vez y escribe las figuras en `output/ej2/`; una prueba (`tests/vae_doc_consistency.rs`) recalcula cada línea citada y falla si el documento difiere del código.

Para regenerar todo: `bash scripts/descargar_menas.sh` (texturas, una sola vez) y `cargo run --release -- ej2` (el ejercicio completo, con veredicto de aceptación); por separado, `cargo run --release -- ores` (dataset), `cargo run --release -- vae` (entrenamiento y una muestra del prior), `cargo run --release -- vae-sample` (búsqueda del bloque nuevo), `cargo run --release -- vae-manifold` (grilla del plano latente) y `cargo run --release -- vae-baseline` (contraste con el autoencoder simple y aceptación). Todas las corridas son deterministas por semilla y los comandos individuales imprimen los mismos números que `ej2` para la misma configuración.

## Dataset de bloques

**Respuesta a 2a.** El dataset son las texturas reales de los bloques de mena de Minecraft 1.21.4: los 8 minerales del juego (carbón, hierro, cobre, oro, redstone, esmeralda, lapislázuli y diamante) en piedra y en pizarra profunda, más el oro y el cuarzo del Nether, 18 bloques en total. Cada textura es de 16 x 16 píxeles RGB y se aplana a un vector de 768 valores en [0, 1] (los 3 canales de cada píxel seguidos). Las texturas son de Mojang y no se versionan: `scripts/descargar_menas.sh` baja el cliente oficial del juego desde los servidores de Mojang, verifica su SHA-1, extrae las 18 texturas y las 3 rocas base (piedra, pizarra profunda y netherrack) y verifica el SHA-256 de cada una, de modo que el dataset es el mismo en cualquier máquina. La corrida imprime:

```
bloques de mena: 18 texturas de Minecraft 1.21.4 de 16x16 RGB (768 valores en [0,1]) | 9 minerales, 3 rocas
separacion (error medio por canal al bloque mas cercano): minima 20.9 niveles, mediana 25.8 niveles
```

La distancia entre dos imágenes es el error medio por canal, en niveles de 0 a 255. La línea `separacion:` resume la distancia de cada bloque a su vecino más cercano: el par más parecido difiere en 20.9 niveles por canal y la mediana es 25.8. El vecino más cercano de cada bloque está siempre en la misma roca: el fondo ocupa la mayor parte de la textura y pesa más en la distancia que el mineral, y los vecinos más frecuentes son los minerales de poco color (hierro, cobre y carbón).

Cada bloque tiene dos atributos, su mineral y su roca, y el dataset no es el producto completo de los dos: El dataset tiene 18 de las 27 combinaciones posibles de 9 minerales por 3 rocas: faltan los 7 minerales que no aparecen en el Nether y el cuarzo en piedra y en pizarra. Esto define qué puede ser un bloque nuevo: un mineral en una roca donde el dataset no lo tiene, o un bloque con más de un mineral en la misma roca, que no existe en el dataset ni en el juego. La prueba `combinations_present_are_18_of_27` (`tests/vae_judge.rs`) recalcula esos números desde el dataset.

La figura `output/ej2/ores.png` muestra los 18 bloques con su nombre `{mineral}-{roca}`, en el orden del dataset: los 8 de piedra, los 8 de pizarra y los 2 del Nether. Se ve que el mineral se reconoce sobre todo por el color de las vetas y la roca por el fondo: gris claro para la piedra, gris oscuro para la pizarra y rojo oscuro para el netherrack.

## Arquitectura y esquema variacional

**Respuesta a 2b.** La configuración congelada es:

```
configuracion: vae-h128x32-l2-s0 | arquitectura: encoder 768-128-32-4 | decoder 2-32-128-768 | optimizador: Adam(lr=0.003) | reconstruccion: BCE + KL | epocas: 3000 | semilla de entrenamiento: 0
```

**El problema que resuelve el esquema variacional.** En la teoría de la materia (autoencoder generativo) se entrena un autoencoder, se deja de lado el encoder y se eligen valores de las variables latentes z para que el decoder genere muestras nuevas. El problema es lo que la teoría llama la conquista del espacio latente: un autoencoder común deja el espacio latente sin estructura, con regiones de códigos separadas por zonas donde el decoder produce cualquier cosa, y nada garantiza que un punto elegido produzca una muestra válida. La solución es darle al espacio latente una estructura estadística: en lugar de mirar la salida para un único punto z, mirar las salidas para un conjunto de puntos.

**Codificador estocástico.** Se asume un prior p(z) = N(0, I) y se aproxima la posterior con q(z|x) = N(mu(x), Sigma(x)) con Sigma diagonal. El encoder tiene por lo tanto 2 x 2 = 4 salidas: la media y el logaritmo de la varianza de cada una de las 2 dimensiones latentes (de ahí el `4` de `encoder 768-128-32-4`). Se predice el logaritmo de la varianza y se la obtiene con exp(.), parametrización de la teoría que mantiene la varianza positiva y evita problemas numéricos; la capa de salida del encoder es la identidad para que la media y el logaritmo de la varianza no estén acotados.

**Truco de reparametrización.** Muestrear z directamente de N(mu, Sigma) corta el camino del gradiente. Se sortea eps de N(0, I) y se calcula z = mu + eps * sigma con sigma = exp(0.5 * logvar): la aleatoriedad queda fuera de la red, la capa es diferenciable y el error de reconstrucción fluye hacia el encoder. En cada época se sortea un eps nuevo para los 18 bloques, de un flujo de números aleatorios independiente del de la inicialización de los pesos.

**Costo.** El costo es el negativo de la cota inferior variacional: el error de reconstrucción más la divergencia KL(q(z|x) || p(z)). Con las dos gaussianas la KL tiene forma cerrada, 0.5 * suma(exp(logvar) + mu^2 - 1 - logvar) sobre las dimensiones latentes, y se usa con peso 1. Las dos magnitudes se miden por imagen (suma sobre los 768 valores o las dimensiones, media sobre los 18 bloques), en nats, de modo que son comparables y su suma es el costo que se minimiza.

**Por qué esta configuración.**

- **Latente de 2 dimensiones.** Hace que el espacio latente sea un plano que se puede decodificar en una grilla y mirar, como el autoencoder generativo de 2 variables de la teoría, y permite ver directamente qué estructura le dio el término KL.
- **Reconstrucción BCE con salida sigmoide.** Es la receta que convergió en los Ej 1a y 1b. Cada canal de color está en [0, 1] y la BCE es el -E log p(x|z) de un modelo Bernoulli por canal; el error cuadrático de la teoría es la otra instancia del mismo término (gaussiana) y el código lo soporta, pero se mantuvo una sola receta en todo el trabajo.
- **Capas ocultas tanh de 128 y 32 neuronas y Adam con tasa 0.003.** Mismas activaciones y optimizador que en los ejercicios anteriores, con una capa más ancha porque cada imagen tiene 768 valores en lugar de 35.
- **Entrenamiento por lote completo** (los 18 bloques en cada paso, con ruido nuevo en cada época), 3000 épocas.
- **Qué evidencia respalda la elección.** Que la configuración elegida cumple 11 comprobaciones explícitas de calidad generativa (la sección "Reconstrucción y espacio latente" las lista) y que un autoencoder idéntico sin el esquema variacional, entrenado con los mismos hiperparámetros, es bastante menos muestreable (contraste de la misma sección). No se hizo un estudio de arquitecturas para el Ej 2 porque la consigna no lo pide; la configuración está congelada en el código (`chosen_vae_config`) y las pruebas la verifican con las semillas 0, 1 y 2.

## Evolución de la pérdida

La tabla muestra la época 1, una de cada 300 y la última, y se escribe completa en `output/ej2/vae_loss.csv` (una fila por época); la figura `output/ej2/vae_loss.png` muestra las tres curvas en escala logarítmica.

```
epoca  reconstruccion  kl  total
    1        533.4155      0.1219    533.5375
  300        487.0128      3.7322    490.7450
  600        474.1135      4.8388    478.9523
  900        469.2610      5.0321    474.2930
 1200        468.0553      4.9253    472.9806
 1500        468.2472      5.0724    473.3195
 1800        467.7353      4.9070    472.6423
 2100        467.3530      4.7894    472.1425
 2400        467.5497      5.1480    472.6978
 2700        467.4384      5.0175    472.4559
 3000        467.1910      4.4434    471.6344
PERDIDA: reconstruccion 533.42 -> 467.19 | KL 0.12 -> 4.44 | total 533.54 -> 471.63
```

Lectura:

- **Reconstrucción.** Empieza en 533.42, que es lo que da la BCE al predecir 0.5 en todos los valores (768 x ln 2, aproximadamente 532.3), y baja hasta 467.19. La caída parece chica, pero hay que compararla con el piso: con objetivos que no son 0 o 1 puros (los colores de una textura) la BCE no puede bajar de la entropía de los propios objetivos, que para este dataset es de unos 466.3 nats por bloque. La reconstrucción final queda a menos de 1 nat de ese piso, es decir, prácticamente todo lo que se podía ganar se ganó; por eso la tabla casi no cambia después de la época 900.
- **KL.** Empieza en 0.12, sube mientras el decoder aprende a usar el código, llega a un máximo de unos 5.4 nats hacia la época 630 y después oscila alrededor de 5 (4.44 al final). Es el compromiso que impone el costo: el encoder gasta unos 5 nats por bloque para describirlo con 2 números, y el término KL le exige que esas descripciones queden cerca del prior N(0, I).
- **Total.** Sigue de cerca a la reconstrucción, porque el KL es unas 100 veces menor.

## Reconstrucción y espacio latente

Reconstruir desde la media del encoder (sin ruido) da:

```
RECONSTRUCCION: error medio 2.54 niveles por canal | pixeles incorrectos (algun canal a mas de 32 niveles): media 0.00, maximo 0 de 256 por bloque; 18/18 bloques sin pixeles incorrectos
```

Un píxel se cuenta como incorrecto si alguno de sus tres canales difiere en más de 32 niveles del original. Ninguno de los 18 bloques tiene un píxel incorrecto y el error medio es de 2.54 niveles por canal, un error que no se ve a simple vista. La figura `output/ej2/vae_reconstruction.png` muestra los originales y debajo sus reconstrucciones: se leen como el mismo bloque, con las vetas en el mismo lugar y los mismos colores.

**El juez de bloques.** Para medir el espacio latente y para decidir si una muestra es nueva hace falta decir qué mineral y qué roca tiene una imagen decodificada. El juez lo hace en dos partes y con umbrales derivados del dataset:

- **Roca.** Es la textura base (piedra, pizarra profunda o netherrack) con la que coinciden más píxeles, con la misma tolerancia de 32 niveles por canal. El fondo es de esa roca si coincide en más de la mitad de los píxeles (al menos 129 de 256).
- **Minerales.** La imagen se descompone como mezcla convexa de los 18 bloques del dataset (pesos no negativos que suman 1, por mínimos cuadrados con descenso por gradiente proyectado sobre el simplex). Un mineral está presente si los bloques de ese mineral suman al menos el 25 % del peso. Como un VAE con 2 dimensiones latentes entrenado con 18 imágenes genera esencialmente mezclas de ellas, esta descomposición describe bien sus salidas, y es robusta a la borrosidad: una mezcla 50/50 de dos bloques se descompone como 50/50 aunque sus colores sean intermedios.
- **Plausible.** El fondo es de una roca, la mezcla explica la imagen con un residuo de a lo sumo la mitad de la separación mínima del dataset (10.45 niveles; la mezcla explica la imagen mejor que la distancia entre los dos bloques más parecidos) y la imagen tiene al menos el contraste de la reconstrucción menos contrastada del modelo. El contraste (desvío medio de cada canal respecto del color medio de la imagen) descarta las imágenes lisas y borrosas: sin él, un gris liso pasaba como un bloque de piedra.
- **Reconocible** es plausible y con una combinación (roca, mineral) del dataset; **nuevo** es plausible y con una combinación ausente.

La línea medida (los números entre paréntesis son los del modelo sobre sus propias reconstrucciones, que pasan todos los umbrales con margen) es:

```
JUEZ: roca = textura base con mas pixeles coincidentes, al menos 129 de 256 (reconstrucciones: al menos 158) | minerales = los de al menos 25 % del peso de la mezcla convexa de los 18 bloques (mayor peso ajeno en las reconstrucciones: 6.6 %) | residuo de la mezcla a lo sumo 10.45 niveles, la mitad de la separacion minima (reconstrucciones: a lo sumo 3.35) | contraste al menos 15.8 niveles, el de la reconstruccion menos contrastada
```

**El plano latente decodificado.** La figura `output/ej2/vae_manifold.png` decodifica una grilla de 15 x 15 puntos sobre z en [-2.5, 2.5] x [-2.5, 2.5] (z1 crece hacia la derecha y z2 hacia arriba). La línea medida es:

```
MANIFOLD: 225 celdas (15x15) sobre z en [-2.5, 2.5]^2 | plausibles 72.9 %, reconocibles 60.9 % | salto medio entre celdas vecinas 11.4 niveles (maximo 35.5; separacion mediana del dataset 25.8) | saltos de mas de 77.3 niveles: 0.0 %
```

Lo que se ve en la imagen: el plano está organizado por color de mineral y por roca. Arriba están el lapislázuli (azul) y el diamante (celeste), a la izquierda el redstone (rojo) y el oro (amarillo), a la derecha la esmeralda (verde) y abajo a la izquierda el Nether, con el fondo rojo del netherrack, el cuarzo y el oro del Nether. Entre esas regiones las transiciones son graduales: el salto medio entre celdas vecinas (11.4 niveles) es menos de la mitad de la separación mediana entre bloques del dataset (25.8) y ningún par vecino salta más de 77.3 niveles. Las celdas no plausibles se concentran en las esquinas y en las zonas donde se encuentran dos rocas, donde el fondo queda mezclado.

**Contraste con un autoencoder sin esquema variacional.** Para mostrar que el esquema variacional es lo que vuelve muestreable el espacio latente, `ej2` entrena además un autoencoder simple con exactamente la misma arquitectura (latente de 2), optimizador, pérdida, épocas y semilla, pero sin capa estocástica ni término KL. Para ambos modelos se decodifican 1000 puntos z ~ N(0, I) con la misma semilla, y cada uno se juzga con los pisos de sus propias reconstrucciones:

```
VAE PRIOR: de 1000 puntos z ~ N(0, I): 77.0 % decodifican a un bloque plausible (fondo de una roca, mezcla de bloques del dataset y contraste de un bloque real), 57.2 % a un bloque reconocible (plausible y con una combinacion del dataset), 19.8 % a un bloque nuevo (plausible y con una combinacion ausente) en 33 combinaciones distintas
AE simple PRIOR: de 1000 puntos z ~ N(0, I): 40.6 % decodifican a un bloque plausible (fondo de una roca, mezcla de bloques del dataset y contraste de un bloque real), 27.8 % a un bloque reconocible (plausible y con una combinacion del dataset), 12.8 % a un bloque nuevo (plausible y con una combinacion ausente) en 9 combinaciones distintas
CONTRASTE: puntos z ~ N(0, I) que decodifican a un bloque plausible: VAE 77.0 % contra autoencoder sin esquema variacional 40.6 % | reconocible: VAE 57.2 % contra 27.8 % | nuevo: VAE 19.8 % contra 12.8 % | salto medio entre celdas vecinas: VAE 11.4 contra 13.4 niveles | reconstruccion media: VAE 0.00 px contra 0.00 px por bloque
```

Interpretación:

- Los dos modelos reconstruyen los 18 bloques sin píxeles incorrectos: tener un buen autoencoder no alcanza para poder generar.
- Los códigos del autoencoder simple no están distribuidos como N(0, I): al sortear del prior, el 40.6 % de los puntos decodifica a un bloque plausible, contra el 77.0 % del VAE, y el 27.8 % a un bloque reconocible, contra el 57.2 %. El término KL del VAE es lo que acerca las posteriores al prior y así hace que muestrear N(0, I) tenga sentido.
- El VAE también genera más variedad nueva: 33 combinaciones nuevas distintas contra 9.
- La grilla del autoencoder simple (`output/ej2/plain_manifold.png`, hecha sobre la caja de sus propios códigos con 10 % de margen) también recorre todos los minerales, pero tiene regiones de colores lavados o saturados (arriba a la izquierda, celdas casi blancas) y bordes más duros entre regiones; el salto medio entre celdas vecinas es 13.4 niveles contra 11.4 del VAE. La diferencia principal no está en esa grilla sino en el prior: la grilla del autoencoder simple está hecha donde están sus códigos, que van de -6.5 a 7.0 en z1 y de -7.3 a 0.8 en z2, y solo 3 de los 18 están a menos de 1 unidad del origen; los sorteos de N(0, I) se concentran justamente cerca del origen, donde ese decoder vio pocos bloques.

**Comprobaciones de aceptación.** El resultado se verifica con 11 comprobaciones que el código evalúa y la corrida imprime:

```
ACEPTACION OK: reconstruccion media (px incorrectos por bloque) = 0.000 (umbral <= 12.000)
ACEPTACION OK: KL final (nats) = 4.443 (umbral >= 1.000)
ACEPTACION OK: celdas plausibles de la grilla = 0.729 (umbral >= 0.500)
ACEPTACION OK: salto medio entre celdas vecinas (niveles) = 11.391 (umbral <= 38.666)
ACEPTACION OK: sorteos del prior plausibles = 0.770 (umbral >= 0.600)
ACEPTACION OK: sorteos del prior nuevos = 0.198 (umbral >= 0.050)
ACEPTACION OK: ventaja sobre el autoencoder simple (sorteos plausibles) = 0.364 (umbral >= 0.300)
ACEPTACION OK: combinaciones nuevas distintas en el prior = 33.000 (umbral >= 8.000)
ACEPTACION OK: peso de un mineral ajeno en las reconstrucciones = 0.066 (umbral <= 0.250)
ACEPTACION OK: control negativo: reconstrucciones del entrenamiento aceptadas como nuevas = 0.000 (umbral <= 0.000)
ACEPTACION OK: control negativo: muestras de la posterior aceptadas como nuevas = 0.026 (umbral <= 0.100)
```

Los umbrales son convenciones de este trabajo y se mantuvieron los del dataset anterior donde la magnitud seguía teniendo sentido (12 píxeles incorrectos, KL de al menos 1 nat, 50 % de la grilla, 1.5 veces la separación mediana para el salto, 60 % del prior, 5 % de nuevos, ventaja de 0.3, 8 combinaciones y los dos controles). La comprobación del peso de un mineral ajeno es nueva: verifica que las reconstrucciones de los bloques de entrenamiento no muestran un segundo mineral, es decir, que el umbral de 25 % no detecta minerales donde no los hay. Las pruebas (`tests/vae_acceptance.rs`) repiten las comprobaciones con las semillas de entrenamiento 0, 1 y 2.

## Generación de un bloque nuevo

**Respuesta a 2c.** La consigna pide generar una muestra nueva desde el espacio latente y juzgar si pertenece al dataset. Con este dataset, nuevo tiene un sentido concreto: una combinación de roca y minerales que el dataset no tiene. Pertenecer al dataset también: un bloque cuyo fondo es una de las tres rocas, cuyas vetas son de minerales del dataset y que tiene la nitidez de un bloque real (el criterio de plausibilidad de la sección anterior).

**Regla de la galería.** El flujo de N(0, I) tiene semilla fija (1514078721) y se examinan 1000 sorteos. Entre los sorteos aceptados como nuevos se conserva uno por combinación, el de mezcla más equilibrada (el mayor peso mínimo entre sus minerales; ante un empate, el de menor residuo y después el sorteo anterior), y se muestran como máximo 8 en ese orden. Un bloque nuevo con mezcla equilibrada muestra claramente sus dos minerales; uno de 75/25 se parece mucho al bloque dominante. El bloque principal es el primero de la galería, elegido por esta regla y no a ojo. La búsqueda dio:

```
BLOQUE NUEVO: z=(0.6041, 0.9937) sorteo 450 de 1000 | esmeralda+lapis-piedra (combinacion ausente del dataset) | mezcla: esmeralda 50 %, lapis 50 % | fondo: 161 de 256 px de piedra (minimo 129) | residuo 5.28 niveles (tope 10.45) | bloque mas cercano: lapis-piedra a 18.0 niveles | criterio: fondo de una roca conocida, mezcla de bloques del dataset con residuo dentro del tope, contraste de un bloque real y una combinacion de roca y minerales que no existe en el dataset
```

```
GALERIA: 198 sorteos aceptados como bloques nuevos de 1000, 33 combinaciones nuevas distintas; se muestran 8 ordenadas por equilibrio de la mezcla
  nuevo 1: sorteo 450 | esmeralda+lapis-piedra | mezcla: esmeralda 50 %, lapis 50 % | fondo 161 px | residuo 5.28 niveles | bloque mas cercano lapis-piedra a 18.0 niveles
  nuevo 2: sorteo 542 | cobre+redstone-piedra | mezcla: cobre 50 %, redstone 50 % | fondo 191 px | residuo 4.48 niveles | bloque mas cercano cobre-piedra a 12.7 niveles
  nuevo 3: sorteo 941 | oro+redstone-piedra | mezcla: oro 52 %, redstone 48 % | fondo 166 px | residuo 6.00 niveles | bloque mas cercano oro-piedra a 15.2 niveles
  nuevo 4: sorteo 730 | esmeralda+diamante-pizarra | mezcla: esmeralda 49 %, diamante 48 % | fondo 179 px | residuo 5.16 niveles | bloque mas cercano esmeralda-pizarra a 16.9 niveles
  nuevo 5: sorteo 160 | carbon+esmeralda-pizarra | mezcla: carbon 53 %, esmeralda 47 % | fondo 189 px | residuo 5.11 niveles | bloque mas cercano carbon-pizarra a 14.1 niveles
  nuevo 6: sorteo 458 | carbon+hierro-piedra | mezcla: carbon 46 %, hierro 46 % | fondo 144 px | residuo 6.28 niveles | bloque mas cercano carbon-piedra a 19.5 niveles
  nuevo 7: sorteo 173 | carbon+cobre-pizarra | mezcla: carbon 46 %, cobre 46 % | fondo 197 px | residuo 5.92 niveles | bloque mas cercano cobre-pizarra a 14.1 niveles
  nuevo 8: sorteo 184 | carbon+esmeralda-piedra | mezcla: carbon 49 %, esmeralda 45 % | fondo 153 px | residuo 9.08 niveles | bloque mas cercano carbon-piedra a 18.9 niveles
```

El bloque principal es una mena de esmeralda y lapislázuli en piedra, mitad y mitad: un bloque que no existe en el dataset ni en el juego. Su fondo coincide con la piedra en 161 de 256 píxeles, la mezcla lo explica con un residuo de 5.28 niveles (el tope es 10.45) y el bloque del dataset más cercano, `lapis-piedra`, está a 18.0 niveles por canal, una distancia del orden de la que separa a dos bloques distintos del dataset (la mínima es 20.9).

**Control negativo.** Si el criterio aceptara cualquier salida del decodificador, no probaría nada. Por eso se lo aplica a imágenes que seguro no son nuevas: las reconstrucciones de los bloques del entrenamiento.

```
CONTROL NEGATIVO: reconstrucciones de los 18 bloques del entrenamiento aceptadas como nuevas: 0/18 (reconocidas con su propia combinacion: 18/18) | muestras de la posterior q(z|x), 30 por bloque: 2.6 % aceptadas como nuevas
```

Se lee así: ninguna de las 18 reconstrucciones (media de la posterior) se acepta como nueva, y las 18 se reconocen con su propia combinación de mineral y roca, es decir, el juez identifica correctamente cada bloque del dataset. Que pasen la plausibilidad es en parte cierto por construcción, porque el umbral de contraste es el de la reconstrucción menos contrastada; la identificación de mineral y roca, en cambio, no depende de ningún piso medido sobre ellas. La evidencia independiente es la segunda mitad de la línea: de las reconstrucciones estocásticas (30 muestras de la posterior q(z|x) por bloque, con una semilla propia del control) el 2.6 % se acepta como nueva, unas ocho veces menos que el 19.8 % de los sorteos del prior. El criterio no acepta cualquier salida del decodificador.

La figura `output/ej2/vae_new_ore.png` muestra los 8 bloques de la galería en orden, cada uno con sus minerales.

**Juicio visual de los autores.** Lo que vemos al abrir la figura, entrada por entrada:

- **Entradas 1 y 3** (esmeralda y lapislázuli en piedra; oro y redstone en piedra): son las más claras. En la 1 se ven gemas verdes y vetas azules en el mismo bloque, y en la 3 vetas amarillas y rojas; las dos se leen como una mena de Minecraft con dos minerales.
- **Entradas 2 y 5** (cobre y redstone en piedra; carbón y esmeralda en pizarra): se ven los dos minerales, pero domina uno. En la 2 manda el rojo del redstone con tonos anaranjados del cobre; en la 5 se ven gemas verdes sobre manchas oscuras de carbón.
- **Entrada 4** (esmeralda y diamante en pizarra): es sutil, porque el verde de la esmeralda y el celeste del diamante son parecidos; se lee como una gema verde azulada.
- **Entradas 6, 7 y 8** (carbón con hierro, con cobre y con esmeralda): son las más débiles. El carbón y el hierro tienen poco color, y la mezcla da vetas oscuras y apagadas; se reconocen como menas plausibles, pero no se distingue a simple vista que tengan dos minerales.

En resumen, 4 de las 8 entradas se ven como bloques nuevos a simple vista, 1 es sutil y 3 son débiles. Los minerales con color propio (esmeralda, lapislázuli, oro, redstone) producen bloques nuevos claros; los apagados (carbón, hierro) producen mezclas que el juez acepta pero que una persona difícilmente distinga.

Lo que el criterio no puede mostrar: describe cada muestra como mezcla de bloques del dataset, y por eso solo puede encontrar novedad que sea combinación de lo que ya hay (dos minerales juntos, o un mineral en otra roca); no puede detectar un mineral inventado con un color que no está en el dataset. Que un mineral pese al menos 25 % en la mezcla significa que el VAE lo está combinando, no que una persona lo vaya a ver con claridad. Este es el juicio de los autores y debe leerse junto con la imagen.

## Alcance y limitaciones

- El dataset tiene 18 imágenes reales con dos atributos (mineral y roca) y cubre 18 de las 27 combinaciones. Los bloques nuevos que encuentra el VAE son todos dos o más minerales en la misma roca: en los 1000 sorteos, ninguna de las 33 combinaciones nuevas es un mineral solo en una roca ajena (por ejemplo, diamante en netherrack), porque entre las regiones del plano latente del Nether y las de piedra o pizarra el fondo queda mezclado y no es plausible.
- Las texturas son de Mojang y no se versionan: reproducir el ejercicio necesita bajarlas una vez de los servidores de Mojang (`scripts/descargar_menas.sh`). El script verifica el SHA-1 del cliente y el SHA-256 de cada textura, así que si Mojang cambiara los archivos la descarga fallaría en vez de cambiar el dataset en silencio.
- El juez describe las imágenes como mezcla convexa de los 18 bloques. Es adecuado para un VAE de 2 dimensiones entrenado con 18 imágenes, que genera esencialmente mezclas, pero no reconocería un mineral con un color que no está en el dataset.
- Los pisos del juez (fondo, residuo, peso ajeno y contraste) se miden sobre las reconstrucciones del propio modelo, y el de contraste es además un umbral del criterio. Por eso que las 18 reconstrucciones sean plausibles es en parte cierto por construcción, y la evidencia independiente es el control con reconstrucciones estocásticas (2.6 % aceptadas como nuevas).
- El criterio de novedad acepta mezclas de minerales apagados que no se distinguen a simple vista (3 de las 8 entradas de la galería); el juicio final de si un bloque es nuevo es visual y subjetivo.
- El tamaño latente está fijado en 2 y no se estudió otra arquitectura, porque la consigna no lo pide; los resultados corresponden a una red elegida con la semilla de entrenamiento 0, y las pruebas repiten las comprobaciones de aceptación con las semillas 0, 1 y 2.
- La reconstrucción no puede bajar de la entropía de los objetivos (unos 466.3 nats por bloque), así que el valor absoluto de la BCE no mide por sí solo la calidad; la calidad se mide en píxeles incorrectos y en niveles por canal.
- No se usó recocido ni ponderación del término KL distinta de 1.
- El contraste con el autoencoder simple usa una sola arquitectura y una sola semilla; muestra la diferencia para esa configuración y no la generaliza a cualquier autoencoder.
