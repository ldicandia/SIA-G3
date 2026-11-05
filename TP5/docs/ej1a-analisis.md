# Ejercicio 1a: autoencoder básico con espacio latente de 2 dimensiones

Análisis del Ejercicio 1a del TP5 (Sistemas de Inteligencia Artificial, ITBA). El objetivo de la consigna es un autoencoder con espacio latente de 2 dimensiones que reproduzca los 32 patrones de 7x5 de `font.h` con un error máximo de 1 pixel por patrón. Todo el código (perceptrón multicapa, backpropagation, SGD, Momentum y Adam) es propio, sobre `ndarray`. Todos los números de este documento salen de la salida de `cargo run --release -- ej1a` y de `cargo run --release -- sweep`, y el test `tests/doc_consistency.rs` verifica que coincidan con lo que produce el código.

Para regenerar todo: `cargo run --release -- sweep` (barrido completo, curvas y comparación) y `cargo run --release -- ej1a` (reporte, subconjunto, plano latente y letra nueva).

## Arquitecturas y técnicas probadas

El autoencoder es, como en la teoría de la materia, dos perceptrones multicapa unidos por una capa central: el encoder produce el código latente `Z = f(X)` y el decoder, con la distribución de neuronas invertida, produce `X' = g(Z)`. Se entrena con cada patrón `X` como salida esperada de sí mismo. La capa central tiene siempre 2 neuronas (restricción de la consigna), de modo que la red debe encontrar una codificación más eficiente que los 35 pixeles de entrada.

**Cómo se entrena y se evalúa.** Entrenamiento por lote completo (los 32 patrones en cada época, 10000 épocas), inicialización Xavier uniforme de los pesos con sesgos en cero y generador ChaCha8 con semilla explícita (semillas 0, 1 y 2 para cada variante; cada corrida es reproducible bit a bit). La capa de salida es siempre sigmoide. Cada 25 épocas, y en la última, se binariza la reconstrucción en 0.5 (igual que el patrón objetivo) y se cuentan los pixeles incorrectos por patrón; el criterio de éxito es que ninguno de los 32 tenga más de 1 pixel incorrecto (tolerancia de 1 pixel). La métrica final se calcula sobre los parámetros finales, no sobre el historial: una corrida puede cruzar transitoriamente la tolerancia y luego salirse por una inestabilidad, y eso cuenta como fallo.

Se hicieron cuatro estudios, cada uno cambiando un solo factor respecto de una base (los resultados completos, una fila por corrida, están en `output/ej1a/sweep.csv`):

1. **Arquitectura** (`arch`): Adam con tasa 0.005, entropía cruzada binaria, activación tanh en las capas ocultas y latente identidad. Se probaron cinco tamaños del encoder (el decoder los refleja): `35-24-2-24-35` (`arch-h24`), `35-20-10-2-10-20-35` (`arch-h20x10`), `35-32-16-2-16-32-35` (`arch-h32x16`), `35-64-32-2-32-64-35` (`arch-h64x32`) y `35-48-24-12-2-12-24-48-35` (`arch-h48x24x12`).
2. **Optimizador** (`optim`): ocultas `[20, 10]`, entropía cruzada, tanh e identidad en el latente. SGD con tasas 0.5 y 0.1; Momentum con beta 0.9 y tasas 0.1 y 0.01; Adam (beta1 0.9, beta2 0.999, epsilon 1e-8) con tasas 0.001, 0.005 y 0.01.
3. **Activaciones** (`act`): ocultas `[20, 10]`, Adam 0.005, entropía cruzada. Combinaciones de activación oculta y activación de la capa latente: tanh con latente identidad, tanh con latente tanh, ReLU con latente identidad y ReLU con latente tanh.
4. **Función de pérdida** (`loss`): ocultas `[20, 10]`, Adam 0.005, tanh e identidad. Entropía cruzada binaria (BCE) frente a error cuadrático medio (MSE), ambas con salida sigmoide.

La pérdida es la suma sobre los 35 pixeles de cada patrón, promediada sobre el lote. Las curvas de la semilla 0 de cada variante están en `output/ej1a/curves_{arch,optim,act,loss}_loss.png` (pérdida de entrenamiento en escala logarítmica) y `output/ej1a/curves_{arch,optim,act,loss}_wrongpx.png` (máximo de pixeles incorrectos por época).

## Comparación de resultados

Las tablas siguientes son `output/ej1a/comparison.md` (generadas por `cargo run --release -- sweep`, 54 corridas: 18 variantes por 3 semillas). La segunda columna cuenta las semillas cuyos 32 patrones terminaron con a lo sumo 1 pixel incorrecto; la tercera es el mejor máximo de pixeles incorrectos entre las semillas; la cuarta es la mediana de la primera época (múltiplo de 25) en que los 32 patrones estuvieron dentro de la tolerancia, y la quinta la mediana de la pérdida final.

### Estudio: arch

| variante | semillas con los 32 patrones a lo sumo 1 pixel | mejor maximo de pixeles incorrectos | mediana de primera epoca dentro de tolerancia | mediana de perdida final |
|---|---|---|---|---|
| arch-h24 | 3/3 | 0 | 1100 | 0.005968 |
| arch-h20x10 | 2/3 | 0 | 1225 | 0.019764 |
| arch-h32x16 | 3/3 | 0 | 575 | 0.001721 |
| arch-h64x32 | 3/3 | 0 | 275 | 0.000143 |
| arch-h48x24x12 | 3/3 | 0 | 400 | 0.000312 |

### Estudio: optim

| variante | semillas con los 32 patrones a lo sumo 1 pixel | mejor maximo de pixeles incorrectos | mediana de primera epoca dentro de tolerancia | mediana de perdida final |
|---|---|---|---|---|
| optim-sgd-lr0.5 | 0/3 | 25 | - | 18.928494 |
| optim-sgd-lr0.1 | 0/3 | 33 | - | 20.802142 |
| optim-mom-lr0.1-b0.9 | 0/3 | 25 | - | 20.802142 |
| optim-mom-lr0.01-b0.9 | 1/3 | 0 | 2500 | 19.132258 |
| optim-adam-lr0.001 | 2/3 | 0 | 3050 | 0.064519 |
| optim-adam-lr0.005 | 2/3 | 0 | 1225 | 0.019764 |
| optim-adam-lr0.01 | 3/3 | 0 | 450 | 0.001212 |

### Estudio: act

| variante | semillas con los 32 patrones a lo sumo 1 pixel | mejor maximo de pixeles incorrectos | mediana de primera epoca dentro de tolerancia | mediana de perdida final |
|---|---|---|---|---|
| act-tanh-latid | 2/3 | 0 | 1225 | 0.019764 |
| act-tanh-lattanh | 3/3 | 0 | 975 | 0.006146 |
| act-relu-latid | 3/3 | 0 | 1500 | 0.000208 |
| act-relu-lattanh | 3/3 | 0 | 1325 | 0.000127 |

### Estudio: loss

| variante | semillas con los 32 patrones a lo sumo 1 pixel | mejor maximo de pixeles incorrectos | mediana de primera epoca dentro de tolerancia | mediana de perdida final |
|---|---|---|---|---|
| loss-bce | 2/3 | 0 | 1225 | 0.019764 |
| loss-mse | 0/3 | 2 | 1012 | 0.203681 |

**Lectura de las tablas** (la fuente es `output/ej1a/comparison.md`; el detalle por semilla, en `output/ej1a/sweep.csv`).

- *Optimizador.* Es el factor que más separa los resultados. SGD (tasas 0.5 y 0.1) y Momentum con tasa 0.1 fallan en las tres semillas: terminan con entre 25 y 33 pixeles incorrectos en el mejor caso y una pérdida final cercana a 19-21. Seis de esas corridas terminan con exactamente la misma pérdida, 20.80214222, aun con semillas y optimizadores distintos; es la firma de un punto fijo degenerado al que la red cae y del que no sale (probablemente una salida casi independiente del patrón), no de un entrenamiento lento. En `output/ej1a/curves_optim_loss.png` se ven las curvas de SGD y de Momentum 0.1 casi planas en torno a 20, sin bajar como las de Adam. Momentum con tasa 0.01 logra una semilla de tres (`optim-mom-lr0.01-b0.9-s1`, 0 pixeles, primera época dentro de tolerancia 2500) pero su pérdida mediana sigue en 19.13. Adam, en cambio, converge con las tres tasas probadas: 2/3 con 0.001 (mediana 3050 épocas hasta la tolerancia), 2/3 con 0.005 (1225) y 3/3 con 0.01 (450 épocas y pérdida mediana 0.001212). Entre las tasas de Adam, una tasa mayor llega antes: pasar de 0.001 a 0.01 reduce la mediana de épocas de 3050 a 450.
- *Arquitectura.* Con Adam 0.005 y BCE, todas las arquitecturas llegan a 0 pixeles en su mejor semilla, pero la robustez y la velocidad mejoran con la capacidad. `arch-h20x10` es la única con una semilla fallida (`arch-h20x10-s1`: 7 pixeles incorrectos en el peor patrón y pérdida final 0.919788). Las tres arquitecturas más grandes (`arch-h32x16`, `arch-h48x24x12`, `arch-h64x32`) quedan 3/3, y `arch-h64x32` es la más rápida (mediana 275 épocas) y la de menor pérdida (0.000143). Una sola capa oculta de 24 neuronas (`arch-h24`) también es 3/3 pero tarda 1100 épocas: no hace falta profundidad para lograr el criterio, la capacidad solo lo acelera.
- *Activaciones.* Con ocultas `[20, 10]`, la base (tanh con latente identidad) es 2/3; ponerle tanh al latente o cambiar a ReLU en las ocultas lleva las tres combinaciones restantes a 3/3. El estudio no muestra una activación superior en velocidad: tanh con latente tanh tarda menos en alcanzar la tolerancia (975) que las dos de ReLU (1500 y 1325), mientras que las de ReLU llegan a menor pérdida final (0.000208 y 0.000127 contra 0.006146).
- *Pérdida.* Con todo lo demás igual, BCE logra 2/3 y MSE 0/3, y la mejor semilla con MSE termina con 2 pixeles incorrectos (el criterio pide como máximo 1). La mediana de "primera época dentro de tolerancia" de MSE (1012) es engañosa: dos de sus tres semillas cruzaron la tolerancia en algún momento y luego la perdieron, y la evaluación final las cuenta como fallo.
- *Inestabilidad.* En las curvas con Adam aparecen saltos abruptos de la pérdida y del máximo de pixeles incorrectos, por ejemplo cerca de la época 1600 con tasa 0.005 en `output/ej1a/curves_optim_loss.png` y los saltos de `arch-h24`, `arch-h20x10` y `arch-h32x16` en `output/ej1a/curves_arch_wrongpx.png`, tras los cuales la red se recupera. Por eso el criterio se evalúa sobre los parámetros finales y no sobre el primer cruce de la tolerancia.

Hay una limitación que conviene dejar dicha: cada estudio varía un factor respecto de una base, y los estudios `optim`, `act` y `loss` usan ocultas `[20, 10]` mientras que la configuración elegida usa `[64, 32]`. Las comparaciones valen para esa base; no se probó la grilla completa de combinaciones.

## Configuración elegida y justificación

La configuración elegida es **`arch-h64x32-s0`**: arquitectura `35-64-32-2-32-64-35` (ocultas `[64, 32]` en el encoder y reflejadas en el decoder), activación tanh en las capas ocultas, capa latente de 2 neuronas con activación identidad, salida sigmoide, pérdida BCE, optimizador Adam con tasa 0.005 (beta1 0.9, beta2 0.999), 10000 épocas, semilla 0. El criterio de selección fue: entre las variantes robustas (todas sus semillas con a lo sumo 1 pixel), la de menor mediana de épocas hasta la tolerancia (275 para `arch-h64x32`); dentro de ella, la semilla de menor pérdida final. Las tres semillas terminan con 0 pixeles; la semilla 0 tiene la menor pérdida (0.00013432, contra 0.00014268 y 0.00015151 de las semillas 1 y 2 en `output/ej1a/sweep.csv`).

Resultado impreso por `ej1a` para esta configuración:

```
RESULTADO: 32/32 patrones con a lo sumo 1 pixel incorrecto (maximo 0)
```

La justificación de cada decisión, ligada a la teoría de la materia y a las filas observadas:

- **Cuello de botella de 2 neuronas.** La teoría plantea que la restricción natural del autoencoder es usar menos neuronas en la capa central que en la entrada, lo que fuerza una codificación más eficiente del conjunto de datos. Los 32 patrones de 35 pixeles se comprimen a un punto de 2 coordenadas y se reconstruyen sin ningún pixel incorrecto: la red no copia la entrada, comprime y decodifica.
- **Entrenamiento como un MLP con salida igual a la entrada.** El autoencoder se entrena con cualquier método válido para un perceptrón multicapa, poniendo como salida esperada el mismo patrón. Se aplica backpropagation por la regla de la cadena a través del decoder, la capa latente y el encoder (derivada en la capa latente incluida), y el código se validó con un test de gradientes por diferencias finitas.
- **Entropía cruzada con salida sigmoide.** Los pixeles son binarios y la salida es una probabilidad, por lo que la entropía cruzada binaria es la pérdida natural; además, su gradiente respecto de la entrada de la neurona de salida se simplifica a `(p - t)`, sin el factor de la derivada de la sigmoide que reduce el gradiente de MSE cuando la salida satura. El estudio `loss` es coherente con esto: BCE logra 2/3 y MSE 0/3 en las mismas condiciones, y MSE queda con una pérdida mediana de 0.203681.
- **Tanh en las ocultas, latente identidad.** La tanh está centrada en cero, y la identidad en el latente deja los códigos sin acotar: los 32 códigos de `output/ej1a/latent_codes.csv` ocupan un rango de aproximadamente -16.2 a 5.9 en `z1` y de -15.6 a 15.7 en `z2`, lo que deja espacio para separarlos. Las filas del estudio `act` matizan esta elección: con ocultas `[20, 10]`, la combinación tanh con latente identidad es la única de las cuatro que queda en 2/3. Con la mayor capacidad de `[64, 32]` la misma combinación es 3/3 (`arch-h64x32`), así que no se la eligió por ser superior en el estudio `act` sino porque, con esa capacidad, no es un factor limitante.
- **Adam.** SGD y Momentum con tasas altas quedan atrapados en un punto degenerado (0/3, pérdida cercana a 20), mientras que Adam, que adapta el paso de cada parámetro con estimaciones del primer y segundo momento del gradiente (combina la inercia de Momentum con un escalado por la magnitud reciente del gradiente), converge con las tres tasas probadas y llega a pérdidas de 0.001 a 0.06 en 10000 épocas. Dentro de Adam, la tasa 0.005 es un compromiso: más rápida que 0.001 (1225 contra 3050 épocas) y menos agresiva que 0.01; la configuración elegida corrió con 0.005 y no con 0.01 porque la selección partió de las variantes robustas ordenadas por velocidad, y `arch-h64x32` ganó dentro de la grilla con esa tasa.
- **Varias semillas.** Con la misma configuración, `arch-h20x10` pasa de 0 pixeles (semillas 0 y 2) a 7 pixeles (semilla 1). Un resultado de una sola semilla habría ocultado esa varianza; por eso cada variante se corrió con tres y se eligió una variante robusta, no una semilla afortunada. Que la configuración elegida sea una semilla concreta se debe a que el entrenamiento es determinista y se necesita un único autoencoder entrenado para las figuras y para las fases siguientes.

Una salvedad de alcance: el criterio de la consigna es reconstruir los mismos 32 patrones con los que se entrena, por lo que no se afirma nada sobre generalización a patrones no vistos; la selección de la configuración también se hizo sobre esos 32 patrones.

## Subconjunto aprendido (AE-04)

Con la configuración elegida se aprenden **los 32 patrones**: ninguno supera 1 pixel incorrecto, y de hecho el máximo es 0. La línea impresa por `ej1a` es:

```
SUBCONJUNTO: se aprendieron los 32 patrones (ninguno supera 1 pixel incorrecto); no queda ningun patron sin aprender
```

El camino de análisis de fallos existe (`src/analysis.rs`: para cada patrón no aprendido informa los pixeles incorrectos, su vecino más cercano en el espacio latente con la distancia y la distancia de Hamming entre ambos patrones, y una causa posible: códigos casi superpuestos, meseta de la pérdida o entrenamiento no concluyente) y está cubierto por tests sintéticos, pero con esta configuración no se activó: no hay patrones fallidos que explicar. Las corridas que sí dejaron patrones sin aprender (por ejemplo `arch-h20x10-s1`, 30 de 32 patrones dentro de la tolerancia, o las de SGD) están en `output/ej1a/sweep.csv` y no son la configuración elegida.

## Espacio latente

`output/ej1a/latent.png` ubica los 32 patrones en el plano latente de la configuración elegida (cada punto con su carácter; las coordenadas exactas están en `output/ej1a/latent_codes.csv`). Los 32 códigos son distintos entre sí. Lo que se ve en el gráfico:

- Las letras `b`, `h`, `k`, `p`, `r` y `n` quedan juntas en el borde izquierdo (`z1` entre -16.2 y -14.4). En `font.h` todas comparten el trazo vertical del lado izquierdo, lo que sugiere que el encoder agrupa por esa característica.
- Sobre una diagonal hacia abajo a la izquierda se ordenan `m`, `e`, `o`, `w`, `u` y `a` (de `z1` = -13.3, `z2` = -5.8 hacia `z1` = -3.7, `z2` = -14.6).
- Los signos de puntuación y letras de trazo vertical con poco relleno (`` ` ``, `|`, `l`, `~` e `i`) quedan arriba, con `z2` entre 10.9 y 15.7.
- Un grupo denso rodea el origen: `t`, `x`, `z`, `g`, `y`, `s`, `j`, `{` y `}`, con `z1` entre -2.6 y 1.2 y `z2` entre -2.1 y 3.9. Es la zona con menor separación entre códigos y donde se ubica la letra nueva generada más abajo.
- `DEL` (todos los pixeles encendidos) es el punto más aislado y el de mayor `z1` (5.8880); `q` (`z2` = -15.5991) y `d` también quedan del lado de `z1` positivo.

La descripción es cualitativa: se tomó de mirar el gráfico y las coordenadas del CSV, no de una métrica de agrupamiento.

## Letra nueva

Para generar una letra que no pertenece al conjunto de entrenamiento, `generate` recorre una grilla de 81 x 81 puntos del plano latente (la caja que contienen los 32 códigos, ampliada un 10 % por cada lado), decodifica todos en un lote, binariza la salida en 0.5 y descarta las que coinciden con alguno de los 32 patrones. Entre las letras restantes (primera aparición de cada patrón distinto) elige la que está a menor distancia de Hamming de un patrón real, pero siempre a una distancia de al menos 2 pixeles: así no puede pasar por una reconstrucción de un patrón con 1 pixel de error (tolerancia de la consigna).

El punto latente elegido es `(-0.1725, -0.4338)`, y su decodificación es:

```
.....
.....
#####
.#.#.
..#..
.#...
####.
```

La letra más cercana del entrenamiento es `z` (distancia de Hamming 2), y el código de entrenamiento más próximo está a distancia latente 0.5767. La figura `output/ej1a/new_letter.png` muestra la letra nueva junto a la `z` más cercana.

Valoración subjetiva: se parece a una `z` deformada (la barra superior y la inferior están, pero la diagonal queda recargada de pixeles encendidos y la barra inferior pierde la esquina derecha); es una forma nueva, pero no una letra legible. Es el resultado esperable de decodificar un punto de una zona densa del plano latente que no es el código de ningún patrón. Con `cargo run --release -- generate --point Z1 Z2` se puede decodificar cualquier punto; por ejemplo, `--point 0 0` cae en el código de `t` (distancia de Hamming 0) y devuelve ese patrón.
