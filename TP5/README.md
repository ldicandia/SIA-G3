# SIA TP5: Deep Learning (autoencoders)

## Qué es

TP5 de Sistemas de Inteligencia Artificial (ITBA). Implementa en Rust, sobre `ndarray`, con backpropagation y optimizadores propios (SGD, Momentum y Adam) y sin frameworks de ML, los tres ejercicios de la consigna:

- **Ej 1a**: autoencoder básico con espacio latente de 2 dimensiones sobre los 32 patrones de 7x5 de `docs/font.h`, con un máximo de 1 pixel incorrecto por patrón.
- **Ej 1b**: autoencoder denoising (entrada distorsionada, salida esperada limpia) y análisis del nivel de ruido que logra eliminar.
- **Ej 2**: autoencoder variacional (VAE) sobre las 18 texturas reales de los bloques de mena de Minecraft (16x16 RGB), y generación de un bloque nuevo desde el espacio latente.

Es un proyecto Cargo independiente (no es miembro de ningún workspace): se compila y se corre desde este directorio, `TP5`. Las dependencias salen de `Cargo.toml`: `anyhow`, `clap` (con `derive`), `ndarray`, `rand`, `rand_chacha` y `plotters`.

## Requisitos

- Ubuntu bajo WSL (donde se verificó todo). Cargo y Python existen solo dentro de WSL; el linker de MSVC de Git Bash no sirve para compilar.
- Rust estable. Versiones de la máquina donde se verificó:
  - `cargo --version`: `cargo 1.98.1 (797e8a9bc 2026-08-05)`
  - `rustc --version`: `rustc 1.98.1 (48a229cea 2026-09-01)`
- Paquetes de Ubuntu presentes donde se verificó: `build-essential`, `pkg-config`, `libfontconfig1-dev`, `libfreetype-dev` y `fonts-dejavu-core`. El binario enlaza contra fontconfig y freetype (`plotters` usa las fuentes del sistema) y necesita una fuente TrueType sans-serif para dibujar los PNG. Los bytes de un PNG dependen de las fuentes instaladas: en otra máquina los gráficos pueden diferir byte a byte aunque todos los números sean iguales.
- Python 3 (solo biblioteca estándar) para descargar las texturas del Ej 2 y para armar la presentación; compilar no lo usa.
- Acceso a internet la primera vez que se corre el Ej 2, para descargar las texturas (ver "Dataset del Ej 2").

## Dataset del Ej 2

El Ej 2 usa las texturas reales de los 18 bloques de mena de Minecraft 1.21.4 (carbón, hierro, cobre, oro, redstone, esmeralda, lapislázuli y diamante en piedra y en pizarra profunda, más el oro y el cuarzo del Nether) y las 3 rocas base (piedra, pizarra profunda y netherrack). Las texturas son propiedad de Mojang y no se versionan en este repositorio:

```bash
bash scripts/descargar_menas.sh
```

baja el cliente oficial de Minecraft 1.21.4 desde los servidores de Mojang (unos 28 MB, el mismo archivo que baja el launcher), verifica su SHA-1, extrae las 21 texturas a `datos/menas` (en `.gitignore`) y verifica el SHA-256 de cada una, así que el dataset es idéntico en cualquier máquina. Si las texturas ya están y sus hashes coinciden, no descarga nada. `bash scripts/regenerar.sh 2` lo corre solo; la variable de entorno `TP5_MENAS` cambia el directorio de las texturas (el script y el binario la respetan).

## Compilar desde un checkout limpio

```bash
git clone https://github.com/ldicandia/SIA-G3.git
cd SIA-G3/TP5
cargo build --release --locked
```

`--locked` obliga a usar las versiones de dependencias fijadas en el `Cargo.lock` versionado: la compilación falla en vez de cambiar una versión en silencio. El repositorio es el del curso (`SIA-G3`) y este proyecto vive en su subdirectorio `TP5`.

## Reproducir cada ejercicio

Un comando por ejercicio, desde el directorio `TP5`:

| Ejercicio | Comando | Subcomandos que corre | Carpeta | Archivos que escribe |
|-----------|---------|-----------------------|---------|----------------------|
| 1a | `bash scripts/regenerar.sh 1a` | `sweep`, `ej1a` | `output/ej1a` | `comparison.md`, `sweep.csv`, `curves_arch_loss.png`, `curves_arch_wrongpx.png`, `curves_optim_loss.png`, `curves_optim_wrongpx.png`, `curves_act_loss.png`, `curves_act_wrongpx.png`, `curves_loss_loss.png`, `curves_loss_wrongpx.png`, `report.csv`, `latent.png`, `latent_codes.csv`, `new_letter.png`, `salida-sweep.txt`, `salida-ej1a.txt` (16 archivos) |
| 1b | `bash scripts/regenerar.sh 1b` | `dae-sweep`, `ej1b` | `output/ej1b` | `dae_sweep.csv`, `dae_comparison.md`, `dae_curves_lat_loss.png`, `dae_curves_hid_loss.png`, `dae_curves_noise_loss.png`, `dae_levels.csv`, `dae_error_vs_noise.png`, `dae_within_vs_noise.png`, `noise_examples.png`, `dae_examples_p0.00.png`, `dae_examples_p0.05.png`, `dae_examples_p0.10.png`, `dae_examples_p0.15.png`, `dae_examples_p0.20.png`, `dae_examples_p0.30.png`, `dae_examples_p0.40.png`, `dae_examples_p0.50.png`, `salida-dae-sweep.txt`, `salida-ej1b.txt` (19 archivos) |
| 2 | `bash scripts/regenerar.sh 2` | `ej2` | `output/ej2` | `ores.png`, `vae_loss.csv`, `vae_loss.png`, `vae_reconstruction.png`, `vae_manifold.png`, `vae_new_ore.png`, `plain_manifold.png`, `salida-ej2.txt` (8 archivos) |
| Los tres | `bash scripts/regenerar.sh todo` | todos los anteriores, en el orden 1a, 1b, 2 | `output/` | los 43 archivos de arriba |

Detalles del comando:

- Para el ejercicio 2, primero corre `bash scripts/descargar_menas.sh` (baja o verifica las texturas).
- Compila una sola vez (`cargo build --release --locked`) y corre los subcomandos con las grillas completas por defecto (la opción `--quick` de `sweep` y `dae-sweep` escribe otras tablas en los mismos archivos y no se usa).
- Antes de cada ejercicio borra únicamente los nombres exactos de archivo de la lista de su fila (nunca una carpeta ni un comodín), de modo que la presencia de cada archivo al final prueba que la escribió esta corrida y no código anterior.
- Guarda el reporte impreso de cada subcomando en `salida-SUBCOMANDO.txt`, dentro de la carpeta del ejercicio.
- Termina con código distinto de cero si un subcomando falla (por ejemplo, si el `ej2` no cumple su aceptación) o si falta o está vacío algún archivo esperado. Al final imprime `REPRODUCCION OK: ejercicio X (N archivos en DIR, S s)` y, para `todo`, `REPRODUCCION COMPLETA: 3 ejercicios en RAIZ (S s)`.
- `--salida RAIZ` cambia la raíz de salida (por defecto `output`; una ruta relativa se toma desde `TP5`): `bash scripts/regenerar.sh todo --salida otra-carpeta` escribe en `otra-carpeta/ej1a`, `otra-carpeta/ej1b` y `otra-carpeta/ej2`.
- La carpeta `output/` está en `.gitignore`: los resultados se regeneran, no se versionan.

Duraciones medidas por el propio script en la máquina de verificación (WSL, perfil release, ya compilado): ejercicio 1a, 10 s; ejercicio 1b, 19 s; ejercicio 2, 11 s; `todo`, 40 s. Son tiempos de esa máquina y varían con la cantidad de núcleos (los barridos `sweep` y `dae-sweep` usan hilos).

## Subcomandos

Cada ejercicio también se puede correr por partes con `cargo run --release -- SUBCOMANDO`. Todos menos `show` y `smoke` aceptan `--out DIR` (directorio de salida) y `cargo run --release -- SUBCOMANDO --help` muestra las opciones de cada uno. El texto es el de `cargo run --release -- --help`.

| Subcomando | Forma | Qué hace |
|------------|-------|----------|
| `show` | `cargo run --release -- show` | Imprime los 32 patrones de font.h como grillas de 7x5 |
| `smoke` | `cargo run --release -- smoke` | Entrena de punta a punta un autoencoder 35-16-2-16-35 y muestra el reporte por patron |
| `sweep` | `cargo run --release -- sweep` | Barrido de arquitecturas, optimizadores, activaciones y perdidas con CSV y comparacion |
| `ej1a` | `cargo run --release -- ej1a` | Entrena la configuracion elegida del Ej 1a y muestra el reporte y el analisis de subconjunto |
| `ej1b` | `cargo run --release -- ej1b` | Entrena el Denoising Autoencoder (entrada ruidosa, objetivo limpio) y mide el ruido que elimina |
| `dae-sweep` | `cargo run --release -- dae-sweep` | Barrido de arquitecturas y calendarios de ruido del DAE (tamano latente, capacidad, ruido de entrenamiento) |
| `noise` | `cargo run --release -- noise` | Muestra letras distorsionadas a cada nivel de ruido (texto y PNG), sin entrenar nada |
| `latent` | `cargo run --release -- latent` | Grafica los 32 patrones en el plano latente de 2 dimensiones (PNG y CSV) |
| `generate` | `cargo run --release -- generate` | Decodifica un punto del plano latente que no es un patron de entrenamiento (letra nueva) |
| `ores` | `cargo run --release -- ores` | Lista los 18 bloques de mena de Minecraft del dataset y escribe su figura (necesita las texturas descargadas) |
| `vae` | `cargo run --release -- vae` | Entrena el autoencoder variacional (VAE) sobre los bloques y decodifica una muestra del prior |
| `vae-sample` | `cargo run --release -- vae-sample` | Muestrea el prior N(0, I) del VAE y busca un bloque nuevo que se juzga perteneciente al dataset |
| `vae-manifold` | `cargo run --release -- vae-manifold` | Decodifica una grilla del plano latente del VAE y mide cuanto es plausible y cuan gradual es |
| `vae-baseline` | `cargo run --release -- vae-baseline` | Contrasta el VAE con un autoencoder simple igual pero sin esquema variacional y verifica la aceptacion |
| `ej2` | `cargo run --release -- ej2` | Corre el Ej 2 completo: dataset, VAE, evaluacion, bloque nuevo, contraste y aceptacion |

Sin `--out`, cada subcomando escribe en `output/ej1a` (`sweep`, `ej1a`, `latent`, `generate`), `output/ej1b` (`ej1b`, `dae-sweep`, `noise`) o `output/ej2` (`ores`, `vae`, `vae-sample`, `vae-manifold`, `vae-baseline`, `ej2`).

## Pruebas

```bash
cargo test --release
```

Corre todas las suites del proyecto: derivadas numéricas de las capas, entrenamiento, barridos, figuras y las aceptaciones de cada ejercicio. Tres familias de pruebas de consistencia comparan los documentos de análisis con el código y fallan si difieren: `tests/doc_consistency.rs` (Ej 1a), `tests/dae_doc_consistency.rs` (Ej 1b) y `tests/vae_doc_consistency.rs` (Ej 2) recalculan cada línea y cifra citada. Además, `tests/readme_consistency.rs` ata este README y `scripts/regenerar.sh` a los nombres de archivo que escribe el código y a los subcomandos de la línea de comandos.

## Documentos de análisis

- `docs/ej1a-analisis.md`: Ej 1a. Arquitecturas y técnicas probadas, comparación de los barridos, configuración elegida y su justificación, subconjunto aprendido, espacio latente y letra nueva.
- `docs/ej1b-analisis.md`: Ej 1b. Arquitectura y ruido de entrenamiento elegidos, estudio de variantes, niveles de ruido, ejemplos visuales y nivel máximo de ruido eliminado.
- `docs/ej2-analisis.md`: Ej 2. Dataset de bloques de mena, arquitectura y esquema variacional, evolución de la pérdida, reconstrucción y espacio latente, bloque nuevo y limitaciones.

## Verificar el checkout limpio

```bash
bash scripts/regenerar.sh todo
bash scripts/verificar_checkout_limpio.sh todo --comparar output
```

El segundo comando copia a un directorio temporal solo los archivos versionados (más los nuevos no ignorados; sin `target/`, `output/`, `.planning/` ni `datos/`), compila esa copia como proyecto independiente con `cargo build --release --locked`, corre allí `regenerar.sh todo` y compara cada archivo regenerado con el de `output`: los CSV, Markdown y PNG deben tener el mismo SHA-256 y los reportes `salida-*.txt` deben ser iguales una vez reemplazada la raíz de salida y sin las líneas con tiempos. Imprime `CHECKOUT LIMPIO OK: todo (N archivos copiados, M comparados)` si todo coincide. Prueba que el checkout contiene todo lo que hace falta y que el resultado no depende de archivos sobrantes del árbol de trabajo. La única excepción son las texturas del Ej 2, que no se versionan: si `datos/menas` existe, la copia las usa desde ahí con `TP5_MENAS` (verificando sus hashes); si no, las descarga. Con `--conservar` deja el directorio temporal para inspeccionarlo.

## Presentación

La presentación es un archivo PowerPoint, `presentacion/TP5-autoencoders.pptx`, versionado en git. Tiene una portada, un Resumen, una sección por ejercicio (Ej 1a, Ej 1b y Ej 2, cada una con arquitectura, técnicas probadas, resultados con sus gráficos y conclusiones) y una diapositiva final de Reproducibilidad. Cada diapositiva lleva notas del orador (visibles en la vista del presentador de PowerPoint) con la explicación hablada, la lectura de las figuras, las preguntas probables y las líneas de reporte citadas tras un párrafo `Fuente: EJ/ARCHIVO`.

```bash
bash scripts/presentacion.sh
```

El comando hace, en orden: regenera los tres ejercicios (`bash scripts/regenerar.sh todo`), construye la presentación a partir de lo regenerado (`presentacion/construir.py`) y verifica que sea consistente con ello (`presentacion/verificar.py`). Termina con `PRESENTACION OK: ARCHIVO (S s)` o con un mensaje en español que nombra el paso que falló. Opciones:

- `--salida RAIZ`: raíz de las salidas regeneradas (por defecto `output`).
- `--destino ARCHIVO`: dónde escribe la presentación (por defecto `presentacion/TP5-autoencoders.pptx`).
- `--sin-regenerar`: no corre `regenerar.sh` y arma la presentación con las salidas que ya hay en `RAIZ`; si faltan, termina con error y nombra `bash scripts/regenerar.sh todo`.

Hace falta solo Python 3 y su biblioteca estándar: no hay nada que instalar y no hace falta PowerPoint para construirla. El archivo se escribe sin comprimir y con fechas fijas, así que dos construcciones a partir de las mismas salidas son idénticas byte a byte.

Tres reglas de diseño y qué verifica cada herramienta:

- **Presupuesto de texto y notas del orador.** Cada diapositiva tiene como máximo 40 palabras fuera del título y de las celdas de las tablas, un título de una sola línea (a lo sumo 60 caracteres), cuerpo de al menos 18 pt y celdas de tabla de al menos 16 pt; a lo sumo 2 tablas por diapositiva (hasta 8 filas de datos y 5 columnas) y ninguna imagen en una diapositiva con tabla. En la diapositiva queda solo lo comprobable contra `output/` o contra la figura de esa misma diapositiva; la interpretación pasa a las notas del orador (al menos 30 palabras de prosa en cada diapositiva salvo la portada).
- **Legibilidad y ubicación de las figuras.** Los gráficos de curvas y del espacio latente no tienen grilla (`disable_mesh()`) y las figuras de letras de 7x5 llevan un único recuadro exterior por patrón sin grilla por celda. Al generar cada PNG, `src/plots.rs` exige que el texto más chico de la figura equivalga a al menos 16 pt en la escala del área de contenido de la diapositiva (`12.333 x 5.35` pulgadas); en la diapositiva cada figura ocupa al menos el 90 % del ancho o del alto del área de contenido (a lo sumo 2 imágenes por diapositiva).
- **Cobertura de la consigna y trazabilidad de números.** La tabla `CONSIGNA` de `presentacion/verificar.py` asocia cada ítem de `docs/SIATP5.pdf` (`1.a.1`, `1.a.2`, `1.a.3`, `1.a.4`, `1.b.1`, `1.b.2`, `2.a`, `2.b`, `2.c`) con su diapositiva y su evidencia obligatoria (figuras o estudios en tablas). Las diapositivas y las notas no tipean cifras: las leen de `output/` mediante `presentacion/lector.py`.
- `presentacion/verificar.py --pptx ARCHIVO --salida RAIZ` comprueba la estructura del paquete y la consistencia con `RAIZ`: secciones y orden, presupuesto de texto y tamaños de letra, notas del orador en cada diapositiva (también se listan con `--notas`), números en diapositivas y en notas contra las salidas y los documentos de análisis, imágenes por SHA-256, texto alternativo y tamaño en la diapositiva, líneas citadas en bloques `salida:` y tras `Fuente:` en las notas, subsecuencias en orden de las filas de las tablas de comparación (o la figura de curvas del estudio), cobertura de la consigna y ausencia de las afirmaciones retiradas. Como compara los bytes de los PNG, que dependen de las fuentes del sistema, solo vale en la máquina que los regeneró.
- `cargo test --release --test deck_consistency` comprueba la presentación contra lo que la biblioteca recalcula en el propio proceso (en diapositivas y en notas: líneas citadas, subsecuencias de tablas o figuras de estudio, cifras, coordenadas y ausencia de afirmaciones retiradas), sin leer `output/` ni comparar bytes de PNG: pasa en cualquier máquina. Por defecto lee `presentacion/TP5-autoencoders.pptx`; si se define la variable de entorno `TP5_DECK`, lee esa ruta (por ejemplo `TP5_DECK=output/presentacion/TP5-autoencoders.pptx` cuando el archivo versionado está abierto en PowerPoint).

Pruebas del propio armado, sin dependencia de `output/`: `python3 -B -m unittest discover -s presentacion/pruebas`.

Organización: cada sección es un módulo `presentacion/secciones/seccion_NN_nombre.py` con una función `agregar(pres, salidas)`; `construir.py` las toma en orden de nombre y `presentacion/disposicion.py` concentra las ayudas de ubicación.

| Módulo | Diapositivas |
|--------|--------------|
| `presentacion/secciones/seccion_00_portada.py` | portada y Resumen |
| `presentacion/secciones/seccion_10_ej1a.py` | Ej 1a |
| `presentacion/secciones/seccion_20_ej1b.py` | Ej 1b |
| `presentacion/secciones/seccion_30_ej2.py` | Ej 2 |
| `presentacion/secciones/seccion_99_cierre.py` | Reproducibilidad |

Para reconstruir una sola sección (por ejemplo, mientras se edita) se usa `python3 -B presentacion/construir.py --solo seccion_30 --destino output/presentacion/parcial.pptx`; el resultado es una presentación parcial que no pasa la verificación completa.

Una presentación reconstruida en otra máquina puede diferir del archivo versionado en los bytes de los PNG (fuentes del sistema) aunque todos los números sean iguales. `bash scripts/verificar_checkout_limpio.sh todo --comparar output --presentacion` hace el mismo recorrido en la copia limpia: tras regenerar y comparar, construye y verifica allí la presentación, y avisa si el archivo resultante es idéntico o distinto del versionado.
