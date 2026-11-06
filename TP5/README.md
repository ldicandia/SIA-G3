# SIA TP5: Deep Learning (autoencoders)

## Qué es

TP5 de Sistemas de Inteligencia Artificial (ITBA). Implementa en Rust, sobre `ndarray`, con backpropagation y optimizadores propios (SGD, Momentum y Adam) y sin frameworks de ML, los tres ejercicios de la consigna:

- **Ej 1a**: autoencoder básico con espacio latente de 2 dimensiones sobre los 32 patrones de 7x5 de `docs/font.h`, con un máximo de 1 pixel incorrecto por patrón.
- **Ej 1b**: autoencoder denoising (entrada distorsionada, salida esperada limpia) y análisis del nivel de ruido que logra eliminar.
- **Ej 2**: autoencoder variacional (VAE) sobre un dataset de emojis de 24x24 construido por código, y generación de un emoji nuevo desde el espacio latente.

Es un proyecto Cargo independiente (no es miembro de ningún workspace): se compila y se corre desde este directorio, `TP5`. Las dependencias salen de `Cargo.toml`: `anyhow`, `clap` (con `derive`), `ndarray`, `rand`, `rand_chacha` y `plotters`.

## Requisitos

- Ubuntu bajo WSL (donde se verificó todo). Cargo y Python existen solo dentro de WSL; el linker de MSVC de Git Bash no sirve para compilar.
- Rust estable. Versiones de la máquina donde se verificó:
  - `cargo --version`: `cargo 1.98.1 (797e8a9bc 2026-08-05)`
  - `rustc --version`: `rustc 1.98.1 (48a229cea 2026-09-01)`
- Paquetes de Ubuntu presentes donde se verificó: `build-essential`, `pkg-config`, `libfontconfig1-dev`, `libfreetype-dev` y `fonts-dejavu-core`. El binario enlaza contra fontconfig y freetype (`plotters` usa las fuentes del sistema) y necesita una fuente TrueType sans-serif para dibujar los PNG. Los bytes de un PNG dependen de las fuentes instaladas: en otra máquina los gráficos pueden diferir byte a byte aunque todos los números sean iguales.
- Python 3 solo hace falta para armar la presentación; compilar y regenerar los resultados no lo usa.

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
| 2 | `bash scripts/regenerar.sh 2` | `ej2` | `output/ej2` | `emojis.png`, `vae_loss.csv`, `vae_loss.png`, `vae_reconstruction.png`, `vae_manifold.png`, `vae_new_emoji.png`, `plain_manifold.png`, `salida-ej2.txt` (8 archivos) |
| Los tres | `bash scripts/regenerar.sh todo` | todos los anteriores, en el orden 1a, 1b, 2 | `output/` | los 43 archivos de arriba |

Detalles del comando:

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
| `emojis` | `cargo run --release -- emojis` | Muestra el dataset de 36 emojis de 24x24 construido por codigo (texto) |
| `vae` | `cargo run --release -- vae` | Entrena el autoencoder variacional (VAE) sobre los emojis y decodifica una muestra del prior |
| `vae-sample` | `cargo run --release -- vae-sample` | Muestrea el prior N(0, I) del VAE y busca un emoji nuevo que se juzga perteneciente al dataset |
| `vae-manifold` | `cargo run --release -- vae-manifold` | Decodifica una grilla del plano latente del VAE y mide cuanto es reconocible y cuan gradual es |
| `vae-baseline` | `cargo run --release -- vae-baseline` | Contrasta el VAE con un autoencoder simple igual pero sin esquema variacional y verifica la aceptacion |
| `ej2` | `cargo run --release -- ej2` | Corre el Ej 2 completo: dataset, VAE, evaluacion, emoji nuevo, contraste y aceptacion |

Sin `--out`, cada subcomando escribe en `output/ej1a` (`sweep`, `ej1a`, `latent`, `generate`), `output/ej1b` (`ej1b`, `dae-sweep`, `noise`) o `output/ej2` (`emojis`, `vae`, `vae-sample`, `vae-manifold`, `vae-baseline`, `ej2`).

## Pruebas

```bash
cargo test --release
```

Corre todas las suites del proyecto: derivadas numéricas de las capas, entrenamiento, barridos, figuras y las aceptaciones de cada ejercicio. Tres familias de pruebas de consistencia comparan los documentos de análisis con el código y fallan si difieren: `tests/doc_consistency.rs` (Ej 1a), `tests/dae_doc_consistency.rs` (Ej 1b) y `tests/vae_doc_consistency.rs` (Ej 2) recalculan cada línea y cifra citada. Además, `tests/readme_consistency.rs` ata este README y `scripts/regenerar.sh` a los nombres de archivo que escribe el código y a los subcomandos de la línea de comandos.

## Documentos de análisis

- `docs/ej1a-analisis.md`: Ej 1a. Arquitecturas y técnicas probadas, comparación de los barridos, configuración elegida y su justificación, subconjunto aprendido, espacio latente y letra nueva.
- `docs/ej1b-analisis.md`: Ej 1b. Arquitectura y ruido de entrenamiento elegidos, estudio de variantes, niveles de ruido, ejemplos visuales y nivel máximo de ruido eliminado.
- `docs/ej2-analisis.md`: Ej 2. Dataset de emojis, arquitectura y esquema variacional, evolución de la pérdida, reconstrucción y espacio latente, emoji nuevo y limitaciones.

## Verificar el checkout limpio

```bash
bash scripts/regenerar.sh todo
bash scripts/verificar_checkout_limpio.sh todo --comparar output
```

El segundo comando copia a un directorio temporal solo los archivos versionados (más los nuevos no ignorados; sin `target/`, `output/` ni `.planning/`), compila esa copia como proyecto independiente con `cargo build --release --locked`, corre allí `regenerar.sh todo` y compara cada archivo regenerado con el de `output`: los CSV, Markdown y PNG deben tener el mismo SHA-256 y los reportes `salida-*.txt` deben ser iguales una vez reemplazada la raíz de salida y sin las líneas con tiempos. Imprime `CHECKOUT LIMPIO OK: todo (N archivos copiados, M comparados)` si todo coincide. Prueba que el checkout contiene todo lo que hace falta y que el resultado no depende de archivos sobrantes del árbol de trabajo. Con `--conservar` deja el directorio temporal para inspeccionarlo.
