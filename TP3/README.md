# SIA TP3 — Perceptrón Simple y Multicapa (Hito de Validación)

Motor de redes neuronales artificiales programado a mano desde cero en **C++17** (con álgebra matricial y backpropagation propios, sin librerías externas), acompañado de un frontend en **Python 3** para la ejecución de pruebas end-to-end y generación de gráficos de convergencia y ajuste.

Este hito cubre exhaustivamente el **Ejercicio (Validación)** recomendado por la cátedra para el TP3 de Sistemas de Inteligencia Artificial (ITBA, 2Q 2026):
1. **Perceptrón Escalonado**: Clasificación de compuerta **AND**.
2. **Perceptrón Lineal**: Ajuste de regresión sobre $y = x$.
3. **Perceptrón No Lineal**: Ajuste de regresión sobre $y = \tanh(x)$.
4. **Perceptrón Multicapa (MLP)**: Resolución del problema no linealmente separable **XOR** con arquitecturas `[2, 2, 1]` y `[2, 3, 2, 1]`, demostrando simultáneamente la imposibilidad del perceptrón simple de resolverlo.
5. **Cálculos a Mano**: Desarrollo analítico paso a paso de propagación hacia adelante y backpropagation para `[2, 2, 1]` y `[2, 3, 2, 1]` contrastados contra tests unitarios en C++.

---

## 1. Entorno y Requisitos Previos

> [!IMPORTANT]
> **Todo se ejecuta dentro de WSL (Ubuntu 24.04).**  
> Debido a inconsistencias en entornos virtuales de Windows para este monorepo (como se experimentó en TP1 y TP2), tanto la compilación C++ como el entorno virtual de Python deben ejecutarse íntegramente en WSL.

### 1.1 Herramientas Requeridas
En Ubuntu 24.04, `g++ 13` y `make` ya vienen instalados. Solo se requiere instalar `cmake` y los paquetes de Python:
```bash
sudo apt update
sudo apt install -y cmake python3 python3-venv python3-pip
```

---

## 2. Puesta en Marcha Rápida (One-Liner)

Para clonar, compilar, verificar ambos suites de pruebas, ejecutar los cuatro validadores y generar todos los gráficos en un solo comando dentro de `TP3/`:

```bash
# Desde el directorio TP3/ en WSL:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && \
cmake -S . -B build && cmake --build build && \
ctest --test-dir build --output-on-failure && \
.venv/bin/python -m pytest && \
./build/tp3 validate and --seed 42 --out runs_validation && \
./build/tp3 validate linear --seed 42 --out runs_validation && \
./build/tp3 validate tanh --seed 42 --out runs_validation && \
./build/tp3 validate xor --seed 42 --out runs_validation && \
.venv/bin/python scripts/validation/plot_validation.py --in runs_validation --out plots
```

---

## 3. Configuración y Compilación Paso a Paso

### 3.1 Entorno Virtual de Python
Cree el entorno virtual e instale las dependencias de análisis y pruebas (`matplotlib`, `pytest`):
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3.2 Compilación del Motor en C++
Configure y compile el proyecto con CMake (estándar C++17, optimización Release):
```bash
cmake -S . -B build
cmake --build build
```
Esto genera dos binarios ejecutables en la carpeta `build/`:
- `build/tp3`: CLI principal del motor.
- `build/tp3_tests`: Runner de tests unitarios (doctest).

---

## 4. Ejecución de Tests

El proyecto cuenta con dos suites de pruebas desacopladas que garantizan la integridad matemática y el comportamiento end-to-end:

### 4.1 Tests Unitarios en C++ (`ctest`)
Ejecuta los tests unitarios integrados que cubren el álgebra de matrices, las funciones de activación y sus derivadas, la actualización del perceptrón simple, el paso de backpropagation del MLP y la reproducción exacta de los cálculos a mano:
```bash
ctest --test-dir build --output-on-failure
```
O directamente con el binario de doctest para un reporte detallado:
```bash
./build/tp3_tests
```
*(32 casos de prueba, 168 aserciones pasan con 0 fallos).*

### 4.2 Tests End-to-End en Python (`pytest`)
Ejecuta la suite que invoca el binario compilado `build/tp3`, valida las salidas JSON, comprueba la convergencia de cada problema, la reproducibilidad determinística ante la misma semilla y la generación de gráficos:
```bash
.venv/bin/python -m pytest -v
```
*(30 tests pasan en ~20 segundos).*

---

## 5. Ejecución de los Validadores (`tp3 validate`)

El CLI `tp3 validate <caso>` permite ejecutar individualmente cada uno de los cuatro problemas de validación:

### 5.1 Compuerta AND (Perceptrón Escalonado)
```bash
./build/tp3 validate and --seed 42 --out runs_validation
```
- **Salida esperada en consola**: Pesos finales, sesgo, error final 0, tabla con los 4 patrones clasificados con exactitud y `correct: 4/4`.
- **Archivo generado**: `runs_validation/and.json`.

### 5.2 Regresión Lineal $y = x$ (Perceptrón Lineal)
```bash
./build/tp3 validate linear --seed 42 --out runs_validation
```
- **Salida esperada en consola**: 50 muestras uniformes en $[-2, 2]$, peso convergente a $\approx 1.0$, sesgo convergente a $\approx 0.0$ y error cuadrático medio (MSE) $< 10^{-3}$.
- **Archivo generado**: `runs_validation/linear.json`.

### 5.3 Regresión No Lineal $y = \tanh(x)$ (Perceptrón No Lineal)
```bash
./build/tp3 validate tanh --seed 42 --out runs_validation
```
- **Salida esperada en consola**: 50 muestras uniformes en $[-2, 2]$, peso convergente a $\approx 1.0$, sesgo convergente a $\approx 0.0$ y MSE $< 10^{-2}$.
- **Archivo generado**: `runs_validation/tanh.json`.

### 5.4 Compuerta XOR (MLP vs Perceptrón Simple)
```bash
./build/tp3 validate xor --seed 42 --out runs_validation
```
- **Salida esperada en consola**:
  1. **MLP `[2, 2, 1]`**: Converge a MSE $< 0.001$, clasificación `correct: 4/4`.
  2. **MLP `[2, 3, 2, 1]`**: Converge a MSE $< 0.001$, clasificación `correct: 4/4`.
  3. **Perceptrón Escalonado**: Clasificación `correct: 1/4 (step perceptron cannot solve non-linearly separable XOR)`.
- **Archivos generados**:
  - `runs_validation/xor_221.json`
  - `runs_validation/xor_2321.json`
  - `runs_validation/xor_step.json`
  - `runs_validation/xor.json` (copia de compatibilidad).

#### Opciones Generales del CLI
- `--seed N`: Semilla pseudoaleatoria (default: `42`).
- `--epochs N`: Cantidad de épocas de entrenamiento (sobrescribe el default de cada caso).
- `--lr F`: Tasa de aprendizaje $\eta$ (sobrescribe el default de cada caso).
- `--out DIR`: Directorio destino para los archivos JSON (default: `runs_validation`).

---

## 6. Generación de Gráficos (`plot_validation.py`)

Para graficar las curvas de pérdida y los ajustes generados por las corridas de validación:
```bash
.venv/bin/python scripts/validation/plot_validation.py --in runs_validation --out plots
```

Los gráficos se generan en el directorio `plots/`:
- `plots/and_loss.png`: Curva de error de la compuerta AND.
- `plots/linear_loss.png`: Curva de MSE para la regresión lineal.
- `plots/linear_fit.png`: Gráfico de dispersión de las 50 muestras vs recta ajustada por el perceptrón.
- `plots/tanh_loss.png`: Curva de MSE para la regresión tangente hiperbólica.
- `plots/tanh_fit.png`: Gráfico de dispersión de las 50 muestras vs curva ajustada en forma de S.
- `plots/xor_loss.png`: **Gráfico comparativo multi-curva** en los mismos ejes que muestra la rápida convergencia de los MLPs `[2, 2, 1]` y `[2, 3, 2, 1]` junto con el estancamiento del perceptrón simple escalonado.
- `plots/xor_221_loss.png`, `plots/xor_2321_loss.png`, `plots/xor_step_loss.png`: Curvas individuales de las arquitecturas en XOR.

---

## 7. Cálculos a Mano de XOR (`docs/xor_a_mano.md`)

En cumplimiento de los requisitos **DOC-01**, **DOC-02** y **DOC-03**:
- El documento [`docs/xor_a_mano.md`](docs/xor_a_mano.md) desarrolla analíticamente con fórmulas completas y números de alta precisión (8 decimales) un paso completo hacia adelante (forward) y hacia atrás (backpropagation) con el patrón $x = [-1.0, 1.0], y = 1.0, \eta = 0.1$ para:
  - **Arquitectura `[2, 2, 1]`**
  - **Arquitectura `[2, 3, 2, 1]`**
- El test C++ en `core/tests/test_mlp.cpp` carga exactamente los pesos iniciales del documento, ejecuta una época y valida que los pesos y sesgos resultantes coinciden con el cálculo manual dentro de una tolerancia estricta de $10^{-6}$.

---

## 8. Estructura del Repositorio

```
TP3/
├── CMakeLists.txt              # Definición de compilación C++17 (tp3_core, tp3, tp3_tests)
├── README.md                   # Esta guía de entrega y ejecución
├── requirements.txt            # Dependencias Python (matplotlib, pytest)
├── pytest.ini                  # Configuración de pytest (pythonpath = .)
├── core/                       # Código fuente del motor en C++17
│   ├── activations.hpp/.cpp    # Funciones de activación (step, identity, tanh, sigmoid) y derivadas
│   ├── matrix.hpp/.cpp         # Álgebra matricial propia (+, -, *, transpose, hadamard, apply)
│   ├── model.hpp               # Interfaz abstracta Model (fit, predict, flat_weights)
│   ├── perceptron.hpp/.cpp     # Perceptrón Simple (escalón, lineal, no lineal)
│   ├── mlp.hpp/.cpp            # Perceptrón Multicapa con Backpropagation sobre Matrix
│   ├── training.hpp/.cpp       # Métricas de entrenamiento (MSE)
│   ├── io/run_json.hpp/.cpp    # Serializador atómico de ejecuciones a JSON
│   ├── validation/datasets.*   # Generadores determinísticos de datasets (AND, lineal, tanh, XOR)
│   ├── main.cpp                # CLI ejecutable (`tp3 validate`)
│   └── tests/                  # Suite de tests unitarios C++ (doctest)
│       ├── test_main.cpp
│       ├── test_matrix.cpp
│       ├── test_activations.cpp
│       ├── test_perceptron.cpp
│       ├── test_mlp.cpp        # Tests de MLP y reproducción DOC-03 de xor_a_mano.md
│       ├── test_smoke_and.cpp
│       └── test_run_json.cpp
├── docs/
│   ├── Enunciado TP3 - 2Q 2026.pdf
│   └── xor_a_mano.md           # Cálculo a mano detallado para [2,2,1] y [2,3,2,1]
├── scripts/
│   ├── validation/             # Scripts para corridas de validación
│   │   └── plot_validation.py  # Graficador matplotlib a partir de los JSON generados
│   ├── ejercicio1/             # Scripts para el Ejercicio 1 (Fraud TinyModel)
│   │   ├── fraud_explore.py
│   │   ├── fraud_preprocess.py
│   │   ├── fraud_metrics.py
│   │   ├── fraud_train_variant.py
│   │   ├── fraud_sweep.py
│   │   ├── fraud_generalization.py
│   │   └── fraud_compare.py
│   └── ejercicio2/             # Scripts para el Ejercicio 2 (Digits MLP)
│       ├── digits_preprocess.py
│       ├── digits_metrics.py
│       ├── digits_train_variant.py
│       ├── digits_lr_sweep.py
│       ├── digits_architecture_sweep.py
│       ├── digits_optimizer_sweep.py
│       └── digits_compare.py
├── tests/                      # Suite de tests end-to-end en Python (pytest)
│   ├── conftest.py             # Fixtures para invocar ./build/tp3 y parsear JSON
│   ├── test_boundary.py        # Validación de aislamiento arquitectónico C++ / Python
│   ├── test_plot_validation.py # Verificación de generación de PNGs
│   ├── test_reproducibility.py # Verificación de determinismo por semilla
│   ├── test_validate_and.py    # Test end-to-end para AND
│   ├── test_validate_linear.py # Test end-to-end para lineal
│   ├── test_validate_tanh.py   # Test end-to-end para tanh
│   └── test_validate_xor.py    # Test end-to-end para XOR (MLP 4/4 vs Step falla)
└── third_party/
    └── doctest.h               # Framework de tests C++ header-only vendoreado (MIT)
```
