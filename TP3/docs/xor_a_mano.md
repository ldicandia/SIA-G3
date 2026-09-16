# Cálculo a Mano de Propagación hacia Adelante y Backpropagation para XOR

Este documento presenta el cálculo analítico paso a paso de una iteración (un forward pass y un backward pass) de entrenamiento con **Backpropagation** sobre el problema de **XOR bipolar**, para dos arquitecturas de Redes Neuronales Multicapa (MLP):
1. **Arquitectura `[2, 2, 1]`** (2 entradas, 1 capa oculta de 2 neuronas, 1 salida).
2. **Arquitectura `[2, 3, 2, 1]`** (2 entradas, 2 capas ocultas de 3 y 2 neuronas, 1 salida).

Ambos cálculos corresponden exactamente al algoritmo implementado en `core/mlp.cpp` y están verificados automáticamente por el test unitario C++ `core/tests/test_mlp.cpp` (requisito **DOC-03**).

---

## 1. Convenciones Matemáticas y Notación

### 1.1 Datos del Ejemplo
- Entrada: $x = \begin{pmatrix} -1.0 & 1.0 \end{pmatrix}$ ($1 \times 2$)
- Salida esperada (XOR bipolar): $y = 1.0$ ($1 \times 1$)
- Tasa de aprendizaje: $\eta = 0.1$
- Función de activación: Tangente hiperbólica para todas las capas:
  $$f(h) = \tanh(h) = \frac{e^h - e^{-h}}{e^h + e^{-h}}$$
  $$f'(h) = 1 - \tanh^2(h) = 1 - f(h)^2$$

### 1.2 Paso hacia Adelante (Forward Pass)
Para cada capa $l = 0, \dots, L-2$:
- Entrada a la capa: $A_l$ (para $l=0$, $A_0 = x$)
- Potencial de activación (pre-activación):
  $$H_l = A_l W_l + b_l$$
  donde $W_l$ tiene dimensión $n_l \times n_{l+1}$ y $b_l$ tiene dimensión $1 \times n_{l+1}$.
- Activación de salida de la capa:
  $$A_{l+1} = f(H_l) = \tanh(H_l)$$

### 1.3 Paso hacia Atrás (Backward Pass / Backpropagation)
Función de costo para el patrón individual: $E = \frac{1}{2} (y - A_L)^2$.
- Error en la capa de salida ($l = L-2$):
  $$err = y - A_L$$
  $$\delta_{out} = err \circ f'(H_{L-2}) = (y - A_L) \circ (1 - A_L^2)$$
- Gradientes locales en capas ocultas (propagación hacia atrás para $l$ desde la penúltima hasta 0):
  $$\delta_l = (\delta_{l+1} W_{l+1}^T) \circ f'(H_l) = (\delta_{l+1} W_{l+1}^T) \circ (1 - A_{l+1}^2)$$
- Actualización de parámetros (Gradiente Descendente Online):
  $$W_l^{new} = W_l + \Delta W_l = W_l + \eta \cdot (A_l^T \delta_l)$$
  $$b_l^{new} = b_l + \Delta b_l = b_l + \eta \cdot \delta_l$$

---

## 2. Arquitectura `[2, 2, 1]`

### 2.1 Pesos y Sesgos Iniciales
- **Capa 0 (Oculta, $2 \to 2$):**
  $$W_0 = \begin{pmatrix} 0.15 & -0.25 \\ 0.35 & 0.45 \end{pmatrix}, \quad b_0 = \begin{pmatrix} 0.10 & -0.20 \end{pmatrix}$$
- **Capa 1 (Salida, $2 \to 1$):**
  $$W_1 = \begin{pmatrix} 0.50 \\ -0.40 \end{pmatrix}, \quad b_1 = \begin{pmatrix} 0.05 \end{pmatrix}$$

### 2.2 Forward Pass
1. **Capa 0 $\to$ Capa 1:**
   $$H_0 = x W_0 + b_0 = \begin{pmatrix} -1.0 & 1.0 \end{pmatrix} \begin{pmatrix} 0.15 & -0.25 \\ 0.35 & 0.45 \end{pmatrix} + \begin{pmatrix} 0.10 & -0.20 \end{pmatrix}$$
   $$H_0 = \begin{pmatrix} (-1)(0.15) + (1)(0.35) + 0.10 & (-1)(-0.25) + (1)(0.45) - 0.20 \end{pmatrix}$$
   $$H_0 = \begin{pmatrix} 0.30000000 & 0.50000000 \end{pmatrix}$$

   Activación:
   $$A_1 = \tanh(H_0) = \begin{pmatrix} \tanh(0.3) & \tanh(0.5) \end{pmatrix} = \begin{pmatrix} 0.29131261 & 0.46211716 \end{pmatrix}$$
   Derivada $f'(H_0) = 1 - A_1^2$:
   $$f'(H_0) = \begin{pmatrix} 1 - 0.29131261^2 & 1 - 0.46211716^2 \end{pmatrix} = \begin{pmatrix} 0.91513696 & 0.78644773 \end{pmatrix}$$

2. **Capa 1 $\to$ Capa 2 (Salida):**
   $$H_1 = A_1 W_1 + b_1 = \begin{pmatrix} 0.29131261 & 0.46211716 \end{pmatrix} \begin{pmatrix} 0.50 \\ -0.40 \end{pmatrix} + 0.05$$
   $$H_1 = (0.29131261)(0.50) + (0.46211716)(-0.40) + 0.05 = 0.14565631 - 0.18484686 + 0.05 = 0.01080944$$

   Activación de salida:
   $$\hat{y} = A_2 = \tanh(0.01080944) = 0.01080902$$
   Derivada en la salida $f'(H_1) = 1 - A_2^2$:
   $$f'(H_1) = 1 - 0.01080902^2 = 0.99988316$$

### 2.3 Backward Pass
1. **Error y Delta de Salida:**
   $$err = y - \hat{y} = 1.0 - 0.01080902 = 0.98919098$$
   $$\delta_1 = err \cdot f'(H_1) = 0.98919098 \times 0.99988316 = 0.98907541$$

2. **Delta de la Capa Oculta:**
   $$\delta_0 = (\delta_1 W_1^T) \circ f'(H_0)$$
   $$\delta_1 W_1^T = 0.98907541 \times \begin{pmatrix} 0.50 & -0.40 \end{pmatrix} = \begin{pmatrix} 0.49453771 & -0.39563016 \end{pmatrix}$$
   $$\delta_0 = \begin{pmatrix} 0.49453771 \times 0.91513696 & -0.39563016 \times 0.78644773 \end{pmatrix} = \begin{pmatrix} 0.45256973 & -0.31114244 \end{pmatrix}$$

3. **Gradientes y Actualización de Parámetros ($\eta = 0.1$):**
   - **Capa 1 (Salida):**
     $$\Delta W_1 = \eta (A_1^T \delta_1) = 0.1 \begin{pmatrix} 0.29131261 \\ 0.46211716 \end{pmatrix} (0.98907541) = \begin{pmatrix} 0.02881301 \\ 0.04570687 \end{pmatrix}$$
     $$\Delta b_1 = \eta \delta_1 = 0.1 \times 0.98907541 = 0.09890754$$
     Pesos y sesgos actualizados:
     $$W_1^{new} = \begin{pmatrix} 0.50 + 0.02881301 \\ -0.40 + 0.04570687 \end{pmatrix} = \begin{pmatrix} 0.52881301 \\ -0.35429313 \end{pmatrix}$$
     $$b_1^{new} = 0.05 + 0.09890754 = 0.14890754$$

   - **Capa 0 (Oculta):**
     $$\Delta W_0 = \eta (A_0^T \delta_0) = 0.1 \begin{pmatrix} -1.0 \\ 1.0 \end{pmatrix} \begin{pmatrix} 0.45256973 & -0.31114244 \end{pmatrix} = \begin{pmatrix} -0.04525697 & 0.03111424 \\ 0.04525697 & -0.03111424 \end{pmatrix}$$
     $$\Delta b_0 = \eta \delta_0 = 0.1 \begin{pmatrix} 0.45256973 & -0.31114244 \end{pmatrix} = \begin{pmatrix} 0.04525697 & -0.03111424 \end{pmatrix}$$
     Pesos y sesgos actualizados:
     $$W_0^{new} = \begin{pmatrix} 0.15 - 0.04525697 & -0.25 + 0.03111424 \\ 0.35 + 0.04525697 & 0.45 - 0.03111424 \end{pmatrix} = \begin{pmatrix} 0.10474303 & -0.21888576 \\ 0.39525697 & 0.41888576 \end{pmatrix}$$
     $$b_0^{new} = \begin{pmatrix} 0.10 + 0.04525697 & -0.20 - 0.03111424 \end{pmatrix} = \begin{pmatrix} 0.14525697 & -0.23111424 \end{pmatrix}$$

---

## 3. Arquitectura `[2, 3, 2, 1]`

### 3.1 Pesos y Sesgos Iniciales
- **Capa 0 ($2 \to 3$):**
  $$W_0 = \begin{pmatrix} 0.15 & -0.20 & 0.25 \\ 0.30 & 0.10 & -0.35 \end{pmatrix}, \quad b_0 = \begin{pmatrix} 0.05 & -0.10 & 0.15 \end{pmatrix}$$
- **Capa 1 ($3 \to 2$):**
  $$W_1 = \begin{pmatrix} 0.40 & -0.30 \\ -0.25 & 0.35 \\ 0.20 & 0.10 \end{pmatrix}, \quad b_1 = \begin{pmatrix} -0.05 & 0.05 \end{pmatrix}$$
- **Capa 2 ($2 \to 1$):**
  $$W_2 = \begin{pmatrix} 0.50 \\ -0.45 \end{pmatrix}, \quad b_2 = \begin{pmatrix} 0.10 \end{pmatrix}$$

### 3.2 Forward Pass
1. **Capa 0 $\to$ Capa 1 ($2 \to 3$):**
   $$H_0 = x W_0 + b_0 = \begin{pmatrix} -1.0 & 1.0 \end{pmatrix} \begin{pmatrix} 0.15 & -0.20 & 0.25 \\ 0.30 & 0.10 & -0.35 \end{pmatrix} + \begin{pmatrix} 0.05 & -0.10 & 0.15 \end{pmatrix}$$
   $$H_0 = \begin{pmatrix} (-0.15 + 0.30 + 0.05) & (0.20 + 0.10 - 0.10) & (-0.25 - 0.35 + 0.15) \end{pmatrix} = \begin{pmatrix} 0.20000000 & 0.20000000 & -0.45000000 \end{pmatrix}$$
   Activación:
   $$A_1 = \tanh(H_0) = \begin{pmatrix} 0.19737532 & 0.19737532 & -0.42189901 \end{pmatrix}$$
   Derivada $f'(H_0) = 1 - A_1^2$:
   $$f'(H_0) = \begin{pmatrix} 0.96104297 & 0.96104297 & 0.82200122 \end{pmatrix}$$

2. **Capa 1 $\to$ Capa 2 ($3 \to 2$):**
   $$H_1 = A_1 W_1 + b_1 = \begin{pmatrix} 0.19737532 & 0.19737532 & -0.42189901 \end{pmatrix} \begin{pmatrix} 0.40 & -0.30 \\ -0.25 & 0.35 \\ 0.20 & 0.10 \end{pmatrix} + \begin{pmatrix} -0.05 & 0.05 \end{pmatrix}$$
   - $H_1[0] = 0.19737532(0.40) + 0.19737532(-0.25) - 0.42189901(0.20) - 0.05 = -0.10477350$
   - $H_1[1] = 0.19737532(-0.30) + 0.19737532(0.35) - 0.42189901(0.10) + 0.05 = 0.01767887$
   $$H_1 = \begin{pmatrix} -0.10477350 & 0.01767887 \end{pmatrix}$$
   Activación:
   $$A_2 = \tanh(H_1) = \begin{pmatrix} -0.10439180 & 0.01767702 \end{pmatrix}$$
   Derivada $f'(H_1) = 1 - A_2^2$:
   $$f'(H_1) = \begin{pmatrix} 0.98910235 & 0.99968752 \end{pmatrix}$$

3. **Capa 2 $\to$ Capa 3 ($2 \to 1$, Salida):**
   $$H_2 = A_2 W_2 + b_2 = (-0.10439180)(0.50) + (0.01767702)(-0.45) + 0.10 = 0.03984944$$
   $$\hat{y} = A_3 = \tanh(0.03984944) = 0.03982836$$
   Derivada en la salida $f'(H_2) = 1 - A_3^2$:
   $$f'(H_2) = 1 - 0.03982836^2 = 0.99841370$$

### 3.3 Backward Pass
1. **Delta de Salida:**
   $$err = y - \hat{y} = 1.0 - 0.03982836 = 0.96017164$$
   $$\delta_2 = err \cdot f'(H_2) = 0.96017164 \times 0.99841370 = 0.95864852$$

2. **Delta de la Capa Oculta 2 ($l=1$, dimensión $1 \times 2$):**
   $$\delta_2 W_2^T = 0.95864852 \begin{pmatrix} 0.50 & -0.45 \end{pmatrix} = \begin{pmatrix} 0.47932426 & -0.43139183 \end{pmatrix}$$
   $$\delta_1 = (\delta_2 W_2^T) \circ f'(H_1) = \begin{pmatrix} 0.47932426 \times 0.98910235 & -0.43139183 \times 0.99968752 \end{pmatrix} = \begin{pmatrix} 0.47410075 & -0.43125703 \end{pmatrix}$$

3. **Delta de la Capa Oculta 1 ($l=0$, dimensión $1 \times 3$):**
   $$\delta_1 W_1^T = \begin{pmatrix} 0.47410075 & -0.43125703 \end{pmatrix} \begin{pmatrix} 0.40 & -0.25 & 0.20 \\ -0.30 & 0.35 & 0.10 \end{pmatrix}$$
   - $(\delta_1 W_1^T)[0] = 0.47410075(0.40) - 0.43125703(-0.30) = 0.31901741$
   - $(\delta_1 W_1^T)[1] = 0.47410075(-0.25) - 0.43125703(0.35) = -0.26946515$
   - $(\delta_1 W_1^T)[2] = 0.47410075(0.20) - 0.43125703(0.10) = 0.05169445$
   $$\delta_0 = (\delta_1 W_1^T) \circ f'(H_0) = \begin{pmatrix} 0.31901741 \times 0.96104297 \\ -0.26946515 \times 0.96104297 \\ 0.05169445 \times 0.82200122 \end{pmatrix}^T = \begin{pmatrix} 0.30658944 & -0.25896759 & 0.04249290 \end{pmatrix}$$

4. **Gradientes y Actualización de Parámetros ($\eta = 0.1$):**
   - **Capa 2 (Salida):**
     $$\Delta W_2 = 0.1 \begin{pmatrix} -0.10439180 \\ 0.01767702 \end{pmatrix} (0.95864852) = \begin{pmatrix} -0.01000750 \\ 0.00169461 \end{pmatrix}$$
     $$\Delta b_2 = 0.1 \times 0.95864852 = 0.09586485$$
     $$W_2^{new} = \begin{pmatrix} 0.48999250 \\ -0.44830539 \end{pmatrix}, \quad b_2^{new} = \begin{pmatrix} 0.19586485 \end{pmatrix}$$

   - **Capa 1 (Oculta 2):**
     $$\Delta W_1 = 0.1 \begin{pmatrix} 0.19737532 \\ 0.19737532 \\ -0.42189901 \end{pmatrix} \begin{pmatrix} 0.47410075 & -0.43125703 \end{pmatrix} = \begin{pmatrix} 0.00935758 & -0.00851195 \\ 0.00935758 & -0.00851195 \\ -0.02000226 & 0.01819469 \end{pmatrix}$$
     $$\Delta b_1 = 0.1 \begin{pmatrix} 0.47410075 & -0.43125703 \end{pmatrix} = \begin{pmatrix} 0.04741008 & -0.04312570 \end{pmatrix}$$
     $$W_1^{new} = \begin{pmatrix} 0.40935758 & -0.30851195 \\ -0.24064242 & 0.34148805 \\ 0.17999774 & 0.11819469 \end{pmatrix}, \quad b_1^{new} = \begin{pmatrix} -0.00258992 & 0.00687430 \end{pmatrix}$$

   - **Capa 0 (Oculta 1):**
     $$\Delta W_0 = 0.1 \begin{pmatrix} -1.0 \\ 1.0 \end{pmatrix} \begin{pmatrix} 0.30658944 & -0.25896759 & 0.04249290 \end{pmatrix} = \begin{pmatrix} -0.03065894 & 0.02589676 & -0.00424929 \\ 0.03065894 & -0.02589676 & 0.00424929 \end{pmatrix}$$
     $$\Delta b_0 = 0.1 \begin{pmatrix} 0.30658944 & -0.25896759 & 0.04249290 \end{pmatrix} = \begin{pmatrix} 0.03065894 & -0.02589676 & 0.00424929 \end{pmatrix}$$
     $$W_0^{new} = \begin{pmatrix} 0.11934106 & -0.17410324 & 0.24575071 \\ 0.33065894 & 0.07410324 & -0.34575071 \end{pmatrix}$$
     $$b_0^{new} = \begin{pmatrix} 0.08065894 & -0.12589676 & 0.15424929 \end{pmatrix}$$

---

## 4. Verificación Automatizada

Los valores analíticos calculados arriba son exactamente los esperados por el test C++ `core/tests/test_mlp.cpp`:
- `TEST_CASE("MLP single backprop step reproduces docs/xor_a_mano.md for [2,2,1]")`
- `TEST_CASE("MLP single backprop step reproduces docs/xor_a_mano.md for [2,3,2,1]")`

Ambos tests configuran los pesos iniciales exactos mediante `MLP::set_weights_and_biases`, ejecutan exactamente una época sobre el patrón $x = [-1, 1], y = 1.0$, y comprueban que todos los pesos y sesgos resultantes coinciden con los de este documento dentro de una tolerancia de $10^{-6}$ (`doctest::Approx(expected).epsilon(1e-6)`).
