# Ejercicio 1 — Campañas de marketing: hallazgos

Los números y gráficos de esta nota salen de corridas de `python -m tp4 run <config>` (la lista
está al final, en "Cómo regenerar"). Los valores citados se copian de `runs/*/summary.json` y
de los CSV de cada corrida, y `python -m tp4.findings` verifica que cada archivo referenciado
exista y que los números clave aparezcan en el texto. Los decimales van con coma.

## 1.1 Exploración y representación (EDA + PCA)

### Datos y calidad

**El dataset está completo: 50 creadores, 33 columnas y ninguna celda faltante.**

- `runs/eda/summary.json`: `n_rows` = 50, `n_columns` = 33 (31 numéricas), `n_missing_cells` = 0.
- El contexto del enunciado hablaba de 49 creadores, pero el archivo `datos/creadores.csv`
  tiene 50 filas. Trabajamos con las 50 y ninguna se descarta en ningún paso.

### Distribuciones, outliers y escalas

**Los outliers son sobre todo creadores especializados, no errores: se reportan y se conservan.**

![Distribuciones con las cercas de Tukey](../../runs/eda/distributions.png)

- Con cercas de Tukey estrictas (k = 1,5) hay 33 celdas atípicas repartidas en 25 creadores
  (`runs/eda/columns.csv` y `runs/eda/outliers.csv`).
- Las columnas con más outliers son `topic_art_pct` (11) y `topic_makeup_pct` (11): todos son
  valores altos, es decir, creadores cuyo contenido es mayoritariamente de arte o de maquillaje
  (por ejemplo `@arteentusmanos` con 48,28% de arte o `@brochasymates` con 58,24% de maquillaje). Después vienen
  `estimated_argentina_reach` (5, columna derivada que se descarta) y `total_engagement_per_1000` (3,
  también derivada); `likes_per_1000`, `comments_per_1000` y `shares_per_1000` tienen 1 cada una.
- Los creadores con más celdas atípicas son `@galeriacotidiana`, `@maquillajereal` y
  `@tallerurbano` (3 cada uno).
- No se eliminan ni se recortan: un porcentaje de arte de 70,24% (`@artequeinspira`) no es ruido,
  es justamente el perfil que le sirve al cliente de arte, y con 50 creadores sacar filas
  cuesta más de lo que aporta.

**Las escalas difieren en cinco órdenes de magnitud, por eso hay que estandarizar.**

![Desvío estándar por variable (escala logarítmica)](../../runs/eda/scale_ranges.png)

- Entre las 25 variables seleccionadas, `followers` tiene un desvío de 208330,8098 y
  `comments_per_1000` de 1,4051: una relación de unas 148 mil veces (`runs/eda/columns.csv`).

### Correlaciones y columnas redundantes

**Seis relaciones exactas o casi exactas explican qué columnas sobran.**

![Mapa de calor de correlaciones](../../runs/eda/correlation_heatmap.png)

Las identidades se comprueban contra los datos, no se suponen (`runs/eda/identities.csv`; las seis se cumplen):

| Identidad | Máximo residuo | Tolerancia |
|-----------|----------------|------------|
| `price_ars = 1500 * price_usd` | 9,313225746e-10 | 0,01 |
| `total_engagement_per_1000` = suma de likes, comments, saves y shares por 1000 | 1,421085472e-14 | 0,025 |
| `estimated_argentina_reach = average_reach * argentina_audience_pct / 100` | 0,4985 (el alcance está redondeado) | 1 |
| `sponsored_posts_pct + organic_posts_pct = 100` | 0 | 0,025 |
| los cuatro `topic_*_pct` suman 100 | 1,421085472e-14 | 0,025 |
| los cuatro `content_*_pct` suman 100 | 1,421085472e-14 | 0,025 |

- Además hay 8 pares con |r| ≥ 0,95 (`runs/eda/near_redundant.csv`). Dos son las identidades
  `price_usd`~`price_ars` (r = 1) y `sponsored_posts_pct`~`organic_posts_pct` (r = −1), ya cubiertas.
  Los otros seis involucran solo variables seleccionadas, en dos grupos:
  - **Tamaño de audiencia:** `average_reach`~`average_video_views` 0,9972844494,
    `average_impressions`~`average_video_views` 0,9883558539 y `average_reach`~`average_impressions` 0,9874391284.
  - **Formato de contenido:** `video_posts_pct`~`short_form_posts_pct` 0,970759663,
    `short_form_posts_pct`~`trend_posts_pct` 0,9608453181 y `video_posts_pct`~`trend_posts_pct` 0,951541331.
- `followers` no entra en ningún par por encima de 0,95, aunque se comporta igual que el bloque de
  audiencia en la PCA (ver más abajo).

### Conjunto de features seleccionado

**Nos quedamos con 25 variables independientes: se descartan 6 columnas que son combinaciones exactas de otras.**

![Correlaciones del conjunto seleccionado](../../runs/eda/correlation_heatmap_selected.png)

Las 6 descartadas y su motivo (`configs/feature_set.json`, copiado en `runs/eda/feature_set.json`):

| Columna descartada | Motivo |
|--------------------|--------|
| `price_ars` | derivada de `price_usd` × 1500 (tipo de cambio fijo) |
| `total_engagement_per_1000` | suma exacta de likes, comments, saves y shares por 1000 |
| `estimated_argentina_reach` | derivada de `average_reach` × `argentina_audience_pct` / 100 (redondeada) |
| `organic_posts_pct` | complemento exacto, 100 − `sponsored_posts_pct` |
| `topic_art_pct` | el bloque de tópicos suma 100; se descarta el miembro de menor desvío |
| `content_showcase_pct` | el bloque de tipo de contenido suma 100; se descarta el miembro de menor desvío |

- Prueba de que el recorte es el correcto: la matriz numérica completa (31 columnas) tiene rango 26
  (`numeric_rank`), o sea 5 relaciones lineales exactas, y el conjunto seleccionado tiene rango 25 de 25
  (`selected_rank`): ninguna combinación lineal exacta sobrevive. La sexta identidad (el alcance estimado) es un
  producto, no lineal, y por eso no baja el rango, pero igual se descarta porque es una función de dos columnas que se quedan.
- **El arte no desaparece:** `topic_art_pct` se reconstruye como `100 − (topic_football_pct + topic_makeup_pct + topic_dogs_pct)`
  para leer al cliente de arte (idem `content_showcase_pct` con los otros tres tipos de contenido).
  Resumen del EDA: `n_selected_features` = 25.

### PCA: varianza explicada

**Una sola componente concentra más de la mitad de la varianza, pero con 2 no alcanza: hacen falta 6 para llegar al 80%.**

![Scree plot y varianza acumulada](../../runs/pca/scree.png)

PCA sobre las 25 variables estandarizadas (`runs/pca/summary.json`, `runs/pca/explained_variance.csv`):

| Indicador | Valor |
|-----------|-------|
| PC1 | 53,2% (autovalor 13,29442837 de 25) |
| PC2 | 8,0% |
| Acumulada con 2 componentes | 61,1% |
| Componentes para ≥ 80% (`k_cum_80`) | 6 (acumulada 0,8297495114) |
| Componentes para ≥ 90% (`k_cum_90`) | 9 (acumulada 0,9064882599) |
| Criterio de Kaiser (autovalor > 1, `kaiser_k`) | 6 |

- Después de PC1 la varianza se reparte de forma pareja: PC3 a PC6 aportan entre 4,1% y 7,4% cada una,
  así que no hay un "codo" claro tras la segunda componente.

### Interpretación de PC1 y PC2

**PC1 ordena a los creadores desde "audiencia grande y video corto de tendencia" hasta "audiencia chica, epígrafes largos y contenido didáctico".**

![Cargas de las primeras componentes](../../runs/pca/loadings_heatmap.png)

Los signos de una componente son arbitrarios: orientamos cada una para que su coeficiente de mayor
valor absoluto sea positivo. Lo que importa es qué variables van juntas y cuáles en contra.

- Mayores cargas en PC1 (`pc1_top5` en `runs/pca/summary.json`, coeficientes en `runs/pca/loadings.csv`):
  `average_caption_words` +0,257; `average_impressions` −0,251; `average_video_views` −0,250;
  `average_reach` −0,249; `trend_posts_pct` −0,248.
- Del lado negativo, además: `video_posts_pct` −0,242, `short_form_posts_pct` −0,241, `price_usd` −0,238,
  `followers` −0,238 y `likes_per_1000` −0,219. Es decir, **tamaño de audiencia, video corto/tendencia y precio se mueven juntos.**
  Lectura práctica: en este dataset los creadores más grandes tienden a ser también los más caros y los que más video corto publican.
- Del lado positivo: `average_caption_words` +0,257, `saves_per_1000` +0,231, `profile_visits_per_1000` +0,213,
  `comments_per_1000` +0,210, `average_watch_time_seconds` +0,201 y `content_tutorial_pct` +0,200: epígrafes largos, guardados,
  comentarios y tutoriales, o sea **un engagement más profundo con menos alcance.**

**PC2 mezcla la audiencia argentina y el tópico: perros y audiencia local arriba, fútbol y contenido patrocinado abajo.**

![Biplot PC1 vs PC2 coloreado por tópico dominante](../../runs/pca/biplot.png)

- Mayores cargas en PC2 (`pc2_top5`): `argentina_audience_pct` +0,439, `sponsored_posts_pct` −0,374,
  `topic_football_pct` −0,372, `topic_dogs_pct` +0,352, `likes_per_1000` −0,209.
- Creadores en los extremos (`runs/pca/scores.csv`):
  - PC1 más negativo: `@futbolsinvueltas` (−6,40), `@galeriacotidiana` (−5,84), `@brillourbano` (−5,68).
  - PC1 más positivo: `@mundoanimal.ar` (+6,77), `@consejoscaninos` (+6,49), `@huellasyconsejos` (+6,16).
  - PC2 más positivo: `@consejoscaninos` (+3,13), `@crearenreels` (+2,86), `@mateconideas` (+2,78).
  - PC2 más negativo: `@reelconideas` (−2,84), `@pasionalbirroja` (−2,26), `@entrelienzos` (−2,25).
- Qué muestran los colores del biplot (tópico dominante: fútbol 15, maquillaje 11, perros 13, arte 11):
  - Los creadores de perros se corren hacia PC1 positivo (media +2,78) y hacia PC2 positivo (media +1,20).
  - Fútbol tiene la media más baja en PC2 (−1,20) y maquillaje la más baja en PC1 (−1,73).
  - Pero la dispersión dentro de cada grupo es grande en PC1 (desvíos de 2,45 a 3,72), así que **PC1 no separa tópicos:
    separa tamaño y estilo.** En PC2 el desvío por grupo es de 0,93 a 1,20: ordena los tópicos (fútbol, arte, maquillaje, perros de abajo hacia arriba)
    pero con solapamiento.
  - Arte (media −0,58 en PC2 y −0,32 en PC1) queda en el centro, sin una dirección propia en estas dos componentes.
- Cuidado al leer nombres: el color es el tópico dominante calculado de las columnas de tópico, no el que sugiere el nombre
  de la cuenta (por ejemplo `@crearenreels` figura como perros y `@makeupconsejos` como fútbol).
- Lo que **no** muestran estas dos componentes: dos componentes cubren 61,1%, y el 38,9% restante
  (que incluye PC3 a PC6) no está en el biplot. Por eso no hay que leer grupos del biplot como clusters definitivos; eso es trabajo del mapa de Kohonen.

### ¿Estandarizar? (barrido)

**Sin estandarizar, PCA solo mide la escala: PC1 sube a 96,0% y lo domina `followers`.**

![Scree plot sin estandarizar](../../runs/pca_scaling_sweep/standardize-false/scree.png)

Barrido paralelo (`runs/pca_scaling_sweep/sweep.csv`), misma semilla y mismos datos, solo cambia `params.standardize`:

| | Estandarizada | Sin estandarizar |
|--|---------------|------------------|
| PC1 | 53,2% | 96,0% |
| PC2 | 8,0% | 3,8% |
| Acumulada con 2 componentes | 61,1% | 99,8% |
| Componentes para 80% / 90% | 6 / 9 | 1 / 1 |
| Variable con mayor carga en PC1 | `average_caption_words` (0,2573) | `followers` (0,8152) |

- Sin estandarizar, las cuatro variables de mayor desvío (`followers` +0,815, `average_impressions` +0,426,
  `average_reach` +0,311, `average_video_views` +0,239) acaparan PC1 y las demás no aparecen: la carga de `price_usd` es +0,006
  (`runs/pca_scaling_sweep/standardize-false/summary.json`). El 99,8% con dos componentes es un artefacto de la escala, no estructura.
- Biplot sin estandarizar (casi todas las flechas colapsan en el origen):

![Biplot sin estandarizar](../../runs/pca_scaling_sweep/standardize-false/biplot.png)

- **Conclusión: todo el análisis usa variables estandarizadas** (como PCA, Kohonen y Oja).

### Decisión: entrada de Kohonen

**Kohonen se entrena con las 25 variables estandarizadas, no con componentes principales.**

**Entrada de Kohonen:** features estandarizadas (25): el mapa se entrena sobre las 25 variables seleccionadas, estandarizadas con z-score, porque la PCA no comprime lo suficiente (2 componentes dejan afuera el 38,9% de la varianza y 6 todavía dejan afuera el 17,0%), el enunciado pide agrupar "en términos de todas sus variables", y así se obtienen directamente los planos de componentes por variable.

Los argumentos, con los números de las corridas:

1. **La reducción pierde demasiado para el beneficio.** Con 2 componentes la acumulada es 61,1%: queda afuera el 38,9% de la varianza.
   Para llegar al 80% hacen falta 6 componentes (`k_cum_80`; acumulada 83,0%) y para el 90% hacen falta 9 (`k_cum_90`; 90,6%). Pasar de 25 a 6 dimensiones
   con 50 creadores ahorra poco cómputo y todavía descarta el 17,0% de la información.
2. **El enunciado.** El ejercicio pide "agrupar los creadores de contenido en términos de todas sus variables". Entrenar sobre las 25 variables
   es lo más fiel a esa consigna, y la PCA ya aporta su propio mapa 2D (el biplot) para comparar.
3. **Planos de componentes por variable.** El resultado esperado del mapa incluye un plano de componentes por cada variable seleccionada.
   Con las 25 variables como entrada, cada plano sale directo de los pesos de las neuronas. Si se entrenara sobre componentes habría que proyectar
   los pesos de vuelta con las cargas (`runs/pca/loadings.csv`), con pérdida de la varianza que se descartó.
4. **El bloque casi colineal de audiencia pesa de más (y lo aceptamos con una verificación).** `average_reach`, `average_impressions` y `average_video_views`
   tienen correlaciones de 0,9874 a 0,9973 (`runs/eda/near_redundant.csv`), y `video_posts_pct`, `short_form_posts_pct` y `trend_posts_pct` de 0,9515 a 0,9708.
   En distancia euclídea sobre variables estandarizadas, tres variables casi iguales pesan unas tres veces en esa dirección:
   cada bloque ocupa 3 de las 25 dimensiones (12% del espacio) para lo que es esencialmente una sola.
   **Tratamiento en la Fase 2: aceptarlo, sin ponderar.** Razón: esa dirección es el eje dominante del dataset (PC1, 53,2% de la varianza,
   donde audiencia y video de tendencia cargan juntos), así que un mapa que la refleje no está distorsionado, está representando
   la estructura principal. Para comprobarlo, la Fase 2 entrena además un mapa de control sobre las 6 primeras componentes (`k_cum_80` = 6)
   con la misma semilla y compara errores de cuantización y topográfico y las regiones resultantes; si las conclusiones cambian,
   se reevalúa con ponderación de bloques. Los planos de componentes de variables de un mismo bloque se leerán como uno solo.

### Supuestos

- **Conjunto de features:** se descarta el miembro de menor desvío de cada bloque que suma 100 (`topic_art_pct`, `content_showcase_pct`);
  se podría haber descartado otro y la información sería la misma, pero se reconstruye cuando hace falta interpretar. Es la decisión
  más costosa de revertir porque las Fases 2 y 3 se apoyan en ella (siempre vía `tp4.features.load_design`).
- **Estandarización:** z-score con desvío poblacional (ddof = 0, como `StandardScaler`). La PCA sin estandarizar solo centra.
- **Signo de las componentes:** cada componente se orienta para que su coeficiente de mayor valor absoluto sea positivo; los autovalores no cambian.
- **Outliers:** cercas de Tukey estrictas con k = 1,5 (`outlier_iqr_k`); solo se marcan.
- **Datos:** 50 creadores aunque el contexto del enunciado mencionaba 49.
- **Interpretación de PCA:** lo dicho sobre PC1 y PC2 se apoya en los signos y las magnitudes de las cargas y en las posiciones del biplot;
  no decimos nada sobre estructura que esas figuras no muestren.

### Cómo regenerar

```bash
.venv/bin/python -m tp4 run configs/eda.json
.venv/bin/python -m tp4 run configs/pca.json
.venv/bin/python -m tp4 run configs/pca_scaling_sweep.json
```

Verificación de la evidencia:

```bash
.venv/bin/python -m tp4.findings docs/hallazgos/ej1.md \
  --cite runs/pca/summary.json:pc1_explained_variance_pct \
  --cite runs/pca/summary.json:pc2_explained_variance_pct \
  --cite runs/pca/summary.json:cumulative_pc2_pct \
  --cite runs/pca/summary.json:pc1_top_feature \
  --cite runs/pca_scaling_sweep/standardize-false/summary.json:pc1_explained_variance_pct \
  --cite runs/eda/summary.json:n_selected_features
```

## 1.2 Implementación: mapa de Kohonen

Todo lo de esta sección sale de `runs/som_design_sweep/`, `runs/som_sweep/`, `runs/som/` y `runs/som_pca_control/`,
regenerados con los comandos de "Cómo regenerar (1.2)". Los valores del `summary.json` se copian con su precisión guardada (4 decimales).
Hay decisiones que son criterio nuestro y el grupo todavía tiene que aceptar: están marcadas en "Supuestos (1.2)".

### Qué se implementó

**Escribimos el mapa de Kohonen a mano en numpy (`tp4/models/som.py`), sin ninguna librería de SOM ni de clustering, y se entrena con las 25 variables estandarizadas, como se decidió en 1.1.**

- Entrenamiento online, un creador por vez: se busca la neurona ganadora (BMU, la de menor distancia euclídea al creador) y se actualizan los pesos
  `W += eta * h(d(BMU, neurona), radio) * (x - W)`. Una iteración es un creador; una época son los 50 creadores en orden aleatorio.
- La vecindad `h` vale 1 en la ganadora y cae con la distancia en la grilla. Hay dos versiones: la regla escalón del curso (1 si `d <= radio`, 0 si no) y la gaussiana
  `exp(-d² / (2 radio²))`. `eta` y el radio decrecen en el tiempo (exponencial, lineal, inversa o constante) hasta un piso (`eta_min`, `radio_min`).
  La inicialización de pesos puede ser `sample` (creadores al azar) o `random` (uniforme dentro del rango de cada variable).
- Todos los hiperparámetros salen de `configs/som.json`, y esa config no se edita a mano (ver la elección más abajo).
- Métricas de desempeño, una línea cada una:
  - **Error de cuantización (QE):** distancia euclídea media entre cada creador y su neurona ganadora (unidades z). Baja si el mapa tiene más neuronas.
  - **Error topográfico (TE):** fracción de creadores cuya segunda neurona más cercana **no** es vecina de la ganadora en la grilla. Mide si el mapa preserva el orden.
  - **Preservación de distancias:** correlación de Pearson, sobre todos los pares de creadores, entre su distancia en el espacio de variables y la distancia en la grilla entre sus neuronas ganadoras.
- Lo que los tests (`tests/test_som_model.py`) garantizan: la ganadora es la neurona más cercana (contra fuerza bruta), la influencia de la vecindad vale 1 en la ganadora y baja con la distancia
  y con un radio que se achica, los pesos quedan dentro del casco convexo de los datos, y ningún módulo de `tp4/` importa una librería de SOM o clustering.

### Elección de la configuración (barridos)

**La configuración no se eligió a mano: la elige una regla fija sobre dos barridos, y el mapa final es de 8x8 hexagonal con vecindad escalón.**

![Métricas contra cada hiperparámetro en el barrido principal](../../runs/som_sweep/sweep_metrics.png)

![QE contra preservación de distancias, con el mapa elegido](../../runs/som_sweep/sweep_tradeoff.png)

Etapa 1, diseño (`runs/som_design_sweep/selection.csv`): topología x vecindad x inicialización sobre un mapa de referencia de 6x6, eta0 0,5, radio0 3 y 200 épocas.
Quedó elegido `topology-hex__neighborhood-step__init-random`.

| Topología | Vecindad | Inicialización | QE | TE | Preservación | Neuronas vacías | Elegido |
|-----------|----------|----------------|----|----|--------------|-----------------|---------|
| hex | gaussiana | sample | 2,818 | 0,02 | 0,762 | 0,3889 | no |
| hex | gaussiana | random | 2,8154 | 0 | 0,7673 | 0,3889 | no |
| hex | escalón | sample | 2,6175 | 0,02 | 0,7191 | 0,3056 | no |
| hex | escalón | random | 2,6006 | 0 | 0,7443 | 0,2222 | **sí** |
| rect | gaussiana | sample | 2,7739 | 0,06 | 0,7578 | 0,3056 | no (TE) |
| rect | gaussiana | random | 2,7749 | 0,06 | 0,7529 | 0,3056 | no (TE) |
| rect | escalón | sample | 2,5167 | 0,02 | 0,6982 | 0,1667 | no (preservación) |
| rect | escalón | random | 2,4103 | 0 | 0,6844 | 0,2778 | no (preservación) |

- Los mapas `rect` tienen menor QE, pero la regla los descarta: ninguno cumple a la vez el TE y la preservación. **Ojo:** el TE es más exigente con `rect` (4 vecinas por neurona) que con
  `hex` (6 vecinas), así que parte de esa diferencia es de definición y no solo de calidad del mapa.
- **La vecindad elegida es la escalón, no la gaussiana** (la gaussiana era el valor por defecto de la primera versión del mapa). La escalón también cumple lo pedido (la influencia no crece con la distancia en la grilla)
  y dio menor QE (2,6006 contra 2,8154 con la misma inicialización).

Etapa 2, hiperparámetros numéricos (`runs/som_sweep/`): 216 mapas = lado de la grilla cuadrada (4, 5, 6, 7, 8, 10) x eta0 (0,1; 0,5; 0,9) x radio0 (1, 2, 3, 5) x épocas (50, 200, 500) sobre el diseño ya elegido.

- **El QE solo siempre prefiere el mapa más grande.** Promedio por lado: 3,09 con 4x4, 2,85 con 5x5, 2,62 con 6x6, 2,39 con 7x7, 2,13 con 8x8 y 1,62 con 10x10. El de menor QE de todo el barrido
  (`rows-10__eta0-0p5__radius0-1__epochs-500`, QE 1,0524) deja 0,63 de sus neuronas vacías y tiene preservación de distancias 0,5914: es el mapa que memoriza a los creadores, no el que los ordena.
- **El TE solo no alcanza para rankear con 50 creadores:** solo toma valores en pasos de 0,02 (un creador de 50), y 173 de los 216 mapas tienen TE exactamente 0.
- La regla, en cuatro etapas (inclusivas, con umbrales que viven en `tp4/analysis/som_selection.py`):
  1. ocupación: a lo sumo 0,5 de neuronas vacías (la pasan 164 de 216, `n_occupancy_ok`);
  2. TE: a lo sumo 0,02 por encima del mejor (159, `n_topology_ok`);
  3. preservación de distancias: a lo sumo 0,05 por debajo de la mejor (35, `n_order_ok`);
  4. entre los que quedan, el de menor QE (empates: menos neuronas, luego menos épocas).

**Configuración final:** `rows-8__eta0-0p5__radius0-5__epochs-500` sobre `topology-hex__neighborhood-step__init-random` (topología hexagonal, vecindad escalón, inicialización aleatoria), con QE 2,1214, TE 0,0 y preservación de distancias 0,776.

- El segundo de la regla es `rows-8__eta0-0p5__radius0-3__epochs-500` (QE 2,1871, preservación 0,755, 0,4531 de neuronas vacías).
- **Caveat que el grupo tiene que ver:** el mapa elegido tiene exactamente 0,5 de neuronas vacías (32 de 64), o sea que pasa el filtro de ocupación solo porque el umbral es inclusivo.
  Los tres umbrales (0,5; 0,02; 0,05) son un criterio del plan, no un resultado: con otros valores la regla podría elegir otro mapa.

### Entrenamiento

**El QE baja de 3,90 en la época 1 a 2,12 y se estabiliza en las últimas 100 épocas; el TE llega a 0 y se queda ahí.**

![Curvas de entrenamiento](../../runs/som/training_curve.png)

- Según `runs/som/history.csv`: QE 3,9000 en la época 1, 2,5060 en la 300, 2,1195 en la 400 y 2,1214 en la 500; entre las épocas 401 y 500 se mueve en un rango de 2,0901 a 2,1215.
  El TE vale 0 en cada una de las épocas 400 a 500 (llegó a 0,68 en la época 10, cuando el radio todavía era grande).
- eta baja de 0,5 a 0,01 y el radio de 5 a 1 (sus pisos), así que en el final solo se actualizan la ganadora y sus vecinas inmediatas.

### Distribución de los creadores en la grilla

**Los 50 creadores ocupan la mitad de las 64 neuronas: 32 quedan vacías, ninguna neurona junta más de 3 creadores.**

![Mapa de impactos](../../runs/som/hit_map.png)

- `runs/som/summary.json`: `n_empty_neurons` = 32 (`empty_neuron_frac` = 0,5), `max_hits` = 3, `mean_hits_occupied` = 1,5625 creadores por neurona ocupada.
- Las neuronas con 3 creadores (`runs/som/neurons.csv`):
  - neurona 0: `@huellasyconsejos`, `@mundoanimal.ar` y `@consejoscaninos` (los tres de perros);
  - neurona 32: `@pasionalbirroja`, `@futbolentrepases` y `@reelconideas` (los tres de fútbol);
  - neurona 50: `@canchaaldia` y `@manosalaobra` (fútbol) con `@canvasycafe` (arte);
  - neurona 38: `@vidaperruna`, `@tacticavisual` y `@makeupconsejos` (los tres con fútbol como tópico dominante, ojo: el color sale de las columnas, no del nombre de la cuenta).
- Las neuronas vacías no son un error: guardan prototipos interpolados entre grupos. Eso importa para leer la matriz U (abajo).

### Matriz U: qué indica una distancia alta

**Una distancia alta entre neuronas vecinas es una frontera: los creadores de un lado y del otro se parecen poco en las variables. Distancia baja es una zona de creadores intercambiables.**

![Matriz U con las fronteras de las regiones](../../runs/som/u_matrix.png)

- Matriz U (distancia media a las vecinas, unidades z): media 2,0213 y máximo 3,2606 (`u_mean`, `u_max`). Las celdas más claras son la de `@tallerurbano` (3,2606) y la de
  `@mateconideas` con `@maquillajereal` (3,0531). Las más oscuras son las neuronas 48 y 50 (U 1,0777 y 1,0667) de la esquina inferior izquierda, donde se juntan `@brochasycolor` con `@reelargentino` y `@canchaaldia` con `@manosalaobra` y `@canvasycafe`: ahí los creadores son los más parecidos entre sí.
- Las tres paredes más fuertes (`runs/som/u_edges.csv`, distancia en unidades z, diferencias de la neurona B respecto de la A) rodean todas a **`@tallerurbano`** (neurona 22):
  1. 22~23, 4,4767: `topic_makeup_pct:+1.70;shares_per_1000:-1.61;price_usd:+1.26`;
  2. 22~30, 4,2266: `price_usd:+1.83;topic_dogs_pct:-1.23;shares_per_1000:-1.20`;
  3. 14~22, 3,9982: `topic_makeup_pct:-1.67;sponsored_posts_pct:-1.47;shares_per_1000:+1.23`.
  Las vecinas de `@tallerurbano` son neuronas vacías, así que la pared dice que no se parece a ningún otro creador: es un perfil aislado (en 1.1 ya era uno de los de más celdas atípicas).
  Esas neuronas difieren de la suya hasta 1,8 desvíos en maquillaje, `shares_per_1000` y `price_usd`.
- Las paredes entre dos neuronas **ocupadas** son más informativas. De las 161 aristas del mapa, solo 25 tienen creadores en los dos lados, y las más fuertes son:
  - `@lookdecancha` | `@tribunacreativa` (60~61, 3,2628): `average_watch_time_seconds:-1.21;average_video_views:+1.15;average_impressions:+1.15`. Es parte de la frontera entre las regiones 2 y 3: mismo tópico (fútbol) pero otro nivel de alcance y de tiempo de visualización.
  - `@trazosyreels` | `@rinconcreativo` (18~19, 2,7484): `saves_per_1000:-1.44;content_tutorial_pct:-1.10;content_entertainment_pct:+1.05`. Es parte de la frontera entre las regiones 1 y 2: dos creadores de arte, uno con muchos guardados y tutoriales y el otro con más entretenimiento.
  - `@estiloydetalle` | `@creatividadaldia` (25~34, 2,3570): `saves_per_1000:-1.24;sponsored_posts_pct:+0.95;argentina_audience_pct:-0.87`. Dentro de la región 1: una diferencia fuerte entre creadores de la misma región.
- Las fronteras de las regiones (líneas rojas) siguen las paredes del centro-arriba y del costado derecho, pero la frontera larga de la mitad inferior pasa por celdas de U intermedia:
  `u_border_ratio` = 1,5729 (la distancia media a través de las fronteras es 1,57 veces la media dentro de las regiones).
  La separación es real pero moderada, y con la mitad del mapa vacía hay que leerla con cuidado.

### Planos de componentes

**Las variables de un mismo bloque casi colineal dibujan el mismo plano, y lo que ordena el mapa es el eje alcance y video contra engagement profundo.**

![Planos de componentes en unidades originales](../../runs/som/component_planes.png)

Los planos se leen en unidades originales (`runs/som/prototypes.csv`). Siguiendo el compromiso de 1.1, cada bloque casi colineal se lee como una sola variable:

- **Audiencia** (`average_reach`, `average_impressions`, `average_video_views`): sus planos son prácticamente iguales (correlación entre los 64 prototipos 1,00); `followers` y `price_usd` los acompañan (0,96).
  Valor alto en la esquina inferior derecha del mapa.
- **Video corto y tendencia** (`video_posts_pct`, `short_form_posts_pct`, `trend_posts_pct`): correlación de 0,97 a 0,99 entre sus planos, valor alto en el centro-derecha y abajo a la derecha.
- **Engagement profundo** (`saves_per_1000`, `average_caption_words`, `comments_per_1000`, `profile_visits_per_1000`): altos en el borde izquierdo y en la esquina superior izquierda,
  y correlacionados negativamente con el alcance (de −0,76 a −0,91 con `average_reach`). Es el mismo eje PC1 de 1.1 (audiencia y video corto contra epígrafes largos y guardados).
- **Tópicos:** `topic_makeup_pct` alto arriba a la derecha, `topic_dogs_pct` arriba a la izquierda y `topic_football_pct` abajo en el centro.
- **Arte reconstruido** (`topic_art_pct`, `100 − fútbol − maquillaje − perros`): alto en una zona del centro-izquierda, con el máximo de 63,3% en la neurona 35 (`@artealpaso`); también es alto en las neuronas de `@pincelcotidiano` (53,4%) y de `@patitasenred` con `@artequeinspira` (49,5%).
  Arte no tiene un rincón propio como maquillaje: está en el medio del mapa, entre los otros tópicos.

### Regiones del mapa y perfil de los creadores

**El mapa se parte en 3 regiones contiguas: engagement profundo con audiencia chica (21 creadores), un grupo intermedio y mezclado (18) y alcance masivo con video y entretenimiento (11).**

![Regiones contiguas del mapa](../../runs/som/region_map.png)

![Perfil de cada región (media z)](../../runs/som/region_profiles.png)

- **Cómo se definieron.** Las regiones son grupos de neuronas contiguas (aglomeración de Ward con restricción de contigüidad sobre los 64 prototipos). `k` lo elige una regla registrada
  (`runs/som/region_scan.csv`): entre los `k` de 2 a 8 cuyas regiones tienen al menos 3 creadores, el que maximiza la razón entre la distancia media a través de las fronteras y dentro de las regiones.
  Eligió **k = 3** (`n_regions`, razón 1,5729); los `k` de 6 a 8 quedan descartados por una región de solo 2 creadores. **Ojo:** la razón es casi plana entre `k` = 3 y 8 (de 1,52 a 1,57),
  o sea que la U-matrix no determina `k` con fuerza. `k` = 5 (razón 1,5515, mínimo 8 creadores por región) es la siguiente granularidad defendible si hace falta más detalle.
- **Perfil de cada región** (`runs/som/regions.csv` y `runs/som/region_profiles.csv`; z = desvíos respecto del promedio de los 50 creadores):

| Región | Neuronas / creadores | Tópicos dominantes | Más altas (z) | Más bajas (z) |
|--------|----------------------|--------------------|---------------|---------------|
| 1 | 26 / 21 | perros 9, fútbol 6, arte 4, maquillaje 2 | `saves_per_1000` +1,0; `average_caption_words` +1,0; `comments_per_1000` +0,8 | `video_posts_pct` −1,0; `trend_posts_pct` −1,0; `short_form_posts_pct` −0,9 |
| 2 | 22 / 18 | fútbol 6, arte 6, maquillaje 3, perros 3 | `trend_posts_pct` +0,6; `short_form_posts_pct` +0,5; `video_posts_pct` +0,5 | `saves_per_1000` −0,5; `average_caption_words` −0,4; `profile_visits_per_1000` −0,3 |
| 3 | 16 / 11 | maquillaje 6, fútbol 3, perros 1, arte 1 | `average_video_views` +1,5; `average_reach` +1,5; `average_impressions` +1,5 | `average_caption_words` −1,2; `comments_per_1000` −1,0; `saves_per_1000` −1,0 |

- **Región 1, audiencia chica y comunidad que interactúa.** Seguidores 52029 de promedio (z −0,82), epígrafes de 149,6 palabras, 18,9 guardados por mil y solo 40,7% de posts en video;
  perros es el tópico más repetido (38,2% de contenido de perros en promedio). Creadores: `@huellasyconsejos`, `@mundoanimal.ar`, `@consejoscaninos`, `@canchaaldia`, `@pasionalbirroja`.
  En lenguaje de campaña: **profundidad educativa y comunidad cercana**, no alcance. Lo muestran las variables de engagement y de tutoriales/reseñas altas (`content_tutorial_pct` +0,66, `content_review_pct` +0,75) y el alcance bajo.
- **Región 2, intermedia y mezclada.** Es la región con más arte (28,6% de contenido de arte en promedio, la más alta; `@artealpaso`, `@pincelcotidiano`, `@rinconcreativo`, `@artequeinspira`) pero también 6 de fútbol.
  Más video corto y tendencia (60,6% de posts en video, 48,8% de tendencia), 65,1% de alcance a no seguidores y más publicaciones por mes (18,9), con alcance y seguidores en el promedio (`followers` z +0,06).
  No tiene un perfil de tópico: es el grupo "de paso" entre las regiones 1 y 3, y es donde están las neuronas con más arte. Para una campaña sirve cuando importa el descubrimiento (alcance a no seguidores) y no el tópico.
- **Región 3, alcance masivo y entretenimiento.** Seguidores 525920 de promedio (z +1,48), alcance 205146, impresiones 281961 y 157233 vistas de video, con un precio medio de 3846,71 USD (z +1,25) y 69,1% de contenido de entretenimiento;
  pocos epígrafes (43,6 palabras), comentarios y guardados bajos. El maquillaje domina (43,8% de contenido en promedio; `@mateconideas`, `@maquillajereal`, `@brillourbano`, `@glamencasa`),
  con `@futbolsinvueltas`, `@golconmate` y `@galeriacotidiana`. En lenguaje de campaña: **alcance masivo y descubrimiento, a un precio alto y con poca conversación.**
- Los perfiles salen de las medias de los creadores de cada región, no de las neuronas vacías, y con 50 creadores y regiones de 11 a 21 las diferencias chicas (por ejemplo ±0,3 z en la región 2) no son concluyentes.

### Comparación con PC1 y PC2 (biplot)

**Las regiones coinciden con PC1 y casi nada con PC2: PC1 explica el 83% de la diferencia entre regiones, PC2 el 21%.**

![Regiones del mapa sobre el plano PC1-PC2](../../runs/som/biplot_regions.png)

![Planos de PC1 y PC2 en el mapa](../../runs/som/pc_planes.png)

- `pc1_region_eta2` = 0,8314 y `pc2_region_eta2` = 0,2066 (`runs/som/summary.json`): la fracción de la varianza de cada componente que explican las 3 regiones.
- En el biplot de 1.1 (`runs/pca/biplot.png`) las regiones se ordenan de derecha a izquierda: región 1 con PC1 medio +3,61 (engagement profundo, epígrafes largos), región 2 en el centro con −1,31 y
  región 3 con −4,75 (audiencia grande y video corto de tendencia). **Es exactamente la lectura de PC1 de 1.1**, y es lo que esperábamos: el mapa y la PCA ven el mismo eje dominante (53,2% de la varianza).
- En PC2 las regiones casi no se separan: medias de −0,05, −0,62 y +1,11, con desvíos de 1,1 a 1,5 dentro de cada región. Solo la región 3 se corre hacia arriba (PC2 positivo, donde 1.1 ubicaba la audiencia argentina y a perros arriba y fútbol abajo).
  **El eje "perros arriba, fútbol abajo" de PC2 no se corresponde con ninguna región:** los 13 creadores de perros están repartidos en las tres regiones (9, 3 y 1), y la región 1 abarca casi todo el rango de PC2 del biplot (desvío 1,48 en PC2).
- Lo que el biplot no puede mostrar: solo tiene el 61,1% de la varianza (queda afuera el 38,9% de 1.1), y en el mapa se ve el resto. Las paredes de la matriz U de arriba dicen qué variables separan a creadores vecinos
  (por ejemplo, `@lookdecancha` y `@tribunacreativa`, los dos de fútbol, difieren sobre todo en visualizaciones y tiempo de visualización). La frontera entre las regiones 2 y 3 aparece en el biplot como una zona de solapamiento
  (puntos naranjas y violetas mezclados entre PC1 = −4,5 y −1,5), no como un corte limpio: ahí están justamente los 6 creadores que cambian de región en el mapa de control (abajo).

### Mapa de control sobre 6 componentes principales

**El mapa de control sobre 6 componentes (83,0% de la varianza) da prácticamente las mismas regiones: la conclusión de 1.1 no cambia.**

![Mapa sobre las variables y mapa de control sobre 6 componentes](../../runs/som_pca_control/control_region_maps.png)

Cumplimos lo prometido en 1.1: un mapa entrenado sobre las 6 primeras componentes, con los mismos hiperparámetros y la misma semilla (`runs/som_pca_control/control_comparison.csv`). El mapa sobre las variables de esta corrida es exactamente el de `runs/som` (un test lo verifica).

| Métrica | Mapa sobre las 25 variables | Mapa de control (6 componentes) |
|---------|-----------------------------|---------------------------------|
| QE | 2,1214 | 1,4506 (no comparable) |
| TE | 0,0 | 0,02 |
| Preservación de distancias | 0,776 | 0,8009 |
| Neuronas vacías | 0,5 | 0,3906 |
| Regiones (k) | 3 | 3 |

- **El QE no se puede comparar:** el espacio de 6 componentes deja afuera el 17,0% de la varianza (se conserva el 83,0%), así que todas las distancias son más cortas. Se comparan el TE, la preservación y las regiones.
- **Acuerdo de regiones:** `region_ari` = 0,748 (índice de Rand ajustado entre las regiones de los dos mapas, con el mismo k = 3). Con la escala que fijamos (≥ 0,5: grupos en lo esencial iguales; 0,3 a 0,5: acuerdo parcial; < 0,3: grupos distintos), es **acuerdo alto**.
  De los 50 creadores, 44 quedan en la misma región (21 de la región 1 coinciden todos); los 6 que cambian están en la frontera entre las regiones 2 y 3: `@vidaperruna`, `@tacticavisual`, `@lookdecancha` y `@makeupconsejos` pasan de la 2 a la 3,
  y `@bellezasimple` y `@brilloytendencia` de la 3 a la 2 (`runs/som_pca_control/creator_regions.csv`). Las regiones quedan en 21, 16 y 13 creadores contra 21, 18 y 11.
- **Acuerdo de disposición:** `bmu_distance_agreement` = 0,7459 (correlación, sobre todos los pares de creadores, entre su distancia en la grilla de un mapa y en el otro).
- **¿Cambian las conclusiones de 1.1?** No. Como `region_ari` ≥ 0,3, las regiones no dependen de los bloques casi colineales de audiencia y de video, y el tratamiento de 1.1 ("aceptarlo, sin ponderar") se mantiene.
  La ponderación por bloques, que era el plan B de 1.1, no se activa. La única diferencia real es de detalle: el control reparte distinto a 6 creadores de frontera.

### Supuestos (1.2)

- **Umbrales de la regla de selección** (vacías ≤ 0,5; TE a lo sumo 0,02 sobre el mejor; preservación a lo sumo 0,05 bajo la mejor): son un criterio del plan y el grupo todavía tiene que aceptarlos.
  El mapa final cae justo en el borde de ocupación (0,5 de vacías).
- **Geometría y fórmulas:** grilla hexagonal con filas desplazadas (todas las vecinas a distancia 1) y vecindad escalón con borde incluido; los decaimientos son `eta0 → eta_min` y `radio0 → radio_min` exponenciales sobre todas las iteraciones, con el radio nunca por debajo de `radio_min`.
- **Barrido en dos etapas, coordenada por coordenada:** primero el diseño (topología, vecindad, inicialización) sobre una referencia de 6x6 y después los numéricos con el diseño elegido; no se barre el producto completo de las 7 dimensiones.
- **Regiones:** Ward con contigüidad sobre los prototipos (sin ponderar por impactos); `k` lo elige la razón de U entre `k` = 2 a 8 con al menos 3 creadores por región (con k = 3 salió la razón más alta). Una región es una agrupación de creadores, no una categoría de campaña.
- **Escala del ARI:** ≥ 0,5 grupos en lo esencial iguales; 0,3 a 0,5 acuerdo parcial; < 0,3 grupos distintos (criterio del plan; por debajo de 0,3 se reabre la ponderación por bloques de 1.1).
- **Planos de componentes** en unidades originales (`media + w * desvío`), y `topic_art_pct` reconstruido como 100 menos los otros tres tópicos.
- **Perfil de una región:** media de los valores estandarizados de sus creadores, más la media en unidades originales para leerlo. **Matriz U:** distancia media a las neuronas vecinas; una pared hacia una neurona vacía mide aislamiento, no una frontera entre dos grupos.
- Los gráficos se enlazan desde `runs/` (se regeneran, no se versionan). Copiarlos a `docs/` queda para la fase de documentación.

### Cómo regenerar (1.2)

```bash
.venv/bin/python -m tp4 run configs/eda.json
.venv/bin/python -m tp4 run configs/pca.json
.venv/bin/python -m tp4 run configs/pca_scaling_sweep.json
.venv/bin/python -m tp4 run configs/som_design_sweep.json
.venv/bin/python -m tp4 run configs/som_sweep.json
.venv/bin/python -m tp4 run configs/som.json
.venv/bin/python -m tp4 run configs/som_pca_control.json
.venv/bin/python -m tp4.experiments.som_sweep_report check configs/som.json runs/som_design_sweep runs/som_sweep --run runs/som
```

Verificación de la evidencia (los seis valores de 1.1 más los de 1.2):

```bash
.venv/bin/python -m tp4.findings docs/hallazgos/ej1.md \
  --cite runs/pca/summary.json:pc1_explained_variance_pct \
  --cite runs/pca/summary.json:pc2_explained_variance_pct \
  --cite runs/pca/summary.json:cumulative_pc2_pct \
  --cite runs/pca/summary.json:pc1_top_feature \
  --cite runs/pca_scaling_sweep/standardize-false/summary.json:pc1_explained_variance_pct \
  --cite runs/eda/summary.json:n_selected_features \
  --cite runs/som_sweep/summary.json:selected_child \
  --cite runs/som_sweep/summary.json:selected_quantization_error \
  --cite runs/som_sweep/summary.json:selected_topographic_error \
  --cite runs/som_sweep/summary.json:selected_distance_preservation \
  --cite runs/som_design_sweep/summary.json:selected_child \
  --cite runs/som/summary.json:n_regions \
  --cite runs/som/summary.json:u_border_ratio \
  --cite runs/som/summary.json:pc1_region_eta2 \
  --cite runs/som/summary.json:pc2_region_eta2 \
  --cite runs/som/summary.json:empty_neuron_frac \
  --cite runs/som_pca_control/summary.json:region_ari \
  --cite runs/som_pca_control/summary.json:bmu_distance_agreement
```

## 1.3 Índice (regla de Oja)

Todo lo de esta sección sale de `runs/oja/`, `runs/oja_sweep/` y `runs/campaigns/`, regenerados con los comandos de "Cómo regenerar (1.3)". Los valores de los `summary.json` se copian con su precisión guardada (6 decimales para el coseno y la correlación, 4 para normas y autovalores). Los criterios que son nuestros y que el grupo todavía tiene que aceptar están en "Supuestos (1.3)".

### Regla de Oja implementada

**Escribimos una sola neurona de Oja a mano en numpy (`tp4/models/oja.py`), sin librerías de redes ni autovalores, y la entrenamos con las mismas 25 variables estandarizadas de 1.1.**

- La regla del curso, un creador por vez: `y = w·x` y `Δw = η · y · (x − y · w)`. Una época son los 50 creadores en orden aleatorio, con una permutación nueva por época y semilla fija (42).
- Entrada: las 25 variables estandarizadas de 1.1 (las mismas que usan la PCA y el mapa de Kohonen), así que el índice y PC1 se calculan sobre exactamente los mismos datos. Pesos iniciales: uniformes en [0, 1), como pide el curso.
- Después de entrenar, el vector se normaliza a largo 1 y se orienta como en 1.1 (el coeficiente de mayor valor absoluto positivo). **El índice de un creador es `Z · ŵ`**: su fila estandarizada proyectada sobre ese vector. Con `ŵ` de largo 1, el índice está en las mismas unidades que el puntaje de PC1.
- Lo que garantizan los tests (`tests/test_oja_model.py`): sobre datos sintéticos con un autovector conocido la regla lo recupera con |cos| ≥ 0,99 y la norma queda a menos de 0,01 de 1, el paso a mano coincide exacto con la implementación, la corrida es determinista, la divergencia se marca en el resultado en vez de romper, y ningún módulo de `tp4/` importa una librería de redes ni llama a un autovalor o a un SVD.

### Convergencia y elección de la tasa de aprendizaje

**La norma de `w` baja de 2,20 en la primera época y entra en la banda de 1 ± 0,01 en la época 9; al final vale 1,0015.**

![Norma de w y cambio por época](../../runs/oja/convergence.png)

- `runs/oja/summary.json`: `final_w_norm` = 1,0015 y `epochs_to_converge` = 9 (la primera época desde la que la norma queda dentro de `norm_tol` = 0,01). La norma de partida era 3,3422 (`initial_w_norm`).
- Con η constante la regla no se clava en 1,0000 sino un poco arriba (1,0015) y sigue moviéndose un poco por época (`final_delta_w` = 0,004425): es el ruido del paso online, no una falta de convergencia.
- La dirección también converge: el coseno con PC1 pasa de 0,47 en la época 1 a 0,99 en la 9 (`runs/oja/history.csv`, `runs/oja/cos_to_pc1.png`).

![Coseno con PC1 por época](../../runs/oja/cos_to_pc1.png)

**La tasa no se eligió a ojo: un barrido de 9 valores de η0 con 3 semillas (27 corridas) y una regla registrada eligió η0 = 0,0005, sin mirar PC1.**

![Norma de w por época para cada η0 y cada semilla](../../runs/oja_sweep/sweep_convergence.png)

Tabla de `runs/oja_sweep/selection.csv` (por η0, el peor caso entre las 3 semillas):

| η0 | Semillas divergentes | Épocas para converger (máx.) | Desvío máximo de la norma | Desplazamiento máx. de ranking | Pasa estable / norma / orden |
|----|----------------------|------------------------------|----------------------------|--------------------------------|------------------------------|
| 0,00001 | 0 de 3 | no converge | 1,2298 | 42 | sí / no / no |
| 0,0001 | 0 de 3 | 118 | 0,0003 | 0 | sí / sí / sí |
| **0,0005** | 0 de 3 | **24** | 0,0015 | 0 | sí / sí / sí |
| 0,001 | 0 de 3 | 12 | 0,003 | 2 | sí / sí / no |
| 0,002 | 0 de 3 | 6 | 0,0063 | 49 | sí / sí / no |
| 0,005 | 0 de 3 | no converge | 0,0169 | 49 | sí / no / no |
| 0,02 | 0 de 3 | no converge | 0,1149 | 49 | sí / no / no |
| 0,06 | 3 de 3 | | | | no |
| 0,5 | 3 de 3 | | | | no |

- **Cuatro regímenes.** (1) *Demasiado lenta*: con η0 = 0,00001 la norma final de las tres semillas sigue entre 1,2149 y 2,2298 después de 200 épocas y el coseno con PC1 queda entre 0,01 y 0,81. (2) *Converge*: de 0,0001 a 0,002 la norma entra en la banda; cuanto mayor el η0, menos épocas (118, 24, 12, 6). (3) *Oscila fuera de la banda*: con 0,005 y 0,02 la norma se queda dando vueltas con un desvío de 0,0169 y 0,1149. (4) *Diverge*: con 0,06 y 0,5 las tres semillas explotan (en la época 1 con η0 = 0,5).
- **La guía del curso diverge acá.** El autovalor mayor de la matriz de correlación es `lambda1` = 13,2944, y la cota del curso `η0 = 1/(1,2·λ1)` da `course_eta_bound` = 0,0627. Con 0,06, e incluso con la regla de "η0 ≤ 0,5", la corrida online diverge. La razón: cada paso individual escala con ‖x‖² del creador (media 25, hasta unos 57 para `@mundoanimal.ar`, calculado sobre las 25 variables estandarizadas), no con λ1, que es el promedio. Por eso η tiene que ser un orden de magnitud o más chico que la cota.
- **Regla registrada** (`selection_rule` en `runs/oja_sweep/summary.json`): se descartan los η0 con alguna semilla divergente; luego los que dejan alguna semilla fuera de la banda de la norma; luego los cuyas semillas ordenan a los creadores distinto (desplazamiento máximo de ranking mayor que 0); entre los que quedan, el que converge en menos épocas (el máximo entre semillas) y, en empate, el η0 menor. PC1 no interviene.
- Cuántos pasan cada etapa (de `n_eta` = 9): `n_stable` = 7, `n_norm_ok` = 4, `n_index_ok` = 2. Sobreviven 0,0001 y 0,0005; gana 0,0005 con 24 épocas contra 118.

**Tasa de aprendizaje final:** η0 = 0,0005 (`runs/oja_sweep/summary.json`: `selected_eta0` = 0,0005, `selected_child` = `eta0-0p0005__seed-42`), constante, 200 épocas; `configs/oja.json` lleva ese valor y un test lo verifica.

### Comparación con PC1

**El vector de Oja y el primer autovector de la PCA son casi el mismo: coseno de 0,999993 entre los dos y correlación de 0,999999 entre los puntajes de los 50 creadores.**

![Peso de Oja contra carga de PC1, variable por variable](../../runs/oja/weights_vs_pc1.png)

- `cosine_to_pc1` = 0,999993 y `pearson_scores_pc1` = 0,999999 (`runs/oja/summary.json`). La diferencia más grande entre un peso de Oja y la carga de PC1 es de 0,003172 (`max_abs_weight_diff`, en `monthly_posts`: −0,0366 de Oja contra −0,0397 de PC1; `runs/oja/weights_vs_pc1.csv`). En las otras 24 variables la diferencia es menor a 0,0007.

![Índice de Oja contra puntaje de PC1 por creador](../../runs/oja/scores_vs_pc1.png)

- En el gráfico de dispersión los 50 creadores caen sobre la diagonal (`runs/oja/ranking.csv`: `oja_index` y `pc1_score`; por ejemplo `@mundoanimal.ar` 6,7680 contra 6,7728 y `@futbolsinvueltas` −6,4065 contra −6,4034). Ordenar por el índice o por el puntaje de PC1 da exactamente el mismo ranking de los 50 creadores.
- **Es lo esperado, no una coincidencia:** la regla de Oja converge al primer autovector de la matriz de covarianza, y PC1 es justamente ese autovector. Lo que cambia es el camino: Oja llega online, un creador por vez, sin descomponer ninguna matriz.
- **El signo es frágil y por eso se orienta.** La red encuentra el eje, no su sentido: con η0 = 0,0005 las semillas 42 y 7 convergen al opuesto de PC1 (`seeds_flipped`) y la 11 al mismo signo (`seeds_not_flipped`). El `oja_top_feature` de la corrida es `average_caption_words`, igual que `pc1_top_feature`, pero la regla de orientación (coeficiente de mayor |valor| positivo, como en 1.1) se apoya en una ventaja mínima: en PC1, `average_caption_words` (+0,2573) le gana a `average_impressions` (−0,2513) por solo 0,006. Con un η0 de 0,002 o más, la orientación puede quedar al revés en alguna semilla y el índice completo se da vuelta (es lo que muestra el desplazamiento de ranking de 49 en la tabla).

![Pesos antes y después de orientar, por semilla](../../runs/oja_sweep/sign_ambiguity.png)

### Qué mide el índice

**Un índice alto es "engagement profundo con audiencia chica": epígrafes largos, guardados, comentarios, visitas al perfil y tutoriales. Un índice bajo es "audiencia grande, video corto de tendencia y precio alto".**

Lo dicen los signos y las magnitudes de los pesos (`runs/oja/weights_vs_pc1.csv`, ya orientados):

| Lado | Variables (peso de Oja) |
|------|-------------------------|
| Positivo (suma al índice) | `average_caption_words` +0,2574 (el mayor), `saves_per_1000` +0,2309, `profile_visits_per_1000` +0,2130, `comments_per_1000` +0,2100, `average_watch_time_seconds` +0,2015, `content_tutorial_pct` +0,2008 |
| Negativo (resta) | `average_impressions` −0,2518 (el más negativo), `average_video_views` −0,2507, `average_reach` −0,2495, `trend_posts_pct` −0,2475, `video_posts_pct` −0,2420, `short_form_posts_pct` −0,2403, `followers` −0,2386, `price_usd` −0,2385 |

- Es el mismo eje que PC1 de 1.1 (de "audiencia grande y video corto de tendencia" a "audiencia chica, epígrafes largos y contenido didáctico") y el mismo que ordena las regiones del mapa: `pc1_region_eta2` = 0,8314 en 1.2. Los 20 primeros del índice son todos de la región 1 y los cinco últimos de la 3.
- **Qué no mide.** Pesos casi nulos: `argentina_audience_pct` +0,0333, `monthly_posts` −0,0366 y `sponsored_posts_pct` −0,0168. Los pesos de los tópicos son chicos (−0,0666 fútbol, −0,0646 maquillaje, +0,1344 perros; arte no entra, se reconstruye). O sea, el índice casi no sabe de qué habla el creador ni dónde está su audiencia, y de perros sabe solo un poco. **No es un puntaje de calidad ni de "mejor creador":** mezcla tamaño y estilo, y un creador grande, caro y de video corto queda abajo por definición.

### ¿Conviene elegir al creador con el índice más alto?

**No: el índice ordena tamaño y estilo, no tópico ni objetivo, así que el primero del ranking no es el indicado para cualquier campaña.**

Los cinco primeros y los cinco últimos de `runs/oja/ranking.csv` (participación de cada tópico en %, región del mapa de 1.2):

| Rango | Creador | Fútbol | Maquillaje | Perros | Arte | Followers | Región | Índice |
|-------|---------|--------|------------|--------|------|-----------|--------|--------|
| 1 | `@mundoanimal.ar` | 6,75 | 12,31 | **74,21** | 6,73 | 17062 | 1 | 6,7680 |
| 2 | `@consejoscaninos` | 4,44 | 6,76 | **82** | 6,8 | 9604 | 1 | 6,4849 |
| 3 | `@huellasyconsejos` | 10,68 | 9,54 | **64,49** | 15,29 | 13235 | 1 | 6,1537 |
| 4 | `@ideasconhuella` | 14,05 | 13,35 | **55,6** | 17 | 34822 | 1 | 5,1874 |
| 5 | `@perrosfelices.ar` | 7,72 | 13,46 | **70,74** | 8,08 | 33338 | 1 | 4,3959 |
| 46 | `@crearenreels` | 14,86 | 9,93 | **66,99** | 8,22 | 551553 | 3 | −5,4766 |
| 47 | `@golconmate` | **82,18** | 5,89 | 4,16 | 7,77 | 745025 | 3 | −5,5378 |
| 48 | `@brillourbano` | 11,6 | **70,43** | 11,04 | 6,93 | 638718 | 3 | −5,6792 |
| 49 | `@galeriacotidiana` | 11,81 | 13,57 | 13,58 | **61,04** | 381324 | 3 | −5,8391 |
| 50 | `@futbolsinvueltas` | **63,36** | 10,44 | 10,1 | 16,1 | 493194 | 3 | −6,4065 |

- **El tope es de un solo perfil.** Los cinco primeros son de perros, chicos (de 9604 a 34822 followers) y de la región 1; en los diez primeros hay 7 de perros, 2 de fútbol y 1 de maquillaje. Un cliente de fútbol, de maquillaje o de arte recibiría un creador con a lo sumo 14,05% de fútbol, 13,46% de maquillaje o 17% de arte entre los cinco primeros. El tópico no está en el eje: es lo mismo que vimos en 1.1, donde PC1 no separa tópicos (dispersión de 2,45 a 3,72 dentro de cada uno) y separa tamaño y estilo.
- **Abajo no está "lo malo", está lo grande.** Los cinco últimos cubren los cuatro tópicos y son todos de la región 3 (alcance masivo). `@golconmate` tiene la mayor participación de fútbol de todo el catálogo (82,18%) y `@galeriacotidiana` un 61,04% de arte con 84,01% de audiencia argentina, y las dos reglas de 1.4 los eligen. Un índice bajo solo dice "audiencia grande, video de tendencia, caro".
- **Un solo índice no puede expresar el objetivo del cliente.** Un refugio de perros quiere audiencia local y señales de consulta; una marca de ropa quiere alcance argentino barato. Ninguna de esas cosas es el eje de PC1: el peso del índice sobre `argentina_audience_pct` es de apenas +0,0333.
- **Qué hicieron las reglas de 1.4 con el primero del ranking.** `oja_top_creator` es `@mundoanimal.ar` y `oja_top_creator_picked_for` es `[]` (la lista vacía): ningún cliente lo eligió. Es candidato de perros (74,21%, región 1) pero tiene la audiencia argentina más baja de los 9 candidatos (32,89%), con 65,40 USD por mil de alcance argentino (`runs/campaigns/evidence.csv`). Las elecciones de perros son los rangos 4, 3 y 2 del índice, pero las de los otros tres clientes llegan hasta el rango 49 (`@galeriacotidiana`).
- **Cómo sí se puede usar el índice:** como una evidencia más dentro de una regla propia de cada cliente (1.4 lo muestra al lado de la posición en el biplot y la región del mapa), no como un ranking único de campaña.

### Supuestos (1.3)

- **Inicialización y orden:** pesos uniformes en [0, 1) y una permutación de los creadores por época, con la semilla 42; las otras dos semillas del barrido dan el mismo vector hasta el signo (`runs/oja_sweep/sign_ambiguity.png`).
- **Banda de la norma:** "converge a 1" significa que la norma queda a menos de `norm_tol` = 0,01 de 1 (con η constante la norma se estabiliza un poco por encima: 1,0015). Es un umbral nuestro.
- **Escala del índice:** el vector se normaliza a largo 1, así que el índice está en las mismas unidades que PC1 (desvíos z sumados con pesos de largo 1). Un índice de 6,77 no es una probabilidad ni un puntaje de 0 a 100.
- **Regla de selección de η0 sin PC1:** los umbrales (banda de 0,01 y desplazamiento de ranking cero entre semillas) son del plan y el grupo los tiene que aceptar; con otros valores podría ganar 0,0001 o 0,001.
- **Signo:** coeficiente de mayor valor absoluto positivo, como en 1.1; frágil por la ventaja de 0,006 entre dos variables.
- **Lectura del índice:** se apoya solo en los signos y magnitudes de los pesos y en las posiciones de los creadores, no afirma nada que esas tablas no muestren.
- Los gráficos se enlazan desde `runs/` (se regeneran, no se versionan). Copiarlos a `docs/` queda para la fase de documentación.

### Cómo regenerar (1.3)

```bash
.venv/bin/python -m tp4 run configs/oja_sweep.json
.venv/bin/python -m tp4 run configs/oja.json
.venv/bin/python -m tp4 run configs/campaigns.json
```

Verificación de la evidencia (los 18 valores de 1.1 y 1.2 más los de 1.3):

```bash
.venv/bin/python -m tp4.findings docs/hallazgos/ej1.md \
  --cite runs/pca/summary.json:pc1_explained_variance_pct \
  --cite runs/pca/summary.json:pc2_explained_variance_pct \
  --cite runs/pca/summary.json:cumulative_pc2_pct \
  --cite runs/pca/summary.json:pc1_top_feature \
  --cite runs/pca_scaling_sweep/standardize-false/summary.json:pc1_explained_variance_pct \
  --cite runs/eda/summary.json:n_selected_features \
  --cite runs/som_sweep/summary.json:selected_child \
  --cite runs/som_sweep/summary.json:selected_quantization_error \
  --cite runs/som_sweep/summary.json:selected_topographic_error \
  --cite runs/som_sweep/summary.json:selected_distance_preservation \
  --cite runs/som_design_sweep/summary.json:selected_child \
  --cite runs/som/summary.json:n_regions \
  --cite runs/som/summary.json:u_border_ratio \
  --cite runs/som/summary.json:pc1_region_eta2 \
  --cite runs/som/summary.json:pc2_region_eta2 \
  --cite runs/som/summary.json:empty_neuron_frac \
  --cite runs/som_pca_control/summary.json:region_ari \
  --cite runs/som_pca_control/summary.json:bmu_distance_agreement \
  --cite runs/oja/summary.json:cosine_to_pc1 \
  --cite runs/oja/summary.json:pearson_scores_pc1 \
  --cite runs/oja/summary.json:final_w_norm \
  --cite runs/oja/summary.json:epochs_to_converge \
  --cite runs/oja/summary.json:top_creator \
  --cite runs/oja/summary.json:bottom_creator \
  --cite runs/oja/summary.json:oja_top_feature \
  --cite runs/oja/summary.json:lambda1 \
  --cite runs/oja/summary.json:course_eta_bound \
  --cite runs/oja_sweep/summary.json:selected_eta0 \
  --cite runs/oja_sweep/summary.json:selected_child \
  --cite runs/campaigns/summary.json:oja_top_creator_picked_for
```

## 1.4 Decisión para las campañas

El enunciado pide interpretar los resultados "a lo largo de todo el trabajo" y, con esa evidencia, tomar decisiones para los cuatro clientes. La agencia hoy elige por cantidad de seguidores y sus campañas "no generan ingresos suficientes para recuperar la inversión": las publicaciones tienen visibilidad pero no llegan al público al que están dirigidos los productos. Los refugios quieren impulsar consultas y adopciones; los otros tres clientes, aumentar las ventas. Todo lo de esta sección sale de `runs/campaigns/` (más `runs/oja/`, `runs/som/` y `runs/pca/`), regenerados con los comandos de "Cómo regenerar (1.4)". **No hay datos de ventas ni de conversión**: "mejor que elegir por seguidores" se mide con proxies (participación del tópico, audiencia argentina, alcance argentino estimado, engagement, precio y costo por mil de alcance argentino), que están en "Supuestos (1.4)".

### Método de selección

**Una regla transparente por cliente, no un índice único: un filtro de tópico, opcionalmente una región del mapa, y una suma ponderada de puntajes z, y para cada elegido se muestran sus tres evidencias (PCA, mapa y Oja).**

![Biplot de PC1 y PC2](../../runs/pca/biplot.png)

![Regiones del mapa con los creadores](../../runs/som/region_map.png)

![Planos de componentes](../../runs/som/component_planes.png)

- **Por qué por cliente.** La conclusión de 1.3 es que el índice de Oja es el eje de tamaño y estilo (el mismo PC1 de 1.1) y no sabe de tópico ni de audiencia argentina, y que el primero del ranking no fue elegido por ningún cliente. Cada cliente tiene un objetivo distinto, así que la regla lo escribe explícito.
- **Forma de la regla** (`configs/campaigns.json`, un cliente por entrada): (1) candidatos: los creadores cuya columna de tópico (`topic_*_pct`, nunca el nombre de la cuenta: `@crearenreels` y `@makeupconsejos` engañan) llega a al menos 40%; (2) opcional, una región del mapa de Kohonen (`regions`); (3) puntaje = suma de `peso · z(columna)` sobre las columnas elegidas, con z calculado sobre los 50 creadores (desvío poblacional), de modo que el puntaje no depende del filtro; (4) los tres mejores; los empates se resuelven en el orden del catálogo.
- **Candidatos por cliente** (`dogs_n_candidates`, etc. en `runs/campaigns/summary.json`): perros 9 (tópico y región 1), fútbol 14, maquillaje 11, arte 11.
- **La evidencia se recalcula, no se copia:** `runs/campaigns/evidence.csv` recalcula PC1 y PC2, la celda y región del mapa y el índice de Oja desde las mismas configs (`configs/oja.json` y `configs/som.json`), y un test verifica que coinciden con `runs/oja/ranking.csv` y `runs/som/bmus.csv` para los 50 creadores.
- **N = 3** por cliente (`n_picks`). Si un creador queda elegido por dos clientes se marca en `picks.csv` (`also_picked_for`) y en el resumen: `n_overlapping_creators` = 0, `overlapping_creators` = `[]`. Ningún creador sale elegido por dos clientes.
- Cada tabla de abajo muestra por elegido: posición en el biplot (PC1 / PC2 de `runs/campaigns/picks.csv`), región y celda del mapa (fila y columna de 0 a 7) con la lectura de los planos de componentes (el peso de esa neurona en unidades z, de `runs/som/weights.csv`) y el índice de Oja con su rango entre los 50 (`runs/campaigns/evidence.csv`).

### Refugios de perros

**Objetivo:** impulsar consultas y adopciones. Hacen falta creadores de perros con audiencia argentina (la adopción es local) y señales de que la comunidad escribe, guarda y visita el perfil.

- **Regla** (`dogs_rule`): `topic_dogs_pct >= 40; región del mapa en [1]; puntaje = +1·z(argentina_audience_pct) +1·z(profile_visits_per_1000) +0,5·z(comments_per_1000) +0,5·z(saves_per_1000)`.
- **Por qué esas columnas:** la audiencia argentina porque se adopta en el país; visitas al perfil, comentarios y guardados como proxy de consulta e interés real. **Por qué región 1:** en 1.2 la región 1 es "profundidad educativa y comunidad cercana" (epígrafes largos, guardados y comentarios altos, audiencia chica): el perfil del que consulta. Es una decisión nuestra, discutida en "Supuestos (1.4)".

**Elegidos:** `@ideasconhuella`, `@huellasyconsejos` y `@consejoscaninos`.

| Creador | PC1 / PC2 | Región y celda | Lectura de la celda (peso z) | Índice de Oja (rango) |
|---------|-----------|----------------|------------------------------|-----------------------|
| `@ideasconhuella` | 5,19 / 0,83 | 1, fila 2 columna 0 | tutoriales +1,7, perros +1,5, guardados +1,3; video −1,1 | 5,1874 (4) |
| `@huellasyconsejos` | 6,16 / 2,70 | 1, fila 0 columna 0 | perros +1,8, epígrafes +1,7, guardados +1,6; `likes` −2,5, video corto −2,1 | 6,1537 (3) |
| `@consejoscaninos` | 6,49 / 3,13 | 1, fila 0 columna 0 | perros +1,8, epígrafes +1,7, guardados +1,6; `likes` −2,5, video corto −2,1 | 6,4849 (2) |

- `@ideasconhuella` (55,6% perros, 83,11% de audiencia argentina, 27,00 visitas al perfil por mil, 70,52% de contenido de tutoriales): el de más visitas y más didáctico de los tres, aunque el más caro por mil de alcance argentino (47,29 USD).
- `@huellasyconsejos` (64,49% perros): la mayor audiencia argentina de todo el catálogo (94,43%), 23,71 visitas por mil y 75 USD por publicación.
- `@consejoscaninos` (82% perros, la mayor participación de perros): 86,63% de audiencia argentina, 22,68 visitas por mil y 75 USD, en la misma celda que el anterior.
- Los tres están en el extremo de PC1 positivo (5,19 a 6,49) y de PC2 positivo (0,83 a 3,13): audiencia chica, epígrafes largos y audiencia local, justo la zona que 1.1 describe para perros. En el índice de Oja son los rangos 4, 3 y 2: acá el índice y la regla coinciden, pero no con el rango 1. El primero (`@mundoanimal.ar`) queda fuera de los tres por su audiencia argentina de 32,89% (ver 1.3).

### Ropa de entrenamiento de fútbol

**Objetivo:** aumentar las ventas de ropa de entrenamiento. Hacen falta creadores de fútbol con alcance argentino barato (se compra en el país) y alguna señal de interés por el perfil.

- **Regla** (`football_rule`): `topic_football_pct >= 40; puntaje = -1·z(cost_per_1000_arg_reach) +0,5·z(argentina_audience_pct) +0,5·z(profile_visits_per_1000)`; sin filtro de región.
- **Por qué esas columnas:** una venta pide llegar a muchos argentinos por poco dinero (el costo por mil de alcance argentino pesa −1) y que sean argentinos (audiencia argentina) con interés (visitas al perfil).

**Elegidos:** `@futbolcononda`, `@golconmate` y `@lookdecancha`.

| Creador | PC1 / PC2 | Región y celda | Lectura de la celda (peso z) | Índice de Oja (rango) |
|---------|-----------|----------------|------------------------------|-----------------------|
| `@futbolcononda` | −0,18 / −0,36 | 2, fila 7 columna 3 | fútbol +1,8, alcance a no seguidores +1,1, audiencia argentina +0,8 | −0,1879 (26) |
| `@golconmate` | −5,53 / 0,38 | 3, fila 6 columna 6 | vistas +1,8, alcance +1,7, impresiones +1,7; epígrafes −1,3 | −5,5378 (47) |
| `@lookdecancha` | −2,80 / −0,57 | 2, fila 7 columna 4 | fútbol +1,7, alcance a no seguidores +1,2, entretenimiento +1,0; reseñas −1,0 | −2,7962 (37) |

- `@futbolcononda` (73,46% fútbol): 89,37% de audiencia argentina, 18,26 USD por mil de alcance argentino, 76,41% del alcance viene de no seguidores.
- `@golconmate` (82,18% fútbol, la mayor del catálogo): el más barato de los tres (15,63 USD por mil), 77,46% de audiencia argentina y un alcance argentino estimado de 207994, a 3251,20 USD por publicación.
- `@lookdecancha` (69,29% fútbol): 77,76% de audiencia argentina y el mayor alcance a no seguidores (81,96%), pero el más caro por mil de los tres (33,26 USD).

**¿Alcance masivo, descubrimiento o profundidad educativa?** Los números por región de `runs/campaigns/strategy_profiles.csv` (los creadores de fútbol con al menos 40%) dicen que los elegidos encarnan **descubrimiento con alcance masivo, no profundidad educativa**, y que eso es lo que le sirve a una marca que quiere vender.

| Región (estrategia) | Creadores de fútbol | Alcance argentino estimado medio | Audiencia argentina | Alcance a no seguidores | USD por mil de alcance argentino | Guardados por mil | Visitas al perfil por mil | Tutoriales (%) |
|---------------------|---------------------|----------------------------------|---------------------|--------------------------|----------------------------------|-------------------|----------------------------|----------------|
| 3 (alcance masivo) | 3 | 129626 | 57,65 | 71,46 | 27,55 | 7,01 | 12,70 | 7,22 |
| 2 (descubrimiento) | 6 | 49495 | 49,06 | 75,81 | 61,95 | 12,03 | 11,66 | 13,82 |
| 1 (profundidad educativa) | 5 | 7296 | 41,00 | 46,42 | 55,40 | 18,48 | 20,36 | 42,56 |

- **Profundidad educativa (región 1)** es la que más guarda (18,48), visita el perfil (20,36) y enseña (42,56% de tutoriales), pero llega a 7296 argentinos por creador, solo 46,42% del alcance es de no seguidores y cuesta 55,40 USD por mil: es comunidad, no ventas masivas.
- **Alcance masivo (región 3)** llega 129626 argentinos por creador al menor costo por mil (27,55 USD) y con 71,46% de no seguidores, pero solo hay 3 creadores de fútbol ahí, con pocos guardados y tutoriales (7,01 y 7,22%): el alcance es amplio y superficial.
- **Descubrimiento (región 2)** tiene el mayor alcance a no seguidores (75,81%) pero es la más cara por mil (61,95 USD) y su audiencia argentina es la segunda más baja de las tres (49,06%, contra 41,00% de la región 1).
- **Qué hace la regla.** No elige una región: elige el alcance argentino más barato dentro de los candidatos y quedó con dos creadores de la región 2 y uno de la 3. Entre ellos hay audiencia argentina de 77,46 a 89,37% (media 81,53%) contra medias de 41,00 a 57,65% en las regiones, y un costo medio de 19,91 USD por mil contra 27,55 a 61,95. Los tres tienen entre 59,70 y 81,96% de alcance a no seguidores. Para una marca de ropa que busca ventas, el descubrimiento de gente nueva que además es argentina y barata de alcanzar pesa más que la profundidad educativa. Es un argumento de costo y alcance, no de conversión: no medimos compra.
- **Límites.** La región 3 tiene 3 creadores y la 2, 6: son muestras chicas. El costo de la región 2 como grupo (61,95) es mucho mayor que el de sus elegidos (18,26 y 33,26) porque la regla toma los más baratos de la región; si el cliente quisiera profundidad, habría que cambiar los pesos de la regla.

### Productos de belleza

**Objetivo:** aumentar las ventas de productos de belleza. Mismo esquema de ventas que fútbol, con los guardados como señal de contenido (asumimos que guardar indica intención de volver al producto).

- **Regla** (`makeup_rule`): `topic_makeup_pct >= 40; puntaje = -1·z(cost_per_1000_arg_reach) +0,5·z(argentina_audience_pct) +0,5·z(saves_per_1000)`; sin filtro de región.

**Elegidos:** `@reelargentino`, `@brochasymates` y `@maquillajereal`.

| Creador | PC1 / PC2 | Región y celda | Lectura de la celda (peso z) | Índice de Oja (rango) |
|---------|-----------|----------------|------------------------------|-----------------------|
| `@reelargentino` | 4,31 / 0,11 | 1, fila 6 columna 0 | reseñas +1,1, maquillaje +1,0, epígrafes +1,0; `shares` −1,2 | 4,3055 (6) |
| `@brochasymates` | 0,05 / 0,82 | 2, fila 0 columna 5 | tutoriales +1,3, maquillaje +1,0, publicaciones +0,7 | 0,0558 (25) |
| `@maquillajereal` | −4,94 / 2,22 | 3, fila 5 columna 7 | seguidores +2,3, alcance +2,0; comentarios −2,1, guardados −1,5 | −4,9424 (44) |

- `@reelargentino` (59,62% maquillaje): chico (18000 seguidores, 172,53 USD por publicación), 79,55% de audiencia argentina y 19,69 guardados por mil, el que más guarda de los tres; pero el más caro por mil (36,98 USD).
- `@brochasymates` (58,24% maquillaje): la mayor audiencia argentina de los tres (88,23%), 24,91 USD por mil y 44,87% de contenido de tutoriales.
- `@maquillajereal` (66,74% maquillaje): el más barato por mil (17,86 USD) y el más grande de los tres (608003 seguidores), pero con pocos guardados (5,99 por mil) y la audiencia argentina más baja de los tres (69,15%).
- Los tres cubren las tres regiones: es un portafolio que mezcla comunidad (región 1), descubrimiento (2) y alcance (3). El índice de Oja los pone en los rangos 6, 25 y 44: un ranking por índice no habría armado esta combinación.

### Materiales de arte

**Objetivo:** aumentar las ventas de materiales de arte. Mismo esquema, con el contenido tutorial como señal (se aprende y se compra el material). **Nota:** el arte usa `topic_art_pct`, que no entra a los modelos (en 1.1 se descartó por ser el complemento de los otros tres tópicos) y se reconstruye como 100 menos fútbol, maquillaje y perros.

- **Regla** (`art_rule`): `topic_art_pct >= 40; puntaje = -1·z(cost_per_1000_arg_reach) +0,5·z(argentina_audience_pct) +0,5·z(content_tutorial_pct)`; sin filtro de región.

**Elegidos:** `@colorenmovimiento`, `@galeriacotidiana` y `@rinconcreativo`.

| Creador | PC1 / PC2 | Región y celda | Lectura de la celda (peso z) | Índice de Oja (rango) |
|---------|-----------|----------------|------------------------------|-----------------------|
| `@colorenmovimiento` | 1,19 / −0,49 | 2, fila 1 columna 4 | celda de transición, ningún peso pasa de ±1,1: `shares` +0,7, video +0,7; patrocinadas −1,1 | 1,1935 (21) |
| `@galeriacotidiana` | −5,84 / 1,36 | 3, fila 7 columna 7 | impresiones +2,1, alcance +2,0, vistas +2,0; reseñas −1,7 | −5,8391 (49) |
| `@rinconcreativo` | −0,35 / 0,13 | 2, fila 2 columna 3 | celda de transición, ningún peso pasa de ±1,0: tiempo de visualización +0,4; patrocinadas −1,0 | −0,3496 (28) |

- `@colorenmovimiento` (68,49% arte): 71,15 de engagement por mil, 53,78% de tutoriales y 72,08% de audiencia argentina; pero el más caro por mil de los tres (36,53 USD).
- `@galeriacotidiana` (61,04% arte): 84,01% de audiencia argentina y 26,03 USD por mil con un alcance argentino estimado de 186931; casi no tiene tutoriales (0,77%): entra por costo y audiencia, no por contenido, y cuesta 4866,57 USD por publicación.
- `@rinconcreativo` (64,09% arte): 30,05% de tutoriales, 27,93 USD por mil, pero la audiencia argentina más baja de los tres (50,74%, cerca de la media de 52,64% del catálogo).
- El arte no tiene un rincón propio en el mapa (1.2): los planos de las celdas de los elegidos son suaves (el plano reconstruido de arte en esas celdas es 28,4%, 34,5% y 45,3%, contra el máximo de 63,3% del mapa), así que la región ayuda poco y la regla se apoya sobre todo en el costo y la audiencia argentina. Es una limitación del arte en este dataset.

### Comparación con una selección solo por seguidores

**La regla le gana a los seguidores en audiencia argentina en los cuatro clientes y en costo por mil en tres (fútbol, maquillaje y arte), pero llega a menos gente en total en todos, y para los refugios el costo por mil de los más seguidos del tópico es más bajo.**

![Regla contra seguidores, por cliente](../../runs/campaigns/comparison.png)

Se comparan tres selecciones de tres creadores por cliente (`runs/campaigns/comparison.csv`; participación media del tópico en %, audiencia argentina media en %, alcance argentino estimado total, engagement medio por mil, precio total en USD y USD por mil de alcance argentino):

| Cliente | Estrategia | Participación del tópico | Audiencia argentina | Alcance argentino estimado | Engagement por mil | Precio total | USD por mil |
|---------|------------|--------------------------|---------------------|----------------------------|--------------------|--------------|-------------|
| Perros | método | 67,36 | 88,06 | 9138 | 57,09 | 259,49 | 28,40 |
| Perros | seguidores global | 9,48 | 63,48 | 482578 | 64,61 | 15896,60 | 32,94 |
| Perros | seguidores en el tópico | 66,04 | 57,53 | 315276 | 73,36 | 6992,95 | 22,18 |
| Fútbol | método | 74,98 | 81,53 | 364307 | 69,08 | 7252,87 | 19,91 |
| Fútbol | seguidores global | 34,56 | 63,48 | 482578 | 64,61 | 15896,60 | 32,94 |
| Fútbol | seguidores en el tópico | 68,88 | 57,65 | 388879 | 70,73 | 10712,71 | 27,55 |
| Belleza | método | 61,53 | 78,98 | 223277 | 66,53 | 4456,20 | 19,96 |
| Belleza | seguidores global | 45,97 | 63,48 | 482578 | 64,61 | 15896,60 | 32,94 |
| Belleza | seguidores en el tópico | 66,25 | 60,71 | 439539 | 62,54 | 15592,28 | 35,47 |
| Arte | método | 64,54 | 68,94 | 240126 | 75,12 | 6484,76 | 27,01 |
| Arte | seguidores global | 10,00 | 63,48 | 482578 | 64,61 | 15896,60 | 32,94 |
| Arte | seguidores en el tópico | 67,16 | 57,51 | 288999 | 74,48 | 9717,03 | 33,62 |

- **Los más seguidos dentro del tópico de cada cliente** (`*_baseline_topic`, `runs/campaigns/baselines.csv`): perros `@crearenreels`, `@patitasenred` y `@patitascontentas`; fútbol `@golconmate`, `@futbolsinvueltas` y `@tribunacreativa`; belleza `@mateconideas`, `@brillourbano` y `@maquillajereal`; arte `@galeriacotidiana`, `@pincelcotidiano` y `@artequeinspira`. Comparten un creador con los elegidos del método en fútbol (`@golconmate`), belleza (`@maquillajereal`) y arte (`@galeriacotidiana`), y ninguno en perros.
- **Los más seguidos del catálogo son los mismos para todos los clientes** (`followers_global_picks`): `@mateconideas`, `@golconmate` y `@brillourbano`, a 32,94 USD por mil (`followers_global_cost_per_1000_arg_reach`). Para cada cliente dan una participación del tópico de 9,48% (perros), 34,56% (fútbol), 45,97% (belleza) y 10,00% (arte). `@brillourbano` tiene solo 29,78% de audiencia argentina y cuesta 90,28 USD por mil. Para fútbol, `@golconmate` está también entre los elegidos del método.
- **Ventajas de los seguidores (los dos baselines):** es simple, no necesita ningún modelo ni mantenimiento y tiene el mayor alcance bruto: 482578 de alcance argentino estimado contra 9138 a 364307 del método. Dentro del tópico, los más seguidos pueden salir más baratos por mil cuando la cuenta grande es local: en perros 22,18 contra 28,40 USD, con un alcance de 315276 contra 9138, porque el baseline incluye a `@crearenreels` (88,61% de audiencia argentina, 16,31 USD por mil). En belleza y arte, el baseline dentro del tópico tiene una participación un poco mayor (66,25 contra 61,53; 67,16 contra 64,54), porque la regla cambia participación por costo.
- **Desventajas de los seguidores:** ignora el tópico (el top global tiene 9,48% y 10,00% de participación para perros y arte), ignora la audiencia local (63,48% el global y de 57,51 a 60,71% el del tópico, contra 68,94 a 88,06% del método) y el costo por mil no baja con los seguidores: tres de los cuatro clientes pagan menos con el método. La cantidad de seguidores no informa sobre la audiencia argentina: la correlación entre `followers` y `argentina_audience_pct` en los 50 creadores es −0,001 (Pearson, calculada sobre `runs/campaigns/evidence.csv`).
- **Ventajas del método:** audiencia argentina más alta en los cuatro (88,06; 81,53; 78,98; 68,94), costo por mil menor en fútbol (19,91), belleza (19,96) y arte (27,01), participación del tópico mayor en perros y fútbol que la de cualquier selección por seguidores, y cada elegido viene con su evidencia (PCA, mapa y Oja).
- **Desventajas del método:** hace falta un pipeline mantenido (EDA, PCA, mapa, Oja, campañas), las reglas (pesos, umbral de 40%, región) son supuestos nuestros, llega a menos gente en total (para los refugios el alcance argentino estimado total es de apenas 9138), y para perros el costo por mil es más alto que el del baseline dentro del tópico: filtrar por la región 1 sacó a `@crearenreels` (región 3). El engagement medio del método es menor al del baseline en perros (57,09 contra 73,36) y fútbol (69,08 contra 70,73).
- **No hay datos de ventas:** la comparación usa los proxies del plan, y que el método gane en esos proxies no prueba que venda más. Es evidencia para decidir con criterio, no una medición del retorno.

### Mantenimiento del sistema

**Una campaña nueva no necesita reentrenar nada; nuevos creadores o variables y la actualización mensual sí piden repetir todo el pipeline en orden y releer las reglas.**

- **Nueva campaña:** agregar un cliente a `clients` en `configs/campaigns.json` (`name`, `label`, `objective`, `topic`, `min_share`, `regions`, `rank_by`) y correr solo `python -m tp4 run configs/campaigns.json`. Reusa el índice de Oja y el mapa ya entrenados (se recalculan desde sus configs, con la misma semilla); no hay reentrenamiento de modelos.
- **Nuevos creadores o nuevas variables:** actualizar `datos/creadores.csv` (y `configs/feature_set.json` para variables) y repetir en orden: EDA, PCA, barridos del mapa (diseño y numérico), copiar la selección a `configs/som.json` y correr el comando de verificación, el mapa, el control, el barrido de Oja (copiar `selected_eta0` a `configs/oja.json`), Oja y campañas. Después **releer las reglas**: los números de región y sus perfiles (el filtro `regions: [1]` de perros se escribió para el mapa actual y una renumeración que conserve `k` no se detecta; un número de región mayor que `k` sí falla con error), los umbrales de tópico, que las columnas siguen siendo las mismas, y `RANKABLE_COLUMNS` en `tp4/analysis/campaigns.py` si una variable nueva se quiere usar para rankear (es una lista permitida: agregarla es un cambio de código).
- **Actualización mensual:** el mismo reproceso completo con las mismas semillas. Los rankings, las regiones e incluso `k` pueden cambiar, así que hay que releer 1.2 a 1.4. No hay arranque en caliente ni entrenamiento incremental: Oja y el mapa se vuelven a entrenar desde cero, lo que con 50 creadores lleva segundos, y el refresco mensual no está automatizado (se corre a mano). Una selección con presupuesto (elegir creadores hasta agotar un monto) sería una extensión posible, no existe en el código.

```bash
# Nueva campaña (sin reentrenar)
.venv/bin/python -m tp4 run configs/campaigns.json

# Nuevos creadores, nuevas variables o actualización mensual (todo, en orden)
.venv/bin/python -m tp4 run configs/eda.json
.venv/bin/python -m tp4 run configs/pca.json
.venv/bin/python -m tp4 run configs/pca_scaling_sweep.json
.venv/bin/python -m tp4 run configs/som_design_sweep.json
.venv/bin/python -m tp4 run configs/som_sweep.json
# copiar la selección de runs/som_sweep a configs/som.json; después:
.venv/bin/python -m tp4 run configs/som.json
.venv/bin/python -m tp4 run configs/som_pca_control.json
.venv/bin/python -m tp4.experiments.som_sweep_report check configs/som.json runs/som_design_sweep runs/som_sweep --run runs/som
.venv/bin/python -m tp4 run configs/oja_sweep.json
# copiar selected_eta0 de runs/oja_sweep/summary.json a configs/oja.json; después:
.venv/bin/python -m tp4 run configs/oja.json
.venv/bin/python -m tp4 run configs/campaigns.json
```

### Supuestos (1.4)

- **Proxies en vez de ventas:** no existen datos de ventas ni de conversión. "Mejor" significa más participación del tópico, más audiencia argentina, más alcance argentino estimado, más engagement y menor costo por mil.
- **Una publicación por creador:** se asume una publicación patrocinada por creador al precio `price_usd`, que llega a `average_reach` cuentas, de las cuales `argentina_audience_pct` son argentinas (`est_arg_reach`).
- **Costo agrupado:** USD por mil de alcance argentino = suma de precios dividida por la suma de alcances argentinos estimados, por mil (no el promedio de los cocientes de cada creador).
- **Pesos y umbrales:** los pesos de cada cliente, el 40% mínimo de tópico (inclusivo) y el filtro de región 1 para perros son elecciones nuestras que el grupo todavía tiene que aceptar; las audiencias y los costos citados salen de los datos, pero cambiar un peso puede cambiar los elegidos.
- **Numeración de regiones:** `regions: [1]` se refiere al mapa actual (`k` = 3); si el mapa cambia, hay que revisar el filtro a mano.
- **Superposición permitida:** un creador puede ser elegido por dos clientes (se marca); hoy no pasa.
- **Arte reconstruido:** `topic_art_pct` se reconstruye como 100 menos los otros tres tópicos, igual que en 1.1.
- **Alcance de la interpretación:** las lecturas de región y de celda salen de los planos de componentes y de los perfiles de 1.2; no afirmamos nada que esas figuras y tablas no muestren. Una selección con presupuesto, el reentrenamiento incremental y un refresco mensual automático no están construidos: son extensiones posibles.
- Los gráficos se enlazan desde `runs/` (se regeneran, no se versionan). Copiarlos a `docs/` queda para la fase de documentación.

### Cómo regenerar (1.4)

```bash
.venv/bin/python -m tp4 run configs/oja_sweep.json
.venv/bin/python -m tp4 run configs/oja.json
.venv/bin/python -m tp4 run configs/campaigns.json
```

Verificación de la evidencia (los 30 valores de 1.1 a 1.3 más los de 1.4):

```bash
.venv/bin/python -m tp4.findings docs/hallazgos/ej1.md \
  --cite runs/pca/summary.json:pc1_explained_variance_pct \
  --cite runs/pca/summary.json:pc2_explained_variance_pct \
  --cite runs/pca/summary.json:cumulative_pc2_pct \
  --cite runs/pca/summary.json:pc1_top_feature \
  --cite runs/pca_scaling_sweep/standardize-false/summary.json:pc1_explained_variance_pct \
  --cite runs/eda/summary.json:n_selected_features \
  --cite runs/som_sweep/summary.json:selected_child \
  --cite runs/som_sweep/summary.json:selected_quantization_error \
  --cite runs/som_sweep/summary.json:selected_topographic_error \
  --cite runs/som_sweep/summary.json:selected_distance_preservation \
  --cite runs/som_design_sweep/summary.json:selected_child \
  --cite runs/som/summary.json:n_regions \
  --cite runs/som/summary.json:u_border_ratio \
  --cite runs/som/summary.json:pc1_region_eta2 \
  --cite runs/som/summary.json:pc2_region_eta2 \
  --cite runs/som/summary.json:empty_neuron_frac \
  --cite runs/som_pca_control/summary.json:region_ari \
  --cite runs/som_pca_control/summary.json:bmu_distance_agreement \
  --cite runs/oja/summary.json:cosine_to_pc1 \
  --cite runs/oja/summary.json:pearson_scores_pc1 \
  --cite runs/oja/summary.json:final_w_norm \
  --cite runs/oja/summary.json:epochs_to_converge \
  --cite runs/oja/summary.json:top_creator \
  --cite runs/oja/summary.json:bottom_creator \
  --cite runs/oja/summary.json:oja_top_feature \
  --cite runs/oja/summary.json:lambda1 \
  --cite runs/oja/summary.json:course_eta_bound \
  --cite runs/oja_sweep/summary.json:selected_eta0 \
  --cite runs/oja_sweep/summary.json:selected_child \
  --cite runs/campaigns/summary.json:oja_top_creator_picked_for \
  --cite runs/campaigns/summary.json:dogs_picks \
  --cite runs/campaigns/summary.json:football_picks \
  --cite runs/campaigns/summary.json:makeup_picks \
  --cite runs/campaigns/summary.json:art_picks \
  --cite runs/campaigns/summary.json:followers_global_picks \
  --cite runs/campaigns/summary.json:dogs_baseline_topic \
  --cite runs/campaigns/summary.json:football_baseline_topic \
  --cite runs/campaigns/summary.json:makeup_baseline_topic \
  --cite runs/campaigns/summary.json:art_baseline_topic \
  --cite runs/campaigns/summary.json:dogs_cost_per_1000_arg_reach \
  --cite runs/campaigns/summary.json:football_cost_per_1000_arg_reach \
  --cite runs/campaigns/summary.json:makeup_cost_per_1000_arg_reach \
  --cite runs/campaigns/summary.json:art_cost_per_1000_arg_reach \
  --cite runs/campaigns/summary.json:followers_global_cost_per_1000_arg_reach \
  --cite runs/campaigns/summary.json:dogs_mean_topic_share \
  --cite runs/campaigns/summary.json:football_mean_topic_share \
  --cite runs/campaigns/summary.json:makeup_mean_topic_share \
  --cite runs/campaigns/summary.json:art_mean_topic_share \
  --cite runs/campaigns/summary.json:dogs_global_mean_topic_share \
  --cite runs/campaigns/summary.json:football_global_mean_topic_share \
  --cite runs/campaigns/summary.json:makeup_global_mean_topic_share \
  --cite runs/campaigns/summary.json:art_global_mean_topic_share \
  --cite runs/campaigns/summary.json:n_overlapping_creators
```
