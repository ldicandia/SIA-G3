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
