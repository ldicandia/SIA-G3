"""Arma docs/presentacion/TP4-presentacion.pptx a partir de las figuras y números de docs/hallazgos/.

Uso (desde TP4/, con python-pptx y Pillow instalados):
    python docs/presentacion/build_deck.py
Los números de cada diapositiva son los del "Resumen para la presentación" de ej1.md y ej2.md.
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "hallazgos" / "figuras"
OUT = ROOT / "docs" / "presentacion" / "TP4-presentacion.pptx"

BG = RGBColor(0x2A, 0x1F, 0x22)
BAR = RGBColor(0xFF, 0x7F, 0x6E)
TITLE = RGBColor(0xFF, 0xD9, 0xD0)
GOLD = RGBColor(0xFF, 0x8A, 0x75)
CARD = RGBColor(0x3D, 0x2B, 0x2F)
TEXT = RGBColor(0xF6, 0xE9, 0xE6)
MUTED = RGBColor(0xB5, 0x9A, 0x97)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
GROUP = "Grupo 3"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = 13.333, 7.5
state = {"n": 0}


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, radius=None, line=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    if radius is not None:
        s.adjustments[0] = radius
    return s


def text(slide, x, y, w, h, runs, size=18, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.0, space_after=0):
    """runs: str, o lista de párrafos; cada párrafo es str o lista de (texto, {bold,color,size})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        parts = para if isinstance(para, list) else [(para, {})]
        for t, fmt in parts:
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(fmt.get("size", size))
            r.font.bold = fmt.get("bold", bold)
            r.font.color.rgb = fmt.get("color", color)
    return tb


def base(title=None, section=None, number=True):
    s = prs.slides.add_slide(BLANK)
    state["n"] += 1
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    rect(s, 0, 0, 0.22, H, BAR)
    ln = rect(s, 0.6, 7.0, W - 1.2, 0.015, MUTED)
    text(s, 0.65, 7.05, 3, 0.35, GROUP, size=12, color=MUTED)
    if number:
        text(s, W - 1.6, 7.05, 1.0, 0.35, str(state["n"]), size=12, color=MUTED, align=PP_ALIGN.RIGHT)
    text(s, W - 2.6, 0.25, 2.0, 0.4, "ITBA · SIA · TP4", size=12, color=MUTED, align=PP_ALIGN.RIGHT)
    if title:
        text(s, 0.75, 0.35, 10.3, 0.9, title, size=36, color=TITLE, bold=True)
    if section:
        text(s, 0.75, 1.15, 10, 0.5, section, size=20, color=GOLD, bold=True)
    return s


def divider(num, name):
    s = base()
    text(s, 1.2, 2.5, 10, 1.1, num, size=66, color=TITLE, bold=True)
    text(s, 1.2, 3.55, 11, 1.3, name, size=60, color=GOLD, bold=True)
    return s


def picture(slide, path, x, y, w, h, pad=0.1):
    """Panel blanco redondeado con la figura ajustada (sin deformar) al recuadro."""
    iw, ih = Image.open(path).size
    box_w, box_h = w - 2 * pad, h - 2 * pad
    sc = min(box_w / iw, box_h / ih)
    pw, ph = iw * sc, ih * sc
    px, py = x + (w - pw) / 2 - pad, y + (h - ph) / 2 - pad
    rect(slide, px, py, pw + 2 * pad, ph + 2 * pad, WHITE, MSO_SHAPE.ROUNDED_RECTANGLE, 0.03)
    slide.shapes.add_picture(str(path), Inches(px + pad), Inches(py + pad), Inches(pw), Inches(ph))


def bullets(slide, x, y, w, h, items, head=None, size=17):
    rect(slide, x, y, w, h, CARD, MSO_SHAPE.ROUNDED_RECTANGLE, 0.05)
    paras = []
    if head:
        paras.append([(head, {"bold": True, "color": GOLD, "size": size + 3})])
    for it in items:
        if isinstance(it, tuple):
            paras.append([("•  ", {"color": GOLD}), (it[0], {"bold": True}), (it[1], {})])
        else:
            paras.append([("•  ", {"color": GOLD}), (it, {})])
    text(slide, x + 0.2, y + 0.15, w - 0.4, h - 0.3, paras, size=size, space_after=9, spacing=1.05)


def stat(slide, x, y, w, h, value, label, vsize=34):
    rect(slide, x, y, w, h, CARD, MSO_SHAPE.ROUNDED_RECTANGLE, 0.1)
    text(slide, x + 0.1, y + 0.1, w - 0.2, h * 0.5, value, size=vsize, color=GOLD, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.15, y + h * 0.55, w - 0.3, h * 0.42, label, size=13, color=TEXT,
         align=PP_ALIGN.CENTER)


def fig_side(title, section, fig, items, head=None, side="left", tw=4.6, size=17):
    """Texto a un lado y figura al otro."""
    s = base(title, section)
    if side == "left":
        bullets(s, 0.75, 1.85, tw, 4.9, items, head, size)
        picture(s, FIG / fig, 0.75 + tw + 0.3, 1.8, W - 1.0 - tw - 0.3 - 0.75, 5.0)
    else:
        picture(s, FIG / fig, 0.75, 1.8, W - 1.5 - tw - 0.3, 5.0)
        bullets(s, W - 0.75 - tw, 1.85, tw, 4.9, items, head, size)
    return s


def fig_wide(title, section, fig, caption, h_fig=4.35):
    s = base(title, section)
    picture(s, FIG / fig, 0.75, 1.75, W - 1.5, h_fig)
    rect(s, 0.75, 1.75 + h_fig + 0.12, W - 1.5, 6.85 - (1.75 + h_fig + 0.12), CARD,
         MSO_SHAPE.ROUNDED_RECTANGLE, 0.12)
    text(s, 0.95, 1.75 + h_fig + 0.12, W - 1.9, 6.85 - (1.75 + h_fig + 0.12), caption, size=16,
         anchor=MSO_ANCHOR.MIDDLE)
    return s


def table(slide, x, y, w, col_w, rows, header, row_h=0.5, size=15):
    tbl = slide.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * (len(rows) + 1))).table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate([header] + rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = CARD if i else RGBColor(0x4A, 0x33, 0x38)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = i == 0 or j == 0
            r.font.color.rgb = GOLD if i == 0 else TEXT


# ---------------------------------------------------------------- 1. portada
s = base(number=True)
text(s, 1.2, 2.0, 11, 1.2, "Aprendizaje No Supervisado", size=54, color=TITLE, bold=True)
text(s, 1.2, 3.15, 11, 0.8, "Trabajo Práctico N°4", size=36, color=GOLD, bold=True)
text(s, 1.2, 4.3, 11, 1.6,
     ["Campañas de marketing (PCA, Kohonen y Oja)", "Memoria asociativa (Hopfield)",
      "Integrantes: Di Candia, Lucas · [completar con el resto del grupo]"],
     size=22, space_after=6)
text(s, 1.2, 6.2, 5, 0.5, "2Q 2026", size=20, color=MUTED)

# ---------------------------------------------------------------- 2. contenidos
s = base("Tabla de Contenidos")
items = [("01", "Datos, EDA y PCA"), ("02", "Red de Kohonen"), ("03", "Regla de Oja"),
         ("04", "Decisión para las campañas"), ("05", "Memoria asociativa: Hopfield"),
         ("06", "Conclusiones")]
for i, (n, name) in enumerate(items):
    x = 0.8 + (i % 3) * 4.05
    y = 1.9 + (i // 3) * 2.45
    rect(s, x, y, 3.8, 2.2, CARD, MSO_SHAPE.ROUNDED_RECTANGLE, 0.08)
    text(s, x + 0.3, y + 0.35, 3.2, 0.8, n, size=36, color=GOLD, bold=True)
    text(s, x + 0.3, y + 1.15, 3.3, 0.9, name, size=22, color=TITLE, bold=True)

# ================================================================ 01 EDA + PCA
divider("01", "Datos, EDA y PCA")

s = base("Los datos: 50 creadores, 33 columnas", "01. Datos, EDA y PCA")
stat(s, 0.75, 1.9, 2.9, 1.6, "50", "creadores de contenido")
stat(s, 3.85, 1.9, 2.9, 1.6, "33", "columnas, ninguna celda faltante")
stat(s, 6.95, 1.9, 2.9, 1.6, "25", "variables independientes seleccionadas")
stat(s, 10.05, 1.9, 2.5, 1.6, "4", "clientes de la agencia")
bullets(s, 0.75, 3.8, W - 1.5, 3.0, [
    ("Perfil comercial: ", "seguidores, alcance, precio, publicaciones patrocinadas."),
    ("Audiencia: ", "porcentaje argentino, alcance de no seguidores, interacción por mil cuentas."),
    ("Contenido: ", "tópico (fútbol, maquillaje, perros, arte), tipo (tutorial, reseña, entretenimiento) y formato."),
    ("Outliers: ", "33 celdas atípicas en 25 creadores (sobre todo especialistas en arte o maquillaje). Se reportan y se conservan: no son errores."),
    ("Qué se descarta: ", "6 columnas que son combinaciones exactas de otras (precio en ARS, interacción total, alcance estimado, orgánico, arte y showcase). "
     "El arte no desaparece: se reconstruye como 100 menos los otros tres tópicos."),
], size=17)

fig_side("Selección de variables", "01. Datos, EDA y PCA", "eda__correlation_heatmap_selected.png", [
    "De 31 columnas numéricas se descartan 6 que son combinaciones exactas de otras.",
    "El conjunto seleccionado tiene rango 25 de 25: ninguna combinación lineal exacta sobrevive.",
    "Las escalas difieren unas 148 mil veces (followers vs comments_per_1000): hay que estandarizar.",
], head="25 variables", side="left", tw=4.4)

fig_side("PCA: varianza explicada", "01. Datos, EDA y PCA", "pca__scree.png", [
    ("PC1: ", "53,2% de la varianza."),
    ("PC1 + PC2: ", "61,1%."),
    "Una componente concentra más de la mitad, pero con dos no alcanza para representar a los creadores.",
], head="Scree plot", side="left", tw=4.4)

fig_side("Interpretación de PC1 y PC2", "01. Datos, EDA y PCA", "pca__biplot.png", [
    ("PC1: ", "de audiencia grande y video corto de tendencia a audiencia chica, epígrafes largos y contenido didáctico."),
    ("PC2: ", "mezcla audiencia argentina y tópico."),
    "PC1 separa tamaño y estilo, no tópicos.",
], head="Biplot", side="right", tw=4.4, size=16)

fig_side("¿Estandarizar? ¿Reducir dimensión?", "01. Datos, EDA y PCA", "pca_scaling_sweep__standardize-false__scree.png", [
    "Sin estandarizar, la PCA solo mide la escala de cada variable.",
    "Con dos componentes se explica apenas 61,1% de la varianza.",
    ("Decisión: ", "Kohonen se entrena con las 25 variables estandarizadas, no con componentes principales."),
], head="Entrada de Kohonen", side="left", tw=4.6)

# ================================================================ 02 Kohonen
divider("02", "Red de Kohonen")

s = base("Configuración elegida por regla, no a mano", "02. Red de Kohonen")
picture(s, FIG / "som_sweep__sweep_tradeoff.png", 0.75, 1.8, 6.1, 5.0)
stat(s, 7.2, 1.85, 2.7, 1.5, "2,1214", "error de cuantización")
stat(s, 10.1, 1.85, 2.5, 1.5, "0,0", "error topográfico")
bullets(s, 7.2, 3.55, 5.4, 3.25, [
    "Mapa hexagonal de 64 neuronas con vecindad escalón; una regla fija se aplica sobre dos barridos de hiperparámetros.",
    "Cuidado: queda en el borde del umbral de neuronas vacías; con otro umbral la regla podría elegir otro mapa.",
], size=15)

fig_side("Entrenamiento", "02. Red de Kohonen", "som__training_curve.png", [
    ("QE: ", "baja de 3,90 en la época 1 a 2,12 y se estabiliza en las últimas 100 épocas."),
    ("Error topográfico: ", "llega a 0 y se queda ahí."),
], head="Curvas de entrenamiento", side="left", tw=4.2)

fig_side("Regiones del mapa", "02. Red de Kohonen", "som__region_map.png", [
    ("Región 1 (21): ", "engagement profundo, audiencia chica."),
    ("Región 2 (18): ", "intermedio y mezclado."),
    ("Región 3 (11): ", "alcance masivo, video y entretenimiento."),
    "La separación es real pero moderada: 32 de las 64 neuronas están vacías.",
], head="3 regiones contiguas", side="left", tw=4.4, size=16)

fig_side("Matriz U", "02. Red de Kohonen", "som__u_matrix.png", [
    ("Distancia alta entre vecinas: ", "frontera entre grupos."),
    ("Distancia baja: ", "zona homogénea de creadores parecidos."),
    "Las fronteras de las regiones siguen las paredes más altas, pero no todas: la distancia a través de ellas es 1,57 veces la media interna.",
], head="Lectura", side="left", tw=4.4)

fig_wide("Perfil de cada región", "02. Red de Kohonen", "som__region_profiles.png",
         "Media estandarizada de cada variable por región: la región 3 concentra alcance y video; "
         "la región 1, engagement profundo y audiencia chica.", h_fig=3.9)

fig_side("Planos de componentes", "02. Red de Kohonen", "som__component_planes.png", [
    "Las variables de un bloque casi colineal dibujan el mismo plano.",
    "Lo que ordena el mapa es el eje alcance y video contra engagement profundo: el mismo PC1 de 1.1.",
], head="Una variable por plano", side="right", tw=4.2)

fig_side("¿Coinciden las regiones con PC1 y PC2?", "02. Red de Kohonen", "som__biplot_regions.png", [
    ("PC1: ", "explica 0,8314 de la diferencia entre regiones."),
    ("PC2: ", "solo 0,2066."),
    "Las regiones siguen el eje de tamaño y estilo, y casi nada el de tópico.",
], head="Sí, con PC1", side="left", tw=4.4)

fig_wide("Mapa de control sobre componentes principales", "02. Red de Kohonen",
         "som_pca_control__control_region_maps.png",
         "Entrenando con 6 componentes principales (83,0% de la varianza) en vez de las variables se obtienen casi las mismas regiones "
         "(ARI = 0,748; 44 de 50 creadores quedan en la misma región): la conclusión de 1.1 no cambia.", h_fig=4.2)

# ================================================================ 03 Oja
divider("03", "Regla de Oja")

fig_side("Oja converge al mismo eje que PC1", "03. Regla de Oja", "oja__weights_vs_pc1.png", [
    ("Coseno con PC1: ", "0,999993."),
    ("Correlación del índice con PC1: ", "0,999999."),
    "La tasa η0 = 0,0005 salió de una regla registrada que no mira PC1.",
], head="Pesos de Oja vs carga de PC1", side="left", tw=4.6, size=16)

fig_side("¿Qué mide el índice?", "03. Regla de Oja", "oja__scores_vs_pc1.png", [
    "Mide tamaño y estilo (el eje de PC1), no calidad ni encaje con un cliente.",
    ("El primero del ranking, @mundoanimal.ar, ", "no lo eligió ningún cliente."),
    ("Elegir al de mayor índice para cualquier campaña no es correcto.", ""),
], head="Índice vs puntaje de PC1", side="right", tw=4.6, size=16)

# ================================================================ 04 Campañas
divider("04", "Decisión para las campañas")

s = base("Cada cliente, su regla transparente", "04. Decisión para las campañas")
table(s, 0.75, 1.9, W - 1.5, [2.3, 3.1, 3.1, 3.33],
      [["Refugios de perros", "@ideasconhuella", "@huellasyconsejos", "@consejoscaninos"],
       ["Ropa de fútbol", "@futbolcononda", "@golconmate", "@lookdecancha"],
       ["Maquillaje", "@reelargentino", "@brochasymates", "@maquillajereal"],
       ["Materiales de arte", "@colorenmovimiento", "@galeriacotidiana", "@rinconcreativo"]],
      ["Cliente", "Creador 1", "Creador 2", "Creador 3"], row_h=0.62, size=17)
bullets(s, 0.75, 5.2, W - 1.5, 1.6, [
    "Filtro por tópico y puntaje ponderado propio de cada cliente.",
    "Fútbol: los elegidos encarnan descubrimiento con alcance masivo, no profundidad educativa.",
], size=16)

fig_side("Regla vs. elegir solo por seguidores", "04. Decisión para las campañas", "campaigns__comparison.png", [
    ("Audiencia argentina: ", "88,06% perros, 81,53% fútbol, 78,98% maquillaje y 68,94% arte, contra 63,48% de los más seguidos."),
    ("USD por mil de alcance argentino: ", "19,91 fútbol, 19,96 maquillaje y 27,01 arte, contra 32,94."),
    "Límites: llega a menos gente en total y son proxies (no hay datos de ventas).",
], head="La regla gana en los 4 clientes", side="left", tw=5.0, size=15)

# ================================================================ 05 Hopfield
divider("05", "Memoria asociativa: Hopfield")

fig_side("Red de Hopfield escrita a mano", "05. Hopfield · letras 5×5", "hopfield_letters__energy.png", [
    "Pesos de Hebb con diagonal cero, energía y actualización asincrónica o sincrónica.",
    ("La energía no sube en ninguna actualización.", ""),
    "Es la misma red para las letras y para los códigos QR.",
], head="Numpy, sin librerías de redes", side="left", tw=4.6)

fig_side("Ortogonalidad: qué 4 letras guardar", "05. Hopfield · letras 5×5", "hopfield_letters__alphabet.png", [
    ("G, R, T y V: ", "producto interno máximo 1 en valor absoluto."),
    ("Peor conjunto: ", "21."),
    "En el peor conjunto la red ya cambia píxeles de letras guardadas exactas.",
], head="Las más ortogonales", side="left", tw=4.4, size=16)

fig_side("Capacidad de almacenamiento", "05. Hopfield · letras 5×5", "hopfield_letters__capacity.png", [
    ("Cota teórica de Hebb: ", "3,45 patrones."),
    "Guardamos cuatro, por encima de la cota: funciona solo porque son casi ortogonales.",
    "Cuántas letras caben depende de cuán correlacionadas estén, no solo de cuántas sean.",
], head="Más allá de la cota", side="left", tw=4.6, size=16)

fig_side("Recuperación con ruido", "05. Hopfield · letras 5×5", "hopfield_letters__recovery_vs_noise.png", [
    ("Sin errores hasta ruido 0,12 ", "con el modo asincrónico, que siempre converge."),
    "El conjunto peor queda muy por debajo.",
], head="Letras elegidas vs peor conjunto", side="left", tw=4.0)

fig_wide("Un patrón muy ruidoso y un estado espúreo", "05. Hopfield · letras 5×5", "hopfield_letters__spurious.png",
         "Con ruido alto, 39,5% de las consultas termina en un estado espúreo: una mezcla de letras con signo. "
         "Las 16 mezclas de tres letras con signo son puntos fijos.", h_fig=3.0)

fig_side("Los cinco QR se parecen demasiado", "05. Hopfield · códigos QR 37×37", "hopfield_qr__overlaps.png", [
    ("Superposición media: ", "0,6701."),
    ("Módulos compartidos: ", "0,6976."),
    "Son todo menos ortogonales: las letras elegidas estaban en el otro extremo.",
], head="Matriz de superposiciones", side="left", tw=4.6)

fig_wide("Hebb no recupera ningún código QR", "05. Hopfield · códigos QR 37×37", "hopfield_qr__hebb_attractor.png",
         "Ninguno de los 5 códigos es un punto fijo, y 97,47% de las consultas termina en el mismo \"QR promedio\". "
         "Es la interferencia que las letras ortogonales evitaban.", h_fig=4.2)

fig_wide("Con pseudoinversa: la misma red, sin interferencia", "05. Hopfield · códigos QR 37×37",
         "hopfield_qr__recovery_vs_noise.png",
         "Los 5 códigos pasan a ser puntos fijos y se recupera todo hasta 0,35 de ruido. No es la regla del curso: "
         "prueba que lo que rompe a Hebb es la correlación, no la cantidad de patrones.", h_fig=3.9)

fig_side("Iteraciones y energía en cada paso", "05. Hopfield · códigos QR 37×37", "hopfield_qr__energy_per_step.png", [
    "La energía no sube en ningún paso.",
    "Casi todas las consultas convergen en pocos barridos.",
    "Pero la red baja al mínimo más cercano, que con Hebb es el equivocado.",
], head="Convergencia", side="left", tw=4.4, size=16)

fig_wide("Lectura con un decodificador QR", "05. Hopfield · códigos QR 37×37", "hopfield_qr__decode_vs_noise.png",
         "El lector decodifica los 5 originales. Tras la pseudoinversa un código se lee igual de bien que se recupera; "
         "tras Hebb no lee ninguno: el \"QR promedio\" falla de manera visible.", h_fig=3.9)

s = base("Limitaciones del caso de uso", "05. Hopfield · códigos QR 37×37")
lim = [("Correlación entre códigos", "Hebb no guarda ninguno aunque la cota (188,922 patrones) esté muy por encima de cinco."),
       ("Estados espúreos", "Con mucho ruido la red termina en mezclas que no son ningún menú."),
       ("Capacidad efectiva", "Mucho menor que la cota teórica cuando los patrones se parecen."),
       ("Umbral de ruido", "El umbral de 0,35 es el de la pseudoinversa; una foto real no es una matriz limpia y alineada.")]
for i, (h, b) in enumerate(lim):
    x = 0.75 + (i % 2) * 6.05
    y = 1.9 + (i // 2) * 2.5
    rect(s, x, y, 5.85, 2.3, CARD, MSO_SHAPE.ROUNDED_RECTANGLE, 0.07)
    text(s, x + 0.25, y + 0.2, 5.4, 0.6, h, size=22, color=GOLD, bold=True)
    text(s, x + 0.25, y + 0.85, 5.4, 1.4, b, size=17)

# ================================================================ 06 Conclusiones
divider("06", "Conclusiones")

s = base("Ejercicio 1: un eje domina", "06. Conclusiones")
bullets(s, 0.75, 1.85, W - 1.5, 4.95, [
    ("PC1 ordena a los creadores por tamaño y estilo ", "y concentra 53,2% de la varianza."),
    ("Kohonen y Oja encuentran lo mismo: ", "las regiones siguen PC1 (0,8314 contra 0,2066) y Oja converge a PC1 (coseno 0,999993)."),
    ("Por eso ningún índice único decide una campaña: ", "el primero del ranking no lo eligió ningún cliente."),
    ("Reglas por cliente: ", "mejor audiencia argentina que elegir por seguidores en los 4 clientes, y menor costo por mil en fútbol, maquillaje y arte."),
    ("Límite: ", "todo se argumenta con proxies, sin datos de ventas ni conversión; reglas y umbrales son criterios del grupo."),
], size=19)

s = base("Ejercicio 2: la ortogonalidad es el hilo", "06. Conclusiones")
bullets(s, 0.75, 1.85, W - 1.5, 4.95, [
    ("Letras: ", "G, R, T y V (producto interno máximo 1) se recuperan sin errores hasta ruido 0,12; con mucho ruido aparecen estados espúreos."),
    ("QR: ", "comparten ~70% de sus módulos; con Hebb ninguno es punto fijo y 97,47% de las consultas cae en el mismo \"QR promedio\"."),
    ("Pseudoinversa: ", "recupera todo hasta 0,35 de ruido, lo que confirma que la causa es la correlación y no la cantidad."),
    ("Caso de uso: ", "limitado por correlación, estados espúreos, capacidad efectiva y umbral de ruido."),
], size=19)

s = base("¿Preguntas?")
text(s, 1.2, 2.8, 11, 1.2, "¡Gracias!", size=60, color=GOLD, bold=True)
text(s, 1.2, 4.2, 11, 1.0, "Todo el trabajo se regenera con un comando: python -m tp4.delivery verify",
     size=20, color=MUTED)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"{OUT} ({state['n']} diapositivas)")
