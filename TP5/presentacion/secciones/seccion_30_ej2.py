"""Seccion 30: Ejercicio 2 (autoencoder variacional sobre emojis).

Siete diapositivas: el dataset (2.a), el esquema variacional y lo que resuelve (2.b) y la
muestra nueva con el juicio de los autores (2.c). En la diapositiva queda solo lo comprobable
contra output/ o contra la figura; la explicacion va a las notas del orador.

Ninguna cifra se escribe a mano, ni en la diapositiva ni en las notas: todas salen del reporte
que regenera scripts/regenerar.sh, leido con lector.Salidas. El juicio visual de los autores se
lee de docs/ej2-analisis.md, que tests/vae_doc_consistency.rs verifica contra el codigo.
"""

from __future__ import annotations

from pathlib import Path

from disposicion import (
    ALTO_UTIL,
    ANCHO_UTIL,
    cifra,
    diagrama,
    figura_alta,
    figura_ancha,
    texto_justo,
    vinetas,
)
from lector import FaltaEntrada, campo, numero
from pptx_minimo import ANCHO_IN, COLOR_AMBAR, CONTENIDO_Y_IN, MARGEN_IN, dimensiones_png

EJ = "ej2"
SALIDA = "salida-ej2.txt"
FUENTE = f"{EJ}/{SALIDA}"
DOC = Path(__file__).resolve().parents[2] / "docs" / "ej2-analisis.md"

# Nombres de los estilos en src/emoji.rs (EMOJI_EYES y EMOJI_MOUTHS), en el orden de filas y columnas.
OJOS = "dots, happy, wide, wink, closed y x"
BOCAS = "smile, grin, flat, frown, open y small"


def _leer_doc(fragmento: str) -> str:
    """Parrafo del documento de analisis que contiene `fragmento`."""
    if not DOC.is_file():
        raise FaltaEntrada(f"falta {DOC}; es el documento de analisis del Ej 2")
    for linea in DOC.read_text(encoding="utf-8").splitlines():
        if fragmento in linea:
            return linea.strip()
    raise ValueError(f"{DOC.name}: ningun parrafo contiene {fragmento!r}")


class _Datos:
    """Todo lo que las diapositivas toman del reporte y del documento de analisis."""

    def __init__(self, salidas) -> None:
        linea = lambda prefijo: salidas.linea(EJ, SALIDA, prefijo)  # noqa: E731
        self.emojis = linea("emojis:")
        self.separacion = linea("separacion:")
        self.cfg = linea("configuracion:")
        self.perdida = linea("PERDIDA:")
        self.reconstruccion = linea("RECONSTRUCCION:")
        self.manifold = linea("MANIFOLD:")
        self.regiones = linea("REGIONES:")
        self.nuevo = linea("EMOJI NUEVO:")
        self.galeria = linea("GALERIA:")
        self.prior = linea("VAE PRIOR:")
        self.prior_simple = linea("AE simple PRIOR:")
        self.control = linea("CONTROL NEGATIVO:")
        self.contraste = linea("CONTRASTE:")
        self.resultado = linea("RESULTADO EJ2:")
        self.aceptacion = salidas.lineas_con_prefijo(EJ, SALIDA, "ACEPTACION")
        self.entradas_galeria = salidas.lineas_con_prefijo(EJ, SALIDA, "  nuevo ")

        self.n = numero(self.emojis, r"emojis: (\d+) imagenes")
        self.lado = numero(self.emojis, r"imagenes de (\d+x\d+)")
        self.sep_min = numero(self.separacion, r"minima (\d+) px")
        self.sep_mediana = numero(self.separacion, r"mediana (\d+\.\d+) px")
        self.radio = numero(self.separacion, r"radio de cercania (\d+) px")

        self.nombre = campo(self.cfg, "configuracion")
        self.encoder = numero(self.cfg, r"encoder (\S+)").split("-")
        self.decoder = numero(self.cfg, r"decoder (\S+)").split("-")
        self.optimizador = campo(self.cfg, "optimizador")
        self.epocas = campo(self.cfg, "epocas")
        self.semilla = campo(self.cfg, "semilla de entrenamiento")
        self.latente = self.decoder[0]

        self.rec_fin = numero(self.perdida, r"reconstruccion \S+ -> (\d+\.\d+)")
        self.rec_ini = numero(self.perdida, r"reconstruccion (\d+\.\d+) ->")
        self.kl_ini = numero(self.perdida, r"KL (\d+\.\d+) ->")
        self.kl_fin = numero(self.perdida, r"KL \S+ -> (\d+\.\d+)")
        self.total_fin = numero(self.perdida, r"total \S+ -> (\d+\.\d+)")

        self.rec_media = numero(self.reconstruccion, r"media (\d+\.\d+) px")
        self.rec_max = numero(self.reconstruccion, r"maximo (\d+) px")
        self.rec_exactos = numero(self.reconstruccion, r"; (\d+/\d+) emojis sin")

        self.grilla = numero(self.manifold, r"\((\d+x\d+)\)")
        self.rango = numero(self.manifold, r"z en \[-(\d+\.\d+), ")
        self.celdas_reconocibles = numero(self.manifold, r"reconocibles (\d+\.\d+) %")
        self.salto = numero(self.contraste, r"vecinas: VAE (\d+\.\d+) px")
        self.salto_simple = numero(self.contraste, r"vecinas: VAE \S+ px contra (\d+\.\d+) px")

        self.sorteos = numero(self.prior, r"de (\d+) puntos")
        self.reconocibles = numero(self.contraste, r"reconocible: VAE (\d+\.\d+) %")
        self.reconocibles_simple = numero(self.contraste, r"sin esquema variacional (\d+\.\d+) %")
        self.plausibles = numero(self.contraste, r"cara plausible: VAE (\d+\.\d+) %")
        self.plausibles_simple = numero(self.contraste, r"cara plausible: VAE \S+ % contra (\d+\.\d+) %")
        self.hibridos = numero(self.contraste, r"hibrido plausible: VAE (\d+\.\d+) %")
        self.hibridos_simple = numero(self.contraste, r"hibrido plausible: VAE \S+ % contra (\d+\.\d+) %")
        self.rec_simple = numero(self.contraste, r"reconstruccion media: VAE \S+ px contra (\d+\.\d+) px")

        self.z = numero(self.nuevo, r"z=(\(.+?\))")
        self.sorteo = numero(self.nuevo, r"sorteo (\d+) de")
        self.cercano = numero(self.nuevo, r"emoji mas cercano: (\S+) a")
        self.distancia = numero(self.nuevo, r"emoji mas cercano: \S+ a (\d+) px")
        self.ojos_px = numero(self.nuevo, r"ojos: \S+ a (\d+) px")
        self.ojos_piso = numero(self.nuevo, r"ojos: .+?\(piso (\d+) px\)")
        self.boca_px = numero(self.nuevo, r"boca: \S+ a (\d+) px")
        self.boca_piso = numero(self.nuevo, r"boca: .+?\(piso (\d+) px\)")
        if numero(self.resultado, r"emoji nuevo a (\d+) px") != self.distancia:
            raise ValueError(f"{FUENTE}: RESULTADO EJ2 y EMOJI NUEVO no dan la misma distancia")

        self.control_medias = numero(self.control, r"como nuevas: (\d+/\d+) \(")
        self.control_anterior = numero(self.control, r"banda de pixeles: (\d+/\d+)\)")
        self.control_muestras = numero(self.control, r"por emoji: (\d+\.\d+) %")
        self.ok = numero(self.resultado, r"\((\d+)/\d+ comprobaciones")
        self.total = numero(self.resultado, r"\(\d+/(\d+) comprobaciones")
        if len(self.aceptacion) != int(self.total) or not all(l.startswith("ACEPTACION OK") for l in self.aceptacion):
            raise ValueError(f"{FUENTE}: las lineas de aceptacion no coinciden con RESULTADO EJ2")

        # Pasajes del documento de analisis, leidos y nunca retipeados.
        completo = _leer_doc("El dataset es el producto completo")
        self.estilos_ojos = numero(completo, r"de (\d+) estilos de ojos")
        self.estilos_boca = numero(completo, r"por (\d+) estilos de boca")
        _leer_doc("caras de la misma familia que el dataset (mismo contorno, dos ojos y una boca)")
        juicio = _leer_doc("En resumen, solo")
        self.juicio_claras = numero(juicio, r"solo (\d+) de las \d+ entradas")
        self.juicio_entradas = numero(juicio, r"solo \d+ de las (\d+) entradas")
        self.juicio_sutiles = numero(juicio, r"(\d+) son sutiles")
        self.juicio_dudosas = numero(juicio, r"(\d+) es dudosa")
        if int(self.juicio_entradas) != len(self.entradas_galeria):
            raise ValueError(f"{DOC.name}: el juicio habla de {self.juicio_entradas} entradas y la galeria tiene otra cantidad")


# --------------------------------------------------------------------------
# Diapositivas
# --------------------------------------------------------------------------


def _dataset(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Dataset")
    x, ancho = figura_alta(d, salidas, EJ, "emojis.png")
    texto_justo(
        d, x, CONTENIDO_Y_IN + 0.6, ancho,
        vinetas(
            [
                f"{dat.n} emojis de {dat.lado} píxeles, construidos por código.",
                f"{dat.estilos_ojos} estilos de ojos (filas) por {dat.estilos_boca} estilos de boca (columnas).",
                f"Los dos emojis más parecidos difieren en {dat.sep_min} px.",
            ]
        ),
        tam=22,
    )
    d.notas(
        [
            "Mensaje: construimos el dataset, como permite la consigna. Son caras estilo emoji dibujadas por "
            "código, no los emojis de ninguna plataforma.",
            "Cómo leer la figura: cada fila es un estilo de ojos y cada columna un estilo de boca; el contorno "
            f"de la cara es el mismo en todas. Ojos, de arriba hacia abajo: {OJOS}. Bocas, de izquierda a "
            f"derecha: {BOCAS}. Los nombres de los emojis combinan los dos, por ejemplo x-grin.",
            "Cada imagen son valores entre cero y uno, donde uno es tinta; los bordes tienen grises por el "
            "suavizado del dibujo.",
            f"Distancia entre emojis: píxeles que difieren después de binarizar. El par más parecido difiere en "
            f"{dat.sep_min} px; la mediana de la distancia de cada emoji a su vecino más cercano es "
            f"{dat.sep_mediana} px. De ahí sale el radio de cercanía, {dat.radio} px.",
            f"Definición que se usa en el resto del ejercicio: una imagen generada es reconocible si está a lo "
            f"sumo a {dat.radio} px de algún emoji del dataset.",
            "Como están todas las combinaciones de ojos y boca, una muestra nueva no puede ser una combinación "
            "que falte: eso importa en la diapositiva del emoji nuevo.",
        ],
        citas=[(FUENTE, [dat.emojis, dat.separacion])],
    )


def _arquitectura(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Arquitectura")
    y = diagrama(
        d,
        [
            ("Entrada", f"{dat.encoder[0]} píxeles", None),
            ("Encoder", "-".join(dat.encoder[1:-1]), None),
            ("μ y log σ²", f"{dat.encoder[-1]} salidas", None),
            ("z = μ + σ·ε", f"{dat.latente} dimensiones", COLOR_AMBAR),
            ("Decoder", "-".join(dat.decoder[1:-1]), None),
            ("Salida", f"{dat.decoder[-1]} píxeles", None),
        ],
        alto=1.5,
        hueco=0.4,
    )
    texto_justo(
        d, MARGEN_IN, y + 0.5, ANCHO_UTIL,
        vinetas(
            [
                "σ = exp(0.5 · log σ²); ε ~ N(0, I).",
                "Pérdida: reconstrucción (BCE) + KL entre q(z|x) y el prior N(0, I).",
            ]
        ),
    )
    d.notas(
        [
            "Mensaje: es el autoencoder anterior con una capa estocástica en el medio. El encoder ya no da un "
            "código sino una distribución, y la pérdida suma un término que acerca esa distribución al prior.",
            f"Cómo leer el diagrama: el encoder tiene {dat.encoder[-1]} salidas, una media y un logaritmo de "
            f"varianza por cada una de las {dat.latente} dimensiones latentes. Con ellas se arma z sumando a la "
            "media un ruido ε, sorteado de una normal estándar, multiplicado por el desvío. El decoder recibe z.",
            "Por qué se escribe así, el truco de reparametrización: el sorteo queda fuera de la red, en ε, y el "
            "gradiente puede llegar al encoder a través de la media y del desvío.",
            "La pérdida tiene dos términos. La reconstrucción es la entropía cruzada binaria entre la salida y "
            "la imagen. El KL mide cuánto se aparta la distribución del encoder del prior, una normal estándar; "
            "tiene forma cerrada y se suma con peso uno.",
            "Qué problema resuelve, que es lo que pide la consigna: en un autoencoder simple nada obliga a los "
            "códigos a ocupar una zona conocida del plano, así que no se sabe de dónde sortear para generar. El "
            "término KL empuja los códigos hacia el prior, y entonces sortear del prior da puntos que el "
            "decoder sabe decodificar. La evidencia está en la diapositiva siguiente.",
            f"Entrenamiento: {dat.optimizador}, {dat.epocas} épocas con los {dat.n} emojis en cada paso y ruido "
            f"nuevo en cada época, semilla {dat.semilla}. No se estudiaron otras arquitecturas en este ejercicio.",
        ],
        citas=[(FUENTE, [dat.cfg])],
    )


def _contraste(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Técnicas: VAE frente a autoencoder simple")
    alto = 0.905 * ALTO_UTIL
    hueco = 0.5
    anchos = []
    for nombre in ("vae_manifold.png", "plain_manifold.png"):
        w, h = dimensiones_png(salidas.png(EJ, nombre).read_bytes())
        anchos.append(alto * w / h)
    x = (ANCHO_IN - sum(anchos) - hueco) / 2
    for nombre, ancho in zip(("vae_manifold.png", "plain_manifold.png"), anchos):
        d.imagen(salidas.png(EJ, nombre), x, CONTENIDO_Y_IN, ancho, alto, alt=salidas.alt(EJ, nombre))
        x += ancho + hueco
    texto_justo(
        d, MARGEN_IN, CONTENIDO_Y_IN + alto + 0.06, ANCHO_UTIL,
        f"Sorteos del prior que dan un emoji reconocible: VAE {dat.reconocibles} %; autoencoder simple "
        f"{dat.reconocibles_simple} %.",
    )
    d.notas(
        [
            "Mensaje: esta es la evidencia de que el esquema variacional resuelve el problema del espacio "
            "latente. Sorteando del prior, el VAE da emojis reconocibles la mayoría de las veces y el "
            "autoencoder simple, pocas.",
            "Qué se compara: el autoencoder simple tiene la misma arquitectura, optimizador, pérdida de "
            "reconstrucción, épocas y semilla que el VAE, sin la capa estocástica y sin el término KL.",
            f"Cómo leer las figuras: cada una decodifica una grilla de {dat.grilla} puntos del plano latente. La "
            f"del VAE recorre z entre menos {dat.rango} y {dat.rango} en cada eje, la zona donde cae casi todo "
            "sorteo del prior. La del autoencoder simple recorre la caja que ocupan sus propios códigos, porque "
            "no tiene prior.",
            "Qué se ve: en el VAE todas las celdas se ven como caras y los cambios son graduales; de izquierda "
            "a derecha cambian los ojos y de arriba hacia abajo la boca. En el autoencoder simple hay una zona "
            "grande de imágenes de fondo negro.",
            f"No todas las celdas del VAE son emojis del dataset: el {dat.celdas_reconocibles} % de las celdas "
            "es reconocible; el resto son mezclas entre dos estilos.",
            f"Los números del pie salen de sortear {dat.sorteos} puntos del prior en cada modelo. Reconocible "
            f"quiere decir a lo sumo a {dat.radio} px de un emoji del dataset.",
            f"Otras medidas del mismo contraste: caras plausibles, {dat.plausibles} % contra "
            f"{dat.plausibles_simple} %; salto medio entre celdas vecinas, {dat.salto} px contra "
            f"{dat.salto_simple} px.",
            f"Lo que no favorece al VAE, para decirlo antes de que lo pregunten: el autoencoder simple "
            f"reconstruye algo mejor, {dat.rec_simple} px contra {dat.rec_media} px de error medio por emoji. "
            "Reconstruir bien no alcanza para generar.",
            "Límite: el contraste usa una sola arquitectura y una sola semilla.",
        ],
        citas=[(FUENTE, [dat.contraste, dat.prior, dat.prior_simple, dat.manifold])],
    )


def _perdida(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: evolución de la pérdida")
    figura_ancha(
        d, salidas, EJ, "vae_loss.png",
        f"Época {dat.epocas}: reconstrucción {dat.rec_fin}, KL {dat.kl_fin}, total {dat.total_fin}.",
    )
    d.notas(
        [
            "Mensaje: el entrenamiento converge y el término KL queda chico frente a la reconstrucción, pero "
            "no se anula.",
            "Cómo leer la figura: pérdida por época en escala logarítmica. La curva de reconstrucción y la "
            "del total casi se superponen arriba, porque el KL, abajo, es mucho menor.",
            f"La reconstrucción baja de {dat.rec_ini} a {dat.rec_fin}. No llega a cero porque las imágenes "
            "tienen grises en los bordes, y con entropía cruzada un objetivo gris tiene un mínimo positivo.",
            f"El KL empieza en {dat.kl_ini}, tiene un pico en las primeras épocas, baja y vuelve a subir hasta "
            f"estabilizarse cerca de {dat.kl_fin}. Que no sea cero indica que el encoder usa el código: "
            "distintos emojis tienen distribuciones distintas.",
            "Las unidades son nats por imagen, sumados sobre todos los píxeles y promediados sobre el dataset.",
        ],
        citas=[(FUENTE, [dat.perdida])],
    )


def _reconstruccion(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: reconstrucción")
    x, ancho = figura_alta(d, salidas, EJ, "vae_reconstruction.png")
    y = cifra(d, x, CONTENIDO_Y_IN + 0.5, ancho, f"{dat.rec_media} px", "error medio de reconstrucción por emoji")
    texto_justo(
        d, x, y + 0.4, ancho,
        [f"Máximo: {dat.rec_max} px.", f"Sin píxeles incorrectos: {dat.rec_exactos} emojis."],
    )
    d.notas(
        [
            "Mensaje: el VAE reconstruye los emojis con pocos píxeles de error; se reconoce cuál es cada uno.",
            "Cómo leer la figura: en las filas impares están los originales y, debajo de cada uno, su "
            "reconstrucción. El pie es la cantidad de píxeles que difieren del original después de binarizar.",
            "La reconstrucción se hace desde la media del encoder, sin sortear ruido.",
            "Qué se ve en los pies: los errores más grandes están en las bocas grin y flat, la segunda y la "
            "tercera de cada grupo de seis; en el resto el error es chico.",
            f"Para comparar: el autoencoder simple reconstruye con {dat.rec_simple} px de error medio. El VAE "
            "paga un poco de reconstrucción a cambio de un espacio latente del que se puede sortear.",
        ],
        citas=[(FUENTE, [dat.reconstruccion, dat.contraste])],
    )


def _emoji_nuevo(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: emoji nuevo")
    figura_ancha(
        d, salidas, EJ, "vae_new_emoji.png",
        [
            f"Nuevo 1: z = {dat.z}, sorteado del prior; a {dat.distancia} px del emoji más cercano, {dat.cercano}.",
            "Juicio de los autores: pertenece al conjunto (mismo contorno, dos ojos, una boca).",
        ],
    )
    d.notas(
        [
            "Mensaje: sorteando del prior y decodificando, el VAE genera caras que no están en el dataset y "
            "que juzgamos del mismo conjunto. Es lo que pide el último punto de la consigna.",
            "Cómo leer la figura: arriba, ocho emojis generados; el pie es su número de orden y la distancia en "
            "píxeles al emoji del dataset más cercano. Abajo, ese emoji más cercano con su nombre.",
            f"El que presentamos es el primero: el sorteo {dat.sorteo} de {dat.sorteos}. Su emoji más cercano es "
            f"{dat.cercano}, a {dat.distancia} px, más que el radio de cercanía de {dat.radio} px.",
            "Qué llamamos nuevo: que alguna región, ojos o boca, esté más lejos que su piso de ruido de todos "
            "los estilos de esa región, con el contorno intacto. El piso de ruido de una región es el mayor "
            "error del VAE en esa región al reconstruir los emojis del dataset.",
            f"En el primero la novedad está en los ojos: a {dat.ojos_px} px del estilo más cercano, con un piso "
            f"de {dat.ojos_piso} px. La boca queda dentro de su piso, {dat.boca_px} px contra {dat.boca_piso} px.",
            "Por qué no alcanza con la distancia total: el decodificador se equivoca en varios píxeles al "
            f"reconstruir un emoji real, hasta {dat.rec_max} px. Un criterio de distancia total aceptaba como "
            f"nuevas {dat.control_anterior} reconstrucciones del propio dataset; el criterio por región acepta "
            f"{dat.control_medias}, y el {dat.control_muestras} % de las reconstrucciones con ruido. Ese "
            "primer resultado vale por construcción, porque los pisos se miden sobre esas mismas "
            "reconstrucciones; la evidencia independiente es el segundo.",
            f"Juicio de los autores, que es lo que pide la consigna: las ocho son caras de la misma familia que "
            f"el dataset. A simple vista, {dat.juicio_claras} de las {dat.juicio_entradas} se distinguen de su "
            f"emoji más cercano, {dat.juicio_sutiles} son sutiles y {dat.juicio_dudosas} es dudosa. La primera "
            "es de las que se distinguen: donde el más cercano tiene cruces, la generada tiene anillos.",
            "Límites: el juicio es visual y subjetivo; y la proporción de sorteos que el criterio acepta como "
            f"nuevos es parecida en el autoencoder simple, {dat.hibridos} % contra {dat.hibridos_simple} %. Lo "
            "que separa a los dos modelos es que las caras del VAE son plausibles.",
        ],
        citas=[(FUENTE, [dat.nuevo, dat.regiones, dat.galeria] + dat.entradas_galeria + [dat.control])],
    )


def _conclusiones(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Conclusiones")
    texto_justo(
        d, MARGEN_IN, CONTENIDO_Y_IN + 0.5, ANCHO_UTIL,
        vinetas(
            [
                f"Sorteos del prior que dan un emoji reconocible: VAE {dat.reconocibles} %; autoencoder simple "
                f"{dat.reconocibles_simple} %.",
                f"Reconstrucción: {dat.rec_media} px de error medio por emoji.",
                f"Emoji nuevo a {dat.distancia} px del más cercano; los autores lo juzgan parte del conjunto.",
            ]
        ),
        tam=24,
    )
    d.notas(
        [
            "Mensaje de cierre del ejercicio: el esquema variacional vuelve muestreable el espacio latente, y "
            "de ese espacio sale una muestra nueva que juzgamos del conjunto.",
            "Las tres frases salen del reporte: la primera de la línea CONTRASTE, la segunda de la línea "
            "RECONSTRUCCION y la tercera de la línea EMOJI NUEVO más el juicio del documento de análisis.",
            f"El reporte termina con {dat.total} comprobaciones de aceptación y las {dat.ok} dan bien. Sus "
            "umbrales son convenciones de este trabajo, no medidas externas.",
            "Límites: el dataset es chico y construido por nosotros; el latente tiene dos dimensiones y no se "
            "estudiaron otras arquitecturas; el contraste con el autoencoder simple usa una sola semilla; el "
            "criterio de novedad está definido respecto de este dataset y de este decodificador, y el juicio "
            "final es visual.",
            "Si preguntan qué haríamos distinto: más semillas en el contraste y un dataset con combinaciones "
            "ausentes, para poder mostrar una combinación nueva y no solo una región nueva.",
        ],
        citas=[(FUENTE, [dat.contraste, dat.reconstruccion] + dat.aceptacion + [dat.resultado])],
    )


def agregar(pres, salidas) -> None:
    dat = _Datos(salidas)
    _dataset(pres, salidas, dat)
    _arquitectura(pres, dat)
    _contraste(pres, salidas, dat)
    _perdida(pres, salidas, dat)
    _reconstruccion(pres, salidas, dat)
    _emoji_nuevo(pres, salidas, dat)
    _conclusiones(pres, dat)
