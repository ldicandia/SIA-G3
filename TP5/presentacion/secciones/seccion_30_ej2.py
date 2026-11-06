"""Seccion 30: Ejercicio 2 (autoencoder variacional sobre emojis).

Ninguna cifra se escribe a mano: todas salen de los reportes y archivos que
regenera scripts/regenerar.sh, leidos con lector.Salidas. Las lineas que se
muestran literalmente son lineas exactas de ej2/salida-ej2.txt. Las frases
sobre las figuras parafrasean docs/ej2-analisis.md y se contrastaron con la
imagen abierta. Dos pasajes del documento (la oracion sobre el dataset completo
y el juicio visual de los autores) se leen del archivo al construir y nunca se
retipean.
"""

from __future__ import annotations

from pathlib import Path

from lector import FaltaEntrada, campo, numero
from pptx_minimo import (
    ALTO_IN,
    ANCHO_IN,
    COLOR_AMBAR,
    COLOR_FONDO_CIFRA,
    CONTENIDO_Y_IN,
    MARGEN_IN,
    Parrafo,
    altura_estimada,
)

EJ = "ej2"
SALIDA = "salida-ej2.txt"
FUENTE = f"{EJ}/{SALIDA}"
DOC = Path(__file__).resolve().parents[2] / "docs" / "ej2-analisis.md"

ANCHO_UTIL = ANCHO_IN - 2 * MARGEN_IN
Y_FIN = ALTO_IN - MARGEN_IN
TAM_CUERPO = 16
TAM_TEXTO = 14
TAM_PIE = 13
TAM_CAJA = 14
TAM_CODIGO = 11
TAM_CODIGO_CHICO = 10

# Unidades y umbrales de lectura (no son cifras del resultado).
CIEN = 100
HUECO = 0.15
HUECO_CHICO = 0.1
RELLENO_ALTO = 0.01
PARECIDO_PUNTOS = 5.0  # dos porcentajes a menos de esto se describen como casi iguales

ALTO_CAJA = 1.2
ANCHO_CAJA = 1.7
ALTO_CIFRA = 1.4


# --------------------------------------------------------------------------
# Utilidades de lectura y de maquetado
# --------------------------------------------------------------------------


def _enumerar(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


def _leer_doc(inicio: str) -> str:
    """Parrafo completo del documento de analisis que empieza con `inicio`."""
    if not DOC.is_file():
        raise FaltaEntrada(f"falta {DOC}; es el documento de analisis del Ej 2")
    for linea in DOC.read_text(encoding="utf-8").splitlines():
        if linea.startswith(inicio):
            return linea.strip()
    raise ValueError(f"{DOC.name}: ningun parrafo empieza con {inicio!r}")


def _codigo(d, x: float, y: float, w: float, lineas: list[str], *, tam: float = TAM_CODIGO) -> float:
    """Bloque monoespaciado de alto justo; devuelve el alto usado."""
    alto = altura_estimada([Parrafo(l) for l in lineas], w, tam=tam, mono=True) + RELLENO_ALTO
    d.codigo(x, y, w, alto, lineas, fuente=FUENTE, tam=tam)
    return alto


def _texto(
    d, x: float, y: float, w: float, parrafos: list[Parrafo], *, tam: float = TAM_TEXTO, mono: bool = False
) -> float:
    """Caja de texto de alto justo; devuelve el alto usado."""
    alto = altura_estimada(parrafos, w, tam=tam, mono=mono) + RELLENO_ALTO
    d.texto(x, y, w, alto, parrafos, tam=tam, mono=mono)
    return alto


def _vinetas(textos: list[str]) -> list[Parrafo]:
    return [Parrafo(t, vineta=True) for t in textos]


class _Datos:
    """Todo lo que las diapositivas toman del reporte del Ej 2 y del documento."""

    def __init__(self, salidas) -> None:
        def una(prefijo: str) -> str:
            return salidas.linea(EJ, SALIDA, prefijo)

        # Lineas literales.
        self.dataset = una("emojis:")
        self.separacion = una("separacion:")
        self.cfg = una("configuracion:")
        self.tabla = salidas.bloque(EJ, SALIDA, "epoca  reconstruccion", 12)
        self.perdida = una("PERDIDA:")
        self.recon = una("RECONSTRUCCION:")
        self.manifold = una("MANIFOLD:")
        self.regiones = una("REGIONES:")
        self.nuevo = una("EMOJI NUEVO:")
        self.galeria = salidas.bloque(EJ, SALIDA, "GALERIA:", 9)
        self.par = salidas.bloque(EJ, SALIDA, "nuevo (izquierda) y", 25)
        self.vae_prior = una("VAE PRIOR:")
        self.ae_prior = una("AE simple PRIOR:")
        self.control = una("CONTROL NEGATIVO:")
        self.contraste = una("CONTRASTE:")
        self.aceptacion = salidas.lineas_con_prefijo(EJ, SALIDA, "ACEPTACION OK:")
        self.resultado = una("RESULTADO EJ2:")

        # Dataset.
        self.n_emojis = numero(self.dataset, r"emojis: (\d+) imagenes")
        self.lado = numero(self.dataset, r"imagenes de (\d+)x\d+")
        self.sep_min = numero(self.separacion, r"minima (\S+) px")
        self.sep_med = numero(self.separacion, r"mediana (\S+) px")
        self.radio = numero(self.separacion, r"radio de cercania (\S+) px")

        # Configuracion: el encoder emite una media y un log-varianza por dimension latente.
        self.encoder = [int(n) for n in numero(self.cfg, r"encoder ([\d-]+)").split("-")]
        self.decoder = [int(n) for n in numero(self.cfg, r"decoder ([\d-]+)").split("-")]
        self.latente = self.decoder[0]
        if self.encoder[-1] != 2 * self.latente:
            raise ValueError(
                f"{EJ}: la salida del encoder ({self.encoder[-1]}) no es el doble del "
                f"latente ({self.latente}); la diapositiva de arquitectura no lo describe"
            )
        self.optimizador = campo(self.cfg, "optimizador")
        self.epocas = campo(self.cfg, "epocas")
        self.semilla = campo(self.cfg, "semilla de entrenamiento")

        # Perdida.
        self.rec0 = numero(self.perdida, r"reconstruccion (\S+) ->")
        self.rec1 = numero(self.perdida, r"reconstruccion \S+ -> (\S+) \|")
        self.kl0 = numero(self.perdida, r"KL (\S+) ->")
        self.kl1 = numero(self.perdida, r"KL \S+ -> (\S+) \|")

        # Reconstruccion y plano latente.
        self.rec_media = numero(self.recon, r"media (\S+) px")
        self.rec_max = numero(self.recon, r"maximo (\d+) px")
        self.rec_sin = numero(self.recon, r"; (\d+)/\d+ emojis sin pixeles")
        self.celdas = numero(self.manifold, r"MANIFOLD: (\d+) celdas")
        self.celdas_rec = numero(self.manifold, r"reconocibles (\S+) %")
        self.salto_vae = numero(self.manifold, r"salto medio entre celdas vecinas (\S+) px")

        # Regiones.
        self.piso_ojos = numero(self.regiones, r"\): ojos (\d+) px")
        self.piso_boca = numero(self.regiones, r"\): ojos \d+ px, boca (\d+) px")
        self.sep_ojos = numero(self.regiones, r"entre estilos: ojos (\d+) px")
        self.sep_boca = numero(self.regiones, r"entre estilos: ojos \d+ px, boca (\d+) px")

        # Emoji nuevo.
        self.sorteo = numero(self.nuevo, r"sorteo (\d+) de \d+")
        self.sorteos = numero(self.nuevo, r"sorteo \d+ de (\d+)")
        self.cercano = numero(self.nuevo, r"emoji mas cercano: (\S+) a \d+ px")
        self.dist_cercano = numero(self.nuevo, r"emoji mas cercano: \S+ a (\d+) px")
        self.d_ojos = int(numero(self.nuevo, r"ojos: \S+ a (\d+) px del estilo"))
        self.d_boca = int(numero(self.nuevo, r"boca: \S+ a (\d+) px \(piso"))
        self.ojos_fuera = self.d_ojos > int(self.piso_ojos)
        self.boca_fuera = self.d_boca > int(self.piso_boca)
        self.candidatos = numero(self.galeria[0], r"GALERIA: (\d+) sorteos candidatos")
        self.combinaciones = numero(self.galeria[0], r"(\d+) combinaciones distintas")

        # Control negativo.
        self.ctl_cero = numero(self.control, r"aceptadas como nuevas: (\d+)/")
        self.ctl_n = numero(self.control, r"aceptadas como nuevas: \d+/(\d+) \(")
        self.ctl_antes = numero(self.control, r"banda de pixeles: (\d+)/")
        self.ctl_posterior = numero(self.control, r"(\S+) % aceptadas como nuevas")

        # Contraste con el autoencoder simple.
        c = self.contraste
        self.reconocible_vae = numero(c, r"reconocible: VAE (\S+) %")
        self.reconocible_ae = numero(c, r"reconocible: VAE \S+ % contra autoencoder sin esquema variacional (\S+) %")
        self.plausible_vae = numero(c, r"cara plausible: VAE (\S+) %")
        self.plausible_ae = numero(c, r"cara plausible: VAE \S+ % contra (\S+) %")
        self.hibrido_vae = numero(c, r"hibrido plausible: VAE (\S+) %")
        self.hibrido_ae = numero(c, r"hibrido plausible: VAE \S+ % contra (\S+) %")
        self.salto_ae = numero(c, r"celdas vecinas: VAE \S+ px contra (\S+) px")
        self.rec_vae = numero(c, r"reconstruccion media: VAE (\S+) px")
        self.rec_ae = numero(c, r"reconstruccion media: VAE \S+ px contra (\S+) px")

        # Aceptacion.
        self.ok = numero(self.resultado, r"\((\d+)/\d+ comprobaciones")
        self.total = numero(self.resultado, r"\(\d+/(\d+) comprobaciones")
        self.estado = numero(self.resultado, r"RESULTADO EJ2: (\w+)")
        if len(self.aceptacion) != int(self.total):
            raise ValueError(
                f"{EJ}/{SALIDA}: hay {len(self.aceptacion)} lineas ACEPTACION OK y el "
                f"resultado declara {self.total} comprobaciones"
            )

        # Pasajes del documento de analisis, leidos y nunca retipeados.
        self.completo = _leer_doc("El dataset es el producto completo")
        self.juicio = _leer_doc("En resumen, solo")
        self.juicio_distinguibles = numero(self.juicio, r"solo (\d+) de las \d+ entradas")
        self.juicio_entradas = numero(self.juicio, r"solo \d+ de las (\d+) entradas")

    def regiones_nuevas(self) -> str:
        fuera = []
        if self.ojos_fuera:
            fuera.append("los ojos")
        if self.boca_fuera:
            fuera.append("la boca")
        if not fuera:
            raise ValueError(f"{EJ}: el emoji nuevo no tiene ninguna region fuera de su piso de ruido")
        return _enumerar(fuera)

    def parecidos(self, a: str, b: str) -> bool:
        return abs(float(a) - float(b)) <= PARECIDO_PUNTOS


# --------------------------------------------------------------------------
# Diapositivas
# --------------------------------------------------------------------------


def _dataset(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Dataset de emojis")
    ancho_img = 4.5
    d.imagen(
        salidas.png(EJ, "emojis.png"), MARGEN_IN, CONTENIDO_Y_IN, ancho_img, Y_FIN - CONTENIDO_Y_IN,
        alt=salidas.alt(EJ, "emojis.png"),
    )
    x = MARGEN_IN + ancho_img + 0.4
    ancho = ANCHO_IN - MARGEN_IN - x
    y = CONTENIDO_Y_IN
    y += _codigo(d, x, y, ancho, [dat.dataset, dat.separacion]) + HUECO
    y += _texto(
        d, x, y, ancho,
        _vinetas(
            [
                f"El dataset se construyó por código, como permite la consigna: un mismo contorno "
                f"de cara, un conjunto de estilos de ojos y otro de estilos de boca, una imagen por "
                f"cada par ({dat.n_emojis} imágenes de {dat.lado} x {dat.lado} píxeles; en el "
                f"montaje, una fila por estilo de ojos y una columna por estilo de boca).",
                "Son caras estilo emoji dibujadas por código, no ilustraciones de los emojis de "
                "Unicode de ninguna plataforma.",
                f"La separación define cuándo una imagen decodificada es reconocible: a lo sumo "
                f"{dat.radio} px de un emoji del dataset (radio de cercanía); el par más parecido "
                f"difiere en {dat.sep_min} px y la mediana es {dat.sep_med} px.",
            ]
        ),
    ) + HUECO
    _texto(d, x, y, ancho, [Parrafo(dat.completo, negrita=True)], tam=TAM_TEXTO)


def _arquitectura(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Arquitectura del VAE")
    enc, dec = dat.encoder, dat.decoder
    cajas = [
        ("Entrada", f"{enc[0]} píxeles", None),
        ("Encoder", f"capas {'-'.join(str(n) for n in enc[1:-1])}", None),
        ("Media y log-varianza", f"{enc[-1]} salidas", COLOR_AMBAR),
        ("Reparametrización", f"z de {dat.latente} dimensiones", COLOR_AMBAR),
        ("Decoder", f"capas {'-'.join(str(n) for n in dec[1:-1])}", None),
        ("Salida", f"{dec[-1]} píxeles", None),
    ]
    hueco = (ANCHO_UTIL - len(cajas) * ANCHO_CAJA) / (len(cajas) - 1)
    y_diag = CONTENIDO_Y_IN + HUECO_CHICO
    for i, (titulo, detalle, relleno) in enumerate(cajas):
        x = MARGEN_IN + i * (ANCHO_CAJA + hueco)
        d.caja(
            x, y_diag, ANCHO_CAJA, ALTO_CAJA,
            [Parrafo(titulo, negrita=True), Parrafo(detalle)],
            tam=TAM_CAJA, relleno=relleno,
        )
        if i < len(cajas) - 1:
            d.texto(x + ANCHO_CAJA, y_diag + HUECO, hueco, 0.6, "→", tam=24)

    lineas = [
        f"Posterior: q(z|x) es una gaussiana con covarianza diagonal; el encoder emite una media y "
        f"un log-varianza por dimensión latente ({dat.encoder[-1]} salidas, el doble de las "
        f"{dat.latente} dimensiones).",
        "Reparametrización: z = mu + eps * sigma, con sigma = exp(0.5 * logvar) y eps de N(0, I); "
        "la aleatoriedad queda fuera de la red y el gradiente llega al encoder.",
        "Prior: p(z) = N(0, I).",
        "Costo: reconstrucción más KL en forma cerrada, 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar), "
        "con peso 1.",
    ]
    y = y_diag + ALTO_CAJA + HUECO
    _texto(d, MARGEN_IN, y, ANCHO_UTIL, _vinetas(lineas), tam=TAM_CUERPO)
    ancho_cfg = ANCHO_UTIL
    alto_cfg = altura_estimada([Parrafo(dat.cfg)], ancho_cfg, tam=TAM_CODIGO, mono=True) + RELLENO_ALTO
    d.codigo(MARGEN_IN, Y_FIN - alto_cfg, ancho_cfg, alto_cfg, [dat.cfg], fuente=FUENTE, tam=TAM_CODIGO)


def _perdida(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: evolución de la pérdida")
    ancho_img = 5.7
    d.imagen(
        salidas.png(EJ, "vae_loss.png"), MARGEN_IN, CONTENIDO_Y_IN, ancho_img, 3.45,
        alt=salidas.alt(EJ, "vae_loss.png"),
    )
    x = MARGEN_IN + ancho_img + 0.3
    ancho = ANCHO_IN - MARGEN_IN - x
    y = CONTENIDO_Y_IN
    y += _codigo(d, x, y, 5.0, dat.tabla, tam=12) + HUECO
    y += _codigo(d, x, y, ancho, [dat.perdida], tam=TAM_CODIGO)
    y_txt = max(y, CONTENIDO_Y_IN + 3.45) + HUECO
    _texto(
        d, MARGEN_IN, y_txt, ANCHO_UTIL,
        _vinetas(
            [
                f"La reconstrucción cae con fuerza al principio y después lentamente, de "
                f"{dat.rec0} a {dat.rec1}; no llega a cero porque los objetivos son grises del "
                f"supersampling y con entropía cruzada un objetivo gris tiene un mínimo positivo.",
                f"El KL parte de {dat.kl0}, sube y baja durante las primeras épocas mientras el "
                f"decoder aprende a usar el código y se estabiliza en torno a {dat.kl1} nats: es lo "
                f"que el encoder gasta por emoji para describirlo con {dat.latente} números. El "
                f"total sigue de cerca a la reconstrucción porque el KL es mucho menor.",
            ]
        ),
        tam=TAM_CUERPO - 1,
    )


def _tecnicas(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Técnicas probadas: esquema variacional frente a autoencoder simple")
    ancho_col = (ANCHO_UTIL - 0.4) / 2
    x_der = MARGEN_IN + ancho_col + 0.4
    alto_enc, alto_fila, hueco = 0.5, 0.7, HUECO_CHICO
    izquierda = [
        f"Misma arquitectura, optimizador {dat.optimizador}, pérdida, {dat.epocas} épocas y "
        f"semilla {dat.semilla}",
        "Sin capa estocástica: el encoder entrega un único código z por imagen",
        "Sin término KL: nada obliga a los códigos a parecerse al prior",
    ]
    derecha = [
        "Encoder estocástico: media y log-varianza, y z = mu + eps * sigma",
        "Costo: reconstrucción + KL de forma cerrada, con peso 1",
        "El KL acerca las posteriores al prior N(0, I)",
    ]
    d.caja(MARGEN_IN, CONTENIDO_Y_IN, ancho_col, alto_enc, [Parrafo("Autoencoder simple", negrita=True)], tam=TAM_CUERPO)
    d.caja(
        x_der, CONTENIDO_Y_IN, ancho_col, alto_enc,
        [Parrafo("Autoencoder variacional (VAE)", negrita=True)], tam=TAM_CUERPO, relleno=COLOR_AMBAR,
    )
    y = CONTENIDO_Y_IN + alto_enc + hueco
    for texto_i, texto_d in zip(izquierda, derecha):
        d.caja(MARGEN_IN, y, ancho_col, alto_fila, texto_i, tam=TAM_CAJA, relleno=COLOR_FONDO_CIFRA)
        d.caja(x_der, y, ancho_col, alto_fila, texto_d, tam=TAM_CAJA, relleno=COLOR_FONDO_CIFRA)
        y += alto_fila + hueco
    y += HUECO_CHICO
    _texto(
        d, MARGEN_IN, y, ANCHO_UTIL,
        _vinetas(
            [
                f"Técnicas aplicadas: truco de reparametrización, KL en forma cerrada, entropía "
                f"cruzada binaria con salida sigmoide, optimizador {dat.optimizador} y entrenamiento "
                f"por lote completo con ruido nuevo en cada época.",
                f"La configuración elegida cumple {dat.total} comprobaciones explícitas de calidad "
                f"generativa (diapositiva de comprobaciones de aceptación); el contraste es un "
                f"autoencoder idéntico sin el esquema variacional.",
                "No se hizo un estudio de arquitecturas para el Ej 2: la consigna no lo pide y la "
                "configuración está congelada en el código.",
            ]
        ),
        tam=TAM_TEXTO,
    )


def _reconstruccion(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: reconstrucción y plano latente")
    ancho_col = (ANCHO_UTIL - 0.4) / 2
    x_der = MARGEN_IN + ancho_col + 0.4
    alto_img = 3.0
    d.imagen(
        salidas.png(EJ, "vae_reconstruction.png"), MARGEN_IN, CONTENIDO_Y_IN, ancho_col, alto_img,
        alt=salidas.alt(EJ, "vae_reconstruction.png"),
    )
    d.imagen(
        salidas.png(EJ, "vae_manifold.png"), x_der, CONTENIDO_Y_IN, ancho_col, alto_img,
        alt=salidas.alt(EJ, "vae_manifold.png"),
    )
    y = CONTENIDO_Y_IN + alto_img + HUECO_CHICO
    alto_a = _texto(
        d, MARGEN_IN, y, ancho_col,
        [
            Parrafo(
                f"Filas impares: los originales; filas pares: las reconstrucciones desde la media "
                f"del encoder, con sus píxeles incorrectos. Se leen como el emoji original; los "
                f"mayores errores (hasta {dat.rec_max} px) están en las bocas grin y flat de casi "
                f"todos los estilos de ojos."
            )
        ],
        tam=TAM_PIE,
    )
    alto_b = _texto(
        d, x_der, y, ancho_col,
        [
            Parrafo(
                "El eje horizontal lleva los ojos y el vertical la boca, con transiciones graduales "
                "salvo en dos franjas: las columnas centrales, donde los ojos pasan a cruces y aros "
                "con celdas de mezcla en gris, y las filas del tercio inferior, donde aparece la "
                "boca abierta con su mancha negra."
            )
        ],
        tam=TAM_PIE,
    )
    y += max(alto_a, alto_b) + HUECO
    _codigo(d, MARGEN_IN, y, ANCHO_UTIL, [dat.recon, dat.manifold])


def _contraste(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: contraste con el autoencoder simple")
    ancho_img = 3.7
    alto_img = 3.2
    d.imagen(
        salidas.png(EJ, "plain_manifold.png"), MARGEN_IN, CONTENIDO_Y_IN, ancho_img, alto_img,
        alt=salidas.alt(EJ, "plain_manifold.png"),
    )
    x = MARGEN_IN + ancho_img + 0.3
    ancho = ANCHO_IN - MARGEN_IN - x

    if float(dat.rec_ae) <= float(dat.rec_vae):
        recon = (
            f"El autoencoder simple reconstruye igual de bien o mejor ({dat.rec_ae} px contra "
            f"{dat.rec_vae} px por emoji): un buen autoencoder no alcanza para poder generar."
        )
    else:
        recon = (
            f"El autoencoder simple reconstruye algo peor ({dat.rec_ae} px contra {dat.rec_vae} px "
            f"por emoji)."
        )
    if dat.parecidos(dat.hibrido_vae, dat.hibrido_ae):
        hibrido = (
            f"Dicho con claridad: la proporción de sorteos aceptados como híbridos es casi la "
            f"misma en los dos modelos ({dat.hibrido_vae} % contra {dat.hibrido_ae} %), así que "
            f"no los separa; lo que los separa es la plausibilidad."
        )
    else:
        hibrido = (
            f"La proporción de sorteos aceptados como híbridos difiere ({dat.hibrido_vae} % "
            f"contra {dat.hibrido_ae} %)."
        )
    y = CONTENIDO_Y_IN
    _texto(
        d, x, y, ancho,
        _vinetas(
            [
                recon,
                f"Pero sus códigos no están distribuidos como el prior: al sortear z de N(0, I) "
                f"solo {dat.reconocible_ae} % cae donde su decoder produce un emoji reconocible, "
                f"contra {dat.reconocible_vae} % del VAE, y solo {dat.plausible_ae} % decodifica a "
                f"una cara plausible, contra {dat.plausible_vae} %.",
                f"En su grilla hay caras razonables arriba y a la izquierda, pero hacia la esquina "
                f"inferior derecha aparecen imágenes de fondo negro y celdas casi ilegibles; el "
                f"salto medio entre vecinas es {dat.salto_ae} px contra {dat.salto_vae} px del VAE.",
                hibrido,
            ]
        ),
        tam=TAM_PIE,
    )
    y = CONTENIDO_Y_IN + alto_img + HUECO
    _codigo(d, MARGEN_IN, y, ANCHO_UTIL, [dat.contraste, dat.vae_prior, dat.ae_prior], tam=TAM_CODIGO_CHICO)


def _emoji_nuevo(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: emoji nuevo")
    ancho_img = 6.0
    alto_img = 2.25
    d.imagen(
        salidas.png(EJ, "vae_new_emoji.png"), MARGEN_IN, CONTENIDO_Y_IN, ancho_img, alto_img,
        alt=salidas.alt(EJ, "vae_new_emoji.png"),
    )
    y = CONTENIDO_Y_IN + alto_img + HUECO_CHICO
    y += _texto(
        d, MARGEN_IN, y, ancho_img,
        [
            Parrafo(
                "Arriba, los emojis de la galería en orden de margen; abajo, el emoji más cercano "
                "de cada uno con su distancia en píxeles. Casi todos tienen ojos de anillo moteado; "
                "el último tiene los puntos de dots."
            )
        ],
        tam=TAM_PIE,
    )
    x = MARGEN_IN + ancho_img + 0.3
    ancho = ANCHO_IN - MARGEN_IN - x

    if dat.ojos_fuera and not dat.boca_fuera:
        origen = (
            f"la novedad viene de los ojos ({dat.d_ojos} px del estilo más cercano, con piso "
            f"{dat.piso_ojos} px); la boca queda dentro de su piso ({dat.d_boca} px contra "
            f"{dat.piso_boca} px)"
        )
    else:
        origen = f"las regiones fuera de su piso son {dat.regiones_nuevas()}"
    d_centro = (
        f"Un emoji nuevo es una región, ojos o boca, más lejos que el piso de ruido de "
        f"reconstrucción de esa región de todos los estilos de la región, con la cara todavía "
        f"plausible. Como el dataset es el producto completo de ojos por boca, la novedad solo "
        f"puede ser una región, nunca una combinación de estilos conocidos: lo que se acepta es un híbrido, una región "
        f"fuera del piso de ruido del decodificador."
    )
    d_titular = (
        f"El titular es el sorteo de mayor margen entre combinaciones distintas, elegido por una "
        f"regla y no a ojo: sorteo {dat.sorteo} de {dat.sorteos}, a {dat.dist_cercano} px de "
        f"{dat.cercano}; {origen}. La distancia global supera el radio de cercanía ({dat.radio} "
        f"px), pero no basta como criterio: el decodificador falla hasta {dat.rec_max} px al "
        f"reconstruir un emoji real."
    )
    d_primero = (
        f"La primera versión del criterio aceptaba una banda de distancia global al emoji más "
        f"cercano, del tamaño del ruido del decodificador: aceptaba reconstrucciones de emojis "
        f"reales ({dat.ctl_antes}/{dat.ctl_n} en el control negativo) y la muestra que mostraba "
        f"era una copia degradada. Por eso se descartó."
    )
    y_der = CONTENIDO_Y_IN + _texto(
        d, x, CONTENIDO_Y_IN, ancho, _vinetas([d_centro, d_titular, d_primero]), tam=TAM_PIE
    )
    _codigo(d, MARGEN_IN, max(y, y_der) + HUECO, ANCHO_UTIL, [dat.nuevo, dat.regiones], tam=TAM_CODIGO_CHICO)


def _criterio_control(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: criterio de novedad y control negativo")
    ancho = (ANCHO_UTIL - 2 * 0.3) / 3
    cifras = [
        (f"{dat.ctl_cero}/{dat.ctl_n}", "reconstrucciones del entrenamiento aceptadas como nuevas"),
        (f"{dat.ctl_antes}/{dat.ctl_n}", "aceptadas por el criterio anterior de banda de píxeles"),
        (f"{dat.ctl_posterior} %", "de las reconstrucciones estocásticas aceptadas como nuevas"),
    ]
    for i, (valor, etiqueta) in enumerate(cifras):
        d.cifra(MARGEN_IN + i * (ancho + 0.3), CONTENIDO_Y_IN, ancho, ALTO_CIFRA, valor, etiqueta)
    y = CONTENIDO_Y_IN + ALTO_CIFRA + HUECO
    y += _codigo(d, MARGEN_IN, y, ANCHO_UTIL, dat.galeria) + HUECO_CHICO
    y += _codigo(d, MARGEN_IN, y, ANCHO_UTIL, [dat.control]) + HUECO
    _texto(
        d, MARGEN_IN, y, ANCHO_UTIL,
        _vinetas(
            [
                f"El criterio no acepta ninguna de las {dat.ctl_n} reconstrucciones del "
                f"entrenamiento, pero ese cero es cierto por construcción: los pisos se miden sobre "
                f"esas mismas reconstrucciones.",
                f"La evidencia independiente es la proporción de reconstrucciones estocásticas "
                f"aceptadas ({dat.ctl_posterior} %): pequeña pero no nula, así que el criterio no "
                f"acepta cualquier salida del decodificador, y tampoco es perfecto. La banda "
                f"anterior habría aceptado {dat.ctl_antes} de {dat.ctl_n}.",
            ]
        ),
        tam=TAM_PIE,
    )


def _vecino(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: emoji nuevo frente a su vecino más cercano")
    ancho_blk = 5.3
    _codigo(d, MARGEN_IN, CONTENIDO_Y_IN, ancho_blk, dat.par, tam=TAM_CODIGO)
    x = MARGEN_IN + ancho_blk + 0.4
    ancho = ANCHO_IN - MARGEN_IN - x
    y = CONTENIDO_Y_IN
    y += _texto(d, x, y, ancho, [Parrafo("Juicio visual de los autores", negrita=True)], tam=TAM_CUERPO) + HUECO_CHICO
    y += _texto(d, x, y, ancho, [Parrafo(dat.juicio)], tam=TAM_TEXTO) + HUECO
    _texto(
        d, x, y, ancho,
        [
            Parrafo(
                f"Es un juicio subjetivo, no una medida: de las {dat.juicio_entradas} entradas de "
                f"la galería solo {dat.juicio_distinguibles} se distinguen a simple vista de su "
                f"emoji más cercano, y debe leerse junto con la figura de la diapositiva del emoji nuevo.",
                color=COLOR_AMBAR,
            )
        ],
        tam=TAM_TEXTO,
    )


def _aceptacion(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: comprobaciones de aceptación")
    ancho_blk = 9.7
    y = CONTENIDO_Y_IN
    y += _codigo(d, MARGEN_IN, y, ancho_blk, dat.aceptacion, tam=TAM_CODIGO) + HUECO
    x_cifra = MARGEN_IN + ancho_blk + 0.3
    d.cifra(
        x_cifra, CONTENIDO_Y_IN, ANCHO_IN - MARGEN_IN - x_cifra, ALTO_CIFRA + 0.3,
        f"{dat.ok}/{dat.total}", "comprobaciones de aceptación",
    )
    y += _codigo(d, MARGEN_IN, y, ANCHO_UTIL, [dat.resultado], tam=TAM_CODIGO) + HUECO
    _texto(
        d, MARGEN_IN, y, ANCHO_UTIL,
        _vinetas(
            [
                "Los umbrales son convenciones de este trabajo, fijadas con margen respecto de lo "
                "que logra la configuración y sin ajustarlos para aprobar.",
                "Las comprobaciones de control negativo y de plausibilidad se agregaron para cerrar "
                "el hallazgo de la revisión sobre el emoji nuevo, y la de híbridos del prior "
                "reemplazó a una que el autoencoder simple también cumplía.",
            ]
        ),
        tam=TAM_CUERPO,
    )


def _conclusiones(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Conclusiones y límites")
    ancho_cifra = (ANCHO_UTIL - 2 * 0.3) / 3
    cifras = [
        (f"{dat.rec_media} px", "error medio de reconstrucción por emoji"),
        (f"{dat.reconocible_vae} %", "de los sorteos del prior son emojis reconocibles"),
        (f"{dat.ctl_cero}/{dat.ctl_n}", "reconstrucciones del entrenamiento aceptadas como nuevas"),
    ]
    alto_cifra = 1.3
    for i, (valor, etiqueta) in enumerate(cifras):
        d.cifra(MARGEN_IN + i * (ancho_cifra + 0.3), CONTENIDO_Y_IN, ancho_cifra, alto_cifra, valor, etiqueta)

    ancho_col = (ANCHO_UTIL - 0.4) / 2
    x_der = MARGEN_IN + ancho_col + 0.4
    y = CONTENIDO_Y_IN + alto_cifra + HUECO
    conclusiones = [
        f"El VAE reconstruye los {dat.n_emojis} emojis con un error medio de {dat.rec_media} px "
        f"(máximo {dat.rec_max} px); {dat.rec_sin} no tienen píxeles incorrectos.",
        f"El plano latente decodificado es un mapa de caras: {dat.celdas_rec} % de las "
        f"{dat.celdas} celdas son reconocibles y el salto medio entre vecinas es {dat.salto_vae} px.",
        f"El esquema variacional vuelve muestreable el espacio latente: {dat.reconocible_vae} % de "
        f"los sorteos del prior son emojis reconocibles y {dat.plausible_vae} % caras plausibles, "
        f"contra {dat.reconocible_ae} % y {dat.plausible_ae} % del autoencoder simple.",
        f"El emoji nuevo es un híbrido: su región fuera de estilo ({dat.regiones_nuevas()}) supera "
        f"el piso de ruido del decodificador. El control negativo no acepta ninguna de las "
        f"{dat.ctl_n} reconstrucciones y acepta {dat.ctl_posterior} % de las estocásticas; "
        f"resultado del Ej 2: {dat.estado}, {dat.ok}/{dat.total} comprobaciones.",
    ]
    limites = [
        f"Dataset construido de {dat.n_emojis} imágenes con dos factores, producto completo: la "
        f"novedad solo puede ser una región, no una combinación de estilos.",
        f"El piso de ruido de la boca ({dat.piso_boca} px) supera la separación entre estilos de "
        f"boca ({dat.sep_boca} px): casi toda la novedad está en los ojos.",
        f"La proporción de híbridos no separa al VAE del autoencoder simple ({dat.hibrido_vae} % "
        f"contra {dat.hibrido_ae} %); lo que los separa es la plausibilidad.",
        f"El juicio final de novedad es visual y subjetivo: solo {dat.juicio_distinguibles} de las "
        f"{dat.juicio_entradas} entradas de la galería se distinguen a simple vista.",
        "Una sola arquitectura y una sola semilla en el contraste; no hubo estudio de "
        "arquitecturas porque la consigna no lo pide.",
    ]
    alto_titulo = 0.35
    _texto(d, MARGEN_IN, y, ancho_col, [Parrafo("Conclusiones", negrita=True)], tam=TAM_TEXTO)
    _texto(d, x_der, y, ancho_col, [Parrafo("Límites", negrita=True, color=COLOR_AMBAR)], tam=TAM_TEXTO)
    y += alto_titulo + HUECO_CHICO
    _texto(d, MARGEN_IN, y, ancho_col, _vinetas(conclusiones), tam=TAM_PIE)
    _texto(d, x_der, y, ancho_col, _vinetas(limites), tam=TAM_PIE)


def agregar(pres, salidas) -> None:
    dat = _Datos(salidas)
    _dataset(pres, salidas, dat)
    _arquitectura(pres, dat)
    _perdida(pres, salidas, dat)
