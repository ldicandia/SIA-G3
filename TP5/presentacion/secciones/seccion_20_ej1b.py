"""Seccion 20: Ejercicio 1b (denoising autoencoder).

Ninguna cifra se escribe a mano: todas salen de los reportes, tablas y CSV que
regenera scripts/regenerar.sh, leidos con lector.Salidas. Las lineas que se
muestran literalmente son lineas exactas de ej1b/salida-ej1b.txt y de
ej1b/salida-dae-sweep.txt. Las frases sobre las figuras parafrasean
docs/ej1b-analisis.md y se contrastaron con la imagen abierta.
"""

from __future__ import annotations

import re

from lector import campo, numero
from pptx_minimo import ALTO_IN, ANCHO_IN, COLOR_AMBAR, CONTENIDO_Y_IN, MARGEN_IN, Parrafo

EJ = "ej1b"
SALIDA = "salida-ej1b.txt"
BARRIDO = "salida-dae-sweep.txt"
FUENTE = f"{EJ}/{SALIDA}"
FUENTE_BARRIDO = f"{EJ}/{BARRIDO}"

ANCHO_UTIL = ANCHO_IN - 2 * MARGEN_IN
Y_FIN = ALTO_IN - MARGEN_IN
TAM_CUERPO = 16
TAM_TEXTO = 15
TAM_PIE = 14
TAM_CODIGO = 11
TAM_TABLA = 11

# Unidades y umbrales de calculo (no son cifras del resultado).
CIEN = 100
MITAD = 0.5
HUECO = 0.15
HUECO_CHICO = 0.1

# Pesos de las columnas de las tablas de los estudios de dae_comparison.md.
PESOS_TABLA = [1.7, 0.9, 1.0, 0.9, 0.9, 0.9, 1.3, 1.2]
ANCHO_TABLA_IZQ = 7.6

# Alturas de los bloques de texto y de las imagenes de cada diapositiva.
ALTO_CAJA = 1.2
ANCHO_CAJA = 1.7
Y_CONFIG = 6.4
ALTO_CONFIG = 0.6
ANCHO_IMG_DER = 4.53
ALTO_IMG_DER = 2.72
ALTO_IMG_LAT = 2.2
ALTO_BLOQUE_SEL = 0.45
ALTO_FIG_NIVEL = 3.0
ALTO_CAPTION_NIVEL = 1.3
ANCHO_BLOQUE_TABLA = 7.3
ALTO_BLOQUE_TABLA = 2.55
TAM_BLOQUE_TABLA = 16
ALTO_BLOQUE_TITULAR = 0.55
ALTO_IMG_EJEMPLO = 2.0
ALTO_CAPTION_EJEMPLO = 0.6
ALTO_CIFRA = 1.7
ANCHO_CIFRA = 3.4
ANCHO_TEXTO_CONCL = 8.6


# --------------------------------------------------------------------------
# Utilidades de lectura
# --------------------------------------------------------------------------


def _px(n) -> str:
    return "píxel" if str(n) == "1" else "píxeles"


def _enumerar(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


def _pct(fraccion) -> str:
    """Fraccion (0 a 1) como porcentaje con un decimal, igual que la tabla impresa."""
    valor = float(fraccion) * CIEN
    return f"{valor:.1f}"


def _calendario(valor: str) -> str:
    """Calendario de ruido de entrenamiento tal como lo imprime el reporte."""
    m = re.fullmatch(r"fix(\S+)", valor)
    if m:
        return f"fijo en {m.group(1)}"
    m = re.fullmatch(r"u(\S+)-(\S+)", valor)
    if m:
        return f"uniforme entre {m.group(1)} y {m.group(2)}"
    return valor


def _corto(celda: str) -> str:
    """Encabezados acortados de las tablas de los estudios (el cuerpo va tal cual)."""
    celda = celda.replace("semillas elegibles", "semillas ok")
    celda = celda.replace("nivel ", "")
    celda = celda.replace("(mediana)", "(med.)")
    celda = celda.replace("perdida", "pérdida")
    return celda


def _k_de_n(celda: str) -> tuple[int, int]:
    return int(numero(celda, r"^(\d+)/")), int(numero(celda, r"/(\d+)$"))


def _rango_semillas(puntajes: dict[str, list[float]], variante: str) -> tuple[float, float]:
    valores = puntajes.get(variante)
    if not valores:
        raise ValueError(f"{EJ}/{BARRIDO}: no hay corridas de la variante {variante!r}")
    return min(valores), max(valores)


def _estudio(salidas, nombre: str) -> tuple[list[str], list[list[str]]]:
    encabezado, filas = salidas.tabla_md(EJ, "dae_comparison.md", nombre)
    return encabezado, filas


class _Datos:
    """Todo lo que las diapositivas toman de los reportes, las tablas y el CSV."""

    def __init__(self, salidas) -> None:
        self.cfg = salidas.linea(EJ, SALIDA, "configuracion:")
        self.evaluacion = salidas.linea(EJ, SALIDA, "evaluacion:")
        self.tabla = salidas.bloque(EJ, SALIDA, "nivel  entrada_px", 9)
        self.titular = salidas.linea(EJ, SALIDA, "NIVEL MAXIMO ELIMINADO:")
        self.seleccion = salidas.linea(EJ, BARRIDO, "SELECCION:")
        self.corridas = numero(
            salidas.linea(EJ, BARRIDO, "dae-sweep:"), r"dae-sweep: (\d+) corridas"
        )

        self.nombre = campo(self.cfg, "configuracion")
        self.tamanios = [int(n) for n in campo(self.cfg, "arquitectura").split("-")]
        self.perdida = campo(self.cfg, "perdida")
        self.optimizador = campo(self.cfg, "optimizador")
        self.epocas = campo(self.cfg, "epocas")
        self.copias = campo(self.cfg, "copias por epoca")
        self.ruido_entrenamiento = campo(self.cfg, "ruido de entrenamiento")
        self.semilla = campo(self.cfg, "semilla")

        self.nivel_max = numero(self.titular, r"ELIMINADO: (\S+) \(")
        self.umbral = numero(self.titular, r"al menos el (\d+) %")
        self.tol = numero(self.titular, r"a lo sumo (\d+) pixel")
        self.px_tol = _px(self.tol)

        # CSV de niveles: define los niveles y las fracciones con las que se eligen ejemplos.
        self.csv = salidas.csv(EJ, "dae_levels.csv")
        self.niveles = [r["level"] for r in self.csv]
        self.frac = {r["level"]: float(r["frac_within_1px"]) for r in self.csv}
        self.frac_entrada = {r["level"]: float(r["input_frac_within_1px"]) for r in self.csv}

        # Tabla impresa: se citan sus cifras para que coincidan con el bloque literal.
        self.impreso: dict[str, list[str]] = {}
        for linea in self.tabla[1:]:
            partes = linea.split()
            self.impreso[partes[0]] = partes[1:]
        if sorted(self.impreso) != sorted(self.niveles):
            raise ValueError(
                f"{EJ}: los niveles de la tabla impresa {sorted(self.impreso)} no coinciden "
                f"con los de dae_levels.csv {sorted(self.niveles)}"
            )

        n_dist = int(numero(self.evaluacion, r"evaluacion: (\d+) distorsiones"))
        self.n_dist = n_dist
        self.muestras = int(self.csv[0]["n_samples"])
        self.patrones = self.muestras // n_dist

        self.enc_lat, self.lat = _estudio(salidas, "lat")
        self.enc_hid, self.hid = _estudio(salidas, "hid")
        self.enc_noise, self.noise = _estudio(salidas, "noise")
        for enc in (self.enc_hid, self.enc_noise):
            if enc != self.enc_lat:
                raise ValueError(f"{EJ}/dae_comparison.md: las tablas no tienen los mismos encabezados")
        if len(self.enc_lat) != len(PESOS_TABLA):
            raise ValueError(
                f"{EJ}/dae_comparison.md: las tablas tienen {len(self.enc_lat)} columnas "
                f"y la diapositiva espera {len(PESOS_TABLA)}"
            )

        # Puntajes de cada semilla, de las lineas "variante-sN | puntaje X | elegible ...".
        self.puntajes: dict[str, list[float]] = {}
        for linea in salidas.lineas(EJ, BARRIDO):
            m = re.match(r"^(\S+)-s\d+ \| puntaje (\S+) \| elegible", linea)
            if m:
                self.puntajes.setdefault(m.group(1), []).append(float(m.group(2)))

    # -- accesos ---------------------------------------------------------

    def fila_nivel(self, nivel: str) -> list[str]:
        """[entrada_px, salida_px, max_px, dentro_1px_%] como los imprime el reporte."""
        return self.impreso[nivel]

    def nivel_de(self, texto: str) -> str:
        """Nivel del CSV cuyo valor numerico coincide con `texto`."""
        for nivel in self.niveles:
            if float(nivel) == float(texto):
                return nivel
        raise ValueError(f"{EJ}/dae_levels.csv: no hay el nivel {texto!r}")

    def primer_nivel_bajo(self, tabla: dict[str, float], limite: float) -> str:
        for nivel in self.niveles:
            if tabla[nivel] < limite:
                return nivel
        raise ValueError(f"{EJ}/dae_levels.csv: ningun nivel baja de {limite}")

    def puntaje(self, fila: list[str]) -> float:
        return float(fila[2])


# --------------------------------------------------------------------------
# Diapositivas
# --------------------------------------------------------------------------


def _arquitectura(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Arquitectura del denoising autoencoder")
    tam = dat.tamanios
    i_lat = tam.index(min(tam))
    latente = tam[i_lat]
    codificador = "-".join(str(n) for n in tam[1:i_lat])
    decodificador = "-".join(str(n) for n in tam[i_lat + 1 : -1])

    cajas = [
        ("Entrada ruidosa", f"{tam[0]} píxeles", None),
        ("Encoder", f"capas {codificador}", None),
        ("Latente", f"{latente} neuronas", COLOR_AMBAR),
        ("Decoder", f"capas {decodificador}", None),
        ("Salida", f"{tam[-1]} píxeles", None),
        ("Objetivo", "patrón limpio", COLOR_AMBAR),
    ]
    hueco = (ANCHO_UTIL - len(cajas) * ANCHO_CAJA) / (len(cajas) - 1)
    y_diag = CONTENIDO_Y_IN + HUECO_CHICO
    for i, (titulo, detalle, relleno) in enumerate(cajas):
        x = MARGEN_IN + i * (ANCHO_CAJA + hueco)
        d.caja(
            x, y_diag, ANCHO_CAJA, ALTO_CAJA,
            [Parrafo(titulo, negrita=True), Parrafo(detalle)],
            tam=TAM_PIE, relleno=relleno,
        )
        if i < len(cajas) - 1:
            flecha = "↔" if i == len(cajas) - 2 else "→"
            d.texto(x + ANCHO_CAJA, y_diag + HUECO, hueco, 0.6, flecha, tam=24)

    nombre_perdida = {"BCE": "entropía cruzada binaria", "MSE": "error cuadrático medio"}.get(
        dat.perdida, dat.perdida
    )
    viñetas = [
        "La red recibe el patrón distorsionado y se entrena para devolver el patrón limpio: lo que "
        "no forma parte del patrón, el ruido, no puede atravesar el cuello de la red.",
        f"Modelo de ruido: cada píxel se invierte de forma independiente con una probabilidad, el "
        f"nivel. El ruido de entrenamiento es {_calendario(dat.ruido_entrenamiento)} y se sortea de "
        f"nuevo en cada época, con {dat.copias} copias de los {dat.patrones} patrones por época.",
        f"Pérdida {dat.perdida} ({nombre_perdida}), optimizador {dat.optimizador}, {dat.epocas} "
        f"épocas, semilla {dat.semilla}.",
        f"El tamaño del latente es un factor de diseño: la consigna no lo impone como en el Ej 1a, "
        f"y la configuración elegida usa {latente} neuronas.",
    ]
    y_vin = y_diag + ALTO_CAJA + HUECO
    d.texto(
        MARGEN_IN, y_vin, ANCHO_UTIL, Y_CONFIG - HUECO_CHICO - y_vin,
        [Parrafo(t, vineta=True) for t in viñetas], tam=TAM_CUERPO,
    )
    d.codigo(MARGEN_IN, Y_CONFIG, ANCHO_UTIL, ALTO_CONFIG, [dat.cfg], fuente=FUENTE, tam=TAM_CODIGO)


def _tecnicas_latente(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Técnicas probadas: tamaño del latente y capacidad del encoder")
    encabezado = [_corto(c) for c in dat.enc_lat]

    y = CONTENIDO_Y_IN
    alto_lat = d.tabla(
        MARGEN_IN, y, ANCHO_TABLA_IZQ, [encabezado] + dat.lat, anchos=PESOS_TABLA, tam=TAM_TABLA
    )
    y += alto_lat + HUECO
    alto_hid = d.tabla(
        MARGEN_IN, y, ANCHO_TABLA_IZQ, [encabezado] + dat.hid, anchos=PESOS_TABLA, tam=TAM_TABLA
    )
    y += alto_hid + HUECO_CHICO

    # Que factor separa mas: la diferencia entre el mejor y el peor puntaje de cada estudio.
    p_lat = [dat.puntaje(f) for f in dat.lat]
    p_hid = [dat.puntaje(f) for f in dat.hid]
    d_lat = max(p_lat) - min(p_lat)
    d_hid = max(p_hid) - min(p_hid)
    if d_lat >= d_hid:
        factor = "El tamaño del latente"
    else:
        factor = "La capacidad del encoder"
    d.texto(
        MARGEN_IN, y, ANCHO_TABLA_IZQ, Y_FIN - y,
        [
            Parrafo(
                "Arriba, estudio lat (tamaño del latente); abajo, estudio hid (capacidad del "
                "encoder). El puntaje es la media de píxeles incorrectos de la salida en los "
                "niveles de selección: menor es mejor.",
            )
        ],
        tam=TAM_PIE,
    )

    x = MARGEN_IN + ANCHO_TABLA_IZQ + 0.3
    ancho = ANCHO_IN - MARGEN_IN - x
    alto_img = ALTO_IMG_LAT
    d.imagen(
        salidas.png(EJ, "dae_curves_lat_loss.png"), x, CONTENIDO_Y_IN, ancho, alto_img,
        alt=salidas.alt(EJ, "dae_curves_lat_loss.png"),
    )

    # Los dos latentes mas grandes: la diferencia contra la dispersion entre semillas.
    mayor = dat.lat[-1]
    anterior = dat.lat[-2]
    dif = abs(dat.puntaje(mayor) - dat.puntaje(anterior))
    a_min, a_max = _rango_semillas(dat.puntajes, anterior[0])
    m_min, m_max = _rango_semillas(dat.puntajes, mayor[0])
    dispersion = max(a_max - a_min, m_max - m_min)
    if dif <= dispersion:
        cierre = (
            f"queda dentro de la dispersión entre semillas: no hay evidencia de que {mayor[0]} "
            f"sea mejor que {anterior[0]}."
        )
    else:
        cierre = "supera la dispersión entre semillas."
    y_txt = CONTENIDO_Y_IN + alto_img + HUECO_CHICO
    d.texto(
        x, y_txt, ancho, Y_FIN - y_txt,
        [
            Parrafo(
                f"{factor} es el factor que más separa los resultados: el puntaje varía "
                f"{d_lat:.4f} en lat y {d_hid:.4f} en hid.",
                vineta=True,
            ),
            Parrafo(
                f"Entre {anterior[0]} y {mayor[0]} la diferencia ({dif:.4f}) {cierre}",
                vineta=True,
            ),
            Parrafo(
                f"Curva de pérdida de la primera semilla: la de {dat.lat[0][0]} queda por encima de "
                f"las demás y las de {dat.lat[-2][0]} y {dat.lat[-1][0]} casi se superponen.",
                vineta=True,
            ),
        ],
        tam=TAM_PIE,
    )


def _tecnicas_ruido(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Técnicas probadas: ruido de entrenamiento y selección")
    encabezado = [_corto(c) for c in dat.enc_lat]
    ancho_tab = ANCHO_TABLA_IZQ
    alto_tab = d.tabla(
        MARGEN_IN, CONTENIDO_Y_IN, ancho_tab, [encabezado] + dat.noise,
        anchos=PESOS_TABLA, tam=TAM_TABLA,
    )
    x = MARGEN_IN + ancho_tab + 0.3
    ancho = ANCHO_IN - MARGEN_IN - x
    alto_img = ancho / ANCHO_IMG_DER * ALTO_IMG_DER
    d.imagen(
        salidas.png(EJ, "dae_curves_noise_loss.png"), x, CONTENIDO_Y_IN, ancho, alto_img,
        alt=salidas.alt(EJ, "dae_curves_noise_loss.png"),
    )

    y = CONTENIDO_Y_IN + max(alto_tab, alto_img) + HUECO
    d.codigo(MARGEN_IN, y, ANCHO_UTIL, ALTO_BLOQUE_SEL, [dat.seleccion], fuente=FUENTE_BARRIDO, tam=TAM_CODIGO)
    y += ALTO_BLOQUE_SEL + HUECO

    perdidas = [(float(f[7]), f[0]) for f in dat.noise]
    piso_bajo = min(perdidas)[1]
    piso_alto = max(perdidas)[1]
    ancho_regla = 5.6
    d.texto(
        MARGEN_IN, y, ancho_regla, Y_FIN - y,
        [
            "elegible = todas las semillas aprenden los patrones limpios",
            "elegida = menor mediana de puntaje entre las elegibles",
            f"corridas = {dat.corridas}",
        ],
        tam=TAM_CODIGO, mono=True,
    )
    x_txt = MARGEN_IN + ancho_regla + 0.2
    d.texto(
        x_txt, y, ANCHO_IN - MARGEN_IN - x_txt, Y_FIN - y,
        [
            Parrafo(
                "Las pérdidas finales de variantes con distinto calendario de ruido no son "
                "comparables entre sí: un lote con más ruido tiene un piso de pérdida más alto "
                "aunque la red corrija mejor.",
                vineta=True,
            ),
            Parrafo(
                f"En la curva, los pisos quedan ordenados por la cantidad de ruido del lote "
                f"({piso_bajo} lo más abajo, {piso_alto} lo más arriba), no por la calidad de "
                "la corrección.",
                vineta=True,
            ),
        ],
        tam=TAM_PIE,
    )


def _distorsion(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Resultados: distorsión de las entradas")
    alto = Y_FIN - CONTENIDO_Y_IN
    ancho_img = 4.3
    d.imagen(
        salidas.png(EJ, "noise_examples.png"), MARGEN_IN, CONTENIDO_Y_IN, ancho_img, alto,
        alt=salidas.alt(EJ, "noise_examples.png"),
    )
    x = MARGEN_IN + ancho_img + 0.4
    ancho = ANCHO_IN - MARGEN_IN - x
    pixeles = dat.tamanios[0]
    primero = dat.niveles[1]
    # Indice del nivel en el que la mayoria de las entradas ya cuesta reconocer.
    dificil = dat.niveles[-3]
    ultimo = dat.niveles[-1]
    fila_primero = dat.fila_nivel(primero)
    fila_ultimo = dat.fila_nivel(ultimo)
    d.texto(
        x, CONTENIDO_Y_IN, ancho, alto,
        [
            Parrafo(
                "Modelo de ruido: cada píxel se invierte de forma independiente con probabilidad "
                f"igual al nivel p, para los {pixeles} píxeles de cada letra.",
                vineta=True,
            ),
            Parrafo(
                f"Con p igual a un medio, el último nivel evaluado ({ultimo}), el valor de cada "
                "píxel de entrada es cero o uno con igual probabilidad sea cual sea el patrón: la "
                "entrada no lleva información sobre el patrón y ningún método podría recuperarlo.",
                vineta=True,
            ),
            Parrafo(
                f"Niveles evaluados: {_enumerar(dat.niveles)}. Con nivel {dat.niveles[0]} la entrada "
                "es el patrón limpio.",
                vineta=True,
            ),
            Parrafo(
                f"Los píxeles incorrectos medios de la entrada crecen de {fila_primero[0]} en "
                f"{primero} a {fila_ultimo[0]} en {ultimo} (media sobre {dat.muestras} muestras "
                "por nivel).",
                vineta=True,
            ),
            Parrafo(
                f"En la figura, con el pie de cada letra como cantidad de píxeles invertidos: en "
                f"{primero} las letras se leen casi sin esfuerzo y desde {dificil} cuesta reconocer "
                "la mayoría de las entradas.",
                vineta=True,
            ),
        ],
        tam=TAM_PIE,
    )


def _error_por_nivel(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Resultados: error por nivel de ruido")
    ancho = 6.0
    x_izq = MARGEN_IN
    x_der = ANCHO_IN - MARGEN_IN - ancho
    y_fig = CONTENIDO_Y_IN
    y_cap = y_fig + ALTO_FIG_NIVEL + HUECO_CHICO

    d.imagen(
        salidas.png(EJ, "dae_error_vs_noise.png"), x_izq, y_fig, ancho, ALTO_FIG_NIVEL,
        alt=salidas.alt(EJ, "dae_error_vs_noise.png"),
    )
    d.imagen(
        salidas.png(EJ, "dae_within_vs_noise.png"), x_der, y_fig, ancho, ALTO_FIG_NIVEL,
        alt=salidas.alt(EJ, "dae_within_vs_noise.png"),
    )

    ultimo = dat.niveles[-1]
    nivel_cambio = dat.niveles[-4]
    d.texto(
        x_izq, y_cap, ancho, ALTO_CAPTION_NIVEL,
        f"Píxeles incorrectos medios por nivel: la curva de la salida casi coincide con el eje "
        f"horizontal hasta {dat.nivel_max} mientras la de la entrada crece en línea recta; la "
        f"salida cambia de pendiente cerca de {nivel_cambio}.",
        tam=TAM_PIE,
    )
    mitad = dat.primer_nivel_bajo(dat.frac_entrada, MITAD)
    d.texto(
        x_der, y_cap, ancho, ALTO_CAPTION_NIVEL,
        f"Muestras con a lo sumo {dat.tol} {dat.px_tol} incorrecto: sin autoencoder, la entrada "
        f"cae por debajo de la mitad ya en {mitad}; la salida sigue sobre el {dat.umbral} % hasta "
        f"{dat.nivel_max} y después baja de forma continua hasta {dat.fila_nivel(ultimo)[3]} % en {ultimo}.",
        tam=TAM_PIE,
    )
    y_eval = y_cap + ALTO_CAPTION_NIVEL + HUECO_CHICO
    d.texto(
        MARGEN_IN, y_eval, ANCHO_UTIL, Y_FIN - y_eval,
        f"Evaluación: {dat.n_dist} distorsiones no vistas de cada uno de los {dat.patrones} "
        f"patrones en cada nivel ({dat.muestras} muestras por nivel), con una semilla distinta de "
        "la del entrenamiento y de la selección; se binariza la salida y se cuentan los píxeles "
        "que difieren del patrón limpio.",
        tam=TAM_PIE,
    )


def _tabla_y_titular(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Resultados: tabla por nivel y nivel máximo eliminado")
    y = CONTENIDO_Y_IN
    d.codigo(
        MARGEN_IN, y, ANCHO_BLOQUE_TABLA, ALTO_BLOQUE_TABLA, dat.tabla,
        fuente=FUENTE, tam=TAM_BLOQUE_TABLA,
    )
    x_cifra = MARGEN_IN + ANCHO_BLOQUE_TABLA + 0.3
    d.cifra(
        x_cifra, y, ANCHO_IN - MARGEN_IN - x_cifra, ALTO_BLOQUE_TABLA,
        dat.nivel_max, "nivel máximo eliminado",
    )
    y += ALTO_BLOQUE_TABLA + HUECO
    d.codigo(
        MARGEN_IN, y, ANCHO_UTIL, ALTO_BLOQUE_TITULAR, [dat.titular], fuente=FUENTE, tam=TAM_CODIGO
    )
    y += ALTO_BLOQUE_TITULAR + HUECO

    limite = float(dat.umbral) / CIEN
    primero_bajo = dat.primer_nivel_bajo(dat.frac, limite)
    d.texto(
        MARGEN_IN, y, ANCHO_UTIL, Y_FIN - y,
        [
            Parrafo(
                f"El nivel máximo eliminado es una convención: el mayor nivel en el que, a ese "
                f"nivel y a todos los menores, al menos el {dat.umbral} % de las distorsiones no "
                f"vistas queda con a lo sumo {dat.tol} {dat.px_tol} incorrecto.",
                vineta=True,
            ),
            Parrafo(
                f"Con otro umbral el nivel cambia: en {primero_bajo} queda el "
                f"{dat.fila_nivel(primero_bajo)[3]} %, apenas por debajo de {dat.umbral} %. La "
                "información que no depende de la convención es el perfil de la tabla completa.",
                vineta=True,
            ),
        ],
        tam=TAM_TEXTO,
    )


def _ejemplos(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Resultados: original, entrada ruidosa y salida")
    titular = dat.nivel_de(dat.nivel_max)
    siguiente = dat.niveles[dat.niveles.index(titular) + 1]
    mitad = dat.primer_nivel_bajo(dat.frac, MITAD)
    ultimo = dat.niveles[-1]
    # (nivel elegido por regla, lectura confirmada sobre la imagen abierta)
    elegidos = [
        (
            titular,
            "las ocho salidas son exactas aunque cada entrada llega con píxeles invertidos.",
        ),
        (
            siguiente,
            "último nivel de los ejemplos en que las ocho letras vuelven exactas.",
        ),
        (
            mitad,
            "fallan varias letras (a, b, e, s y z); g, k y w quedan dentro de la tolerancia.",
        ),
        (
            ultimo,
            "salidas con forma de letra pero sin relación con la de partida: la entrada no "
            "tiene información.",
        ),
    ]
    vistos: list[str] = []
    ejemplos = []
    for nivel, lectura in elegidos:
        if nivel in vistos:
            continue
        vistos.append(nivel)
        ejemplos.append((nivel, lectura))
    if len(ejemplos) < 2:
        raise ValueError(f"{EJ}/dae_levels.csv: menos de dos niveles distintos para los ejemplos")

    ancho = (ANCHO_UTIL - HUECO) / 2
    alto_celda = ALTO_IMG_EJEMPLO + ALTO_CAPTION_EJEMPLO + HUECO_CHICO
    for i, (nivel, lectura) in enumerate(ejemplos):
        fila, col = divmod(i, 2)
        x = MARGEN_IN + col * (ancho + HUECO)
        y = CONTENIDO_Y_IN + fila * alto_celda
        nombre = f"dae_examples_p{nivel}.png"
        d.imagen(
            salidas.png(EJ, nombre), x, y, ancho, ALTO_IMG_EJEMPLO,
            alt=salidas.alt(EJ, nombre),
        )
        d.texto(
            x, y + ALTO_IMG_EJEMPLO, ancho, ALTO_CAPTION_EJEMPLO,
            f"Nivel {nivel}: {lectura}", tam=TAM_PIE,
        )


def _conclusiones(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Conclusiones")
    n1, n2 = dat.niveles[1], dat.niveles[2]
    f1, f2 = dat.fila_nivel(n1), dat.fila_nivel(n2)
    ultimo = dat.niveles[-1]
    f_ultimo = dat.fila_nivel(ultimo)

    limite = float(dat.umbral) / CIEN
    falla = dat.primer_nivel_bajo(dat.frac, limite)
    i_falla = dat.niveles.index(falla)
    siguientes = dat.niveles[i_falla + 1 : i_falla + 3]
    f_falla = dat.fila_nivel(falla)
    cola = _enumerar([f"{dat.fila_nivel(n)[3]} % en {n}" for n in siguientes])

    p_lat = [dat.puntaje(f) for f in dat.lat]
    p_hid = [dat.puntaje(f) for f in dat.hid]
    separa_latente = (max(p_lat) - min(p_lat)) >= (max(p_hid) - min(p_hid))
    mejor_lat = min(dat.lat, key=dat.puntaje)
    peor_lat = max(dat.lat, key=dat.puntaje)
    mejor_ruido = min(dat.noise, key=dat.puntaje)
    peor_ruido = max(dat.noise, key=dat.puntaje)
    if separa_latente:
        frase_factor = (
            f"El tamaño del latente fue el factor que más separó las arquitecturas (puntaje "
            f"{peor_lat[2]} con {peor_lat[0]} contra {mejor_lat[2]} con {mejor_lat[0]})"
        )
    else:
        frase_factor = "La capacidad del encoder fue el factor que más separó las arquitecturas"
    semillas = _k_de_n(dat.noise[0][1])[1]

    puntos = [
        f"En los niveles bajos elimina casi todo el ruido: de {f1[0]} píxeles incorrectos en la "
        f"entrada a {f1[1]} en la salida en {n1}, y de {f2[0]} a {f2[1]} en {n2}.",
        f"Empieza a fallar en {falla}: el {f_falla[3]} % de las salidas queda con a lo sumo "
        f"{dat.tol} {dat.px_tol} incorrecto, por debajo de la convención de {dat.umbral} %, y "
        f"después baja a {cola}.",
        f"En el nivel más alto, {ultimo}, la salida queda cerca de la entrada ({f_ultimo[1]} "
        f"contra {f_ultimo[0]} píxeles incorrectos) y ningún método puede recuperar el patrón: el "
        f"{f_ultimo[3]} % que queda dentro de la tolerancia es azar, no eliminación de ruido.",
        f"{frase_factor}, y el calendario de ruido de entrenamiento decidió la mejor variante: "
        f"{mejor_ruido[0]} tiene el menor puntaje del estudio ({mejor_ruido[2]}, contra "
        f"{peor_ruido[2]} de {peor_ruido[0]}).",
        f"Límites: solo ruido de inversión de píxeles; la red es un modelo de los {dat.patrones} "
        f"patrones y, si el ruido la engaña, devuelve otro de ellos; la red elegida es una entre "
        f"{semillas} semillas por una regla escrita como código; el umbral de {dat.umbral} % es "
        "una convención de este trabajo.",
    ]
    d.texto(
        MARGEN_IN, CONTENIDO_Y_IN, ANCHO_TEXTO_CONCL, Y_FIN - CONTENIDO_Y_IN,
        [Parrafo(t, vineta=True) for t in puntos], tam=TAM_TEXTO,
    )
    x = MARGEN_IN + ANCHO_TEXTO_CONCL + 0.3
    ancho = ANCHO_IN - MARGEN_IN - x
    alto = (Y_FIN - CONTENIDO_Y_IN - HUECO) / 2
    d.cifra(x, CONTENIDO_Y_IN, ancho, alto, dat.nivel_max, "nivel máximo eliminado")
    d.cifra(
        x, CONTENIDO_Y_IN + alto + HUECO, ancho, alto,
        f"{f_ultimo[3]} %", f"dentro de la tolerancia en {ultimo}: azar",
    )


def agregar(pres, salidas) -> None:
    dat = _Datos(salidas)
    _arquitectura(pres, dat)
    _tecnicas_latente(pres, salidas, dat)
    _tecnicas_ruido(pres, salidas, dat)
    _distorsion(pres, salidas, dat)
    _error_por_nivel(pres, salidas, dat)
    _tabla_y_titular(pres, dat)
    _ejemplos(pres, salidas, dat)
    _conclusiones(pres, dat)
