"""Seccion 10: Ejercicio 1a (autoencoder basico con latente de dos dimensiones).

Ninguna cifra se escribe a mano: todas salen de los reportes, tablas y CSV que
regenera scripts/regenerar.sh, leidos con lector.Salidas. Las lineas que se
muestran literalmente son lineas exactas de ej1a/salida-ej1a.txt. Las frases
sobre las figuras parafrasean docs/ej1a-analisis.md y se contrastaron con la
imagen abierta.
"""

from __future__ import annotations

import re

from lector import campo, numero
from pptx_minimo import ALTO_IN, ANCHO_IN, COLOR_AMBAR, CONTENIDO_Y_IN, MARGEN_IN, Parrafo

EJ = "ej1a"
SALIDA = "salida-ej1a.txt"
FUENTE = f"{EJ}/{SALIDA}"

ANCHO_UTIL = ANCHO_IN - 2 * MARGEN_IN
Y_FIN = ALTO_IN - MARGEN_IN
TAM_CUERPO = 16
TAM_PIE = 14
TAM_CODIGO = 11
TAM_TABLA = 11

# Encabezados cortos de las cuatro tablas de comparison.md: no llevan cifras.
ENCABEZADO = ["variante", "semillas ok", "mejor máx. px", "mediana épocas", "mediana pérdida"]
PESOS_TABLA = [2.3, 0.9, 1.0, 1.1, 1.1]


# --------------------------------------------------------------------------
# Utilidades de lectura
# --------------------------------------------------------------------------


def _px(n: str) -> str:
    return "píxel" if n == "1" else "píxeles"


def _enumerar(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


def _fila(filas: list[list[str]], prefijo: str) -> list[str]:
    for f in filas:
        if f[0].startswith(prefijo):
            return f
    raise ValueError(f"{EJ}/comparison.md: ninguna variante empieza con {prefijo!r}")


def _k_de_n(celda: str) -> tuple[int, int]:
    return int(numero(celda, r"^(\d+)/")), int(numero(celda, r"/(\d+)$"))


def _rango(filas: list[dict[str, str]], etiquetas: list[str], eje: str) -> tuple[str, str]:
    valores = [float(f[eje]) for f in filas if f["label"] in etiquetas]
    if len(valores) != len(etiquetas):
        raise ValueError(f"{EJ}/latent_codes.csv: faltan letras de {etiquetas}")
    return f"{min(valores):.1f}", f"{max(valores):.1f}"


def _estudio(salidas, nombre: str) -> list[list[str]]:
    encabezado, filas = salidas.tabla_md(EJ, "comparison.md", nombre)
    if len(encabezado) != len(ENCABEZADO):
        raise ValueError(
            f"{EJ}/comparison.md: la tabla {nombre!r} tiene {len(encabezado)} columnas "
            f"y la diapositiva espera {len(ENCABEZADO)}"
        )
    return filas


class _Datos:
    """Todo lo que las diapositivas toman de los reportes y las tablas."""

    def __init__(self, salidas) -> None:
        self.cfg = salidas.linea(EJ, SALIDA, "configuracion elegida:")
        self.resultado = salidas.linea(EJ, SALIDA, "RESULTADO:")
        self.ok = numero(self.resultado, r"RESULTADO: (\d+)/\d+ patrones")
        self.total = numero(self.resultado, r"RESULTADO: \d+/(\d+) patrones")
        self.tol = numero(self.resultado, r"a lo sumo (\d+) pixel")
        self.maximo = numero(self.resultado, r"\(maximo (\d+)\)")
        self.nombre = campo(self.cfg, "configuracion elegida")
        self.semilla = campo(self.cfg, "semilla")
        self.variante = re.sub(r"-s\d+$", "", self.nombre)
        self.tamanios = [int(n) for n in campo(self.cfg, "arquitectura").split("-")]
        self.arch = _estudio(salidas, "arch")
        self.optim = _estudio(salidas, "optim")
        self.act = _estudio(salidas, "act")
        self.loss = _estudio(salidas, "loss")
        self.fila_elegida = _fila(self.arch, self.variante)
        # Grilla de la letra nueva: el punto latente y las 7 lineas siguientes.
        self.bloque_punto = salidas.bloque(EJ, SALIDA, "punto latente:", 8)
        self.grilla = self.bloque_punto[1:]
        self.cercana = salidas.linea(EJ, SALIDA, "letra mas cercana del entrenamiento:")
        self.letra_nueva = salidas.linea(EJ, SALIDA, "LETRA NUEVA:")
        self.corridas = numero(salidas.linea(EJ, "salida-sweep.txt", "sweep:"), r"sweep: (\d+) corridas")
        self.px_tol = _px(self.tol)


# --------------------------------------------------------------------------
# Diapositivas
# --------------------------------------------------------------------------


def _arquitectura(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Arquitectura del autoencoder")
    tam = dat.tamanios
    i_lat = tam.index(min(tam))
    latente = tam[i_lat]
    codificador = "-".join(str(n) for n in tam[1:i_lat])
    decodificador = "-".join(str(n) for n in tam[i_lat + 1 : -1])
    dim_letra = f"{len(dat.grilla)}x{len(dat.grilla[0])}"

    ancho_caja, alto_caja = 2.0, 1.3
    hueco = (ANCHO_UTIL - 5 * ancho_caja) / 4
    y_diag = CONTENIDO_Y_IN + 0.1
    cajas = [
        ("Entrada", f"{tam[0]} píxeles ({dim_letra})", None),
        ("Encoder", f"capas {codificador}", None),
        ("Latente", f"{latente} neuronas", COLOR_AMBAR),
        ("Decoder", f"capas {decodificador}", None),
        ("Salida", f"{tam[-1]} píxeles", None),
    ]
    for i, (titulo, detalle, relleno) in enumerate(cajas):
        x = MARGEN_IN + i * (ancho_caja + hueco)
        d.caja(
            x, y_diag, ancho_caja, alto_caja,
            [Parrafo(titulo, negrita=True), Parrafo(detalle)],
            tam=TAM_CUERPO, relleno=relleno,
        )
        if i < len(cajas) - 1:
            d.texto(x + ancho_caja, y_diag + 0.35, hueco, 0.6, "→", tam=24)

    perdida = campo(dat.cfg, "perdida")
    nombre_perdida = {"BCE": "entropía cruzada binaria", "MSE": "error cuadrático medio"}.get(
        perdida, perdida
    )
    viñetas = [
        f"La capa latente tiene {latente} neuronas por restricción de la consigna: la red debe "
        f"comprimir los {tam[0]} píxeles de cada letra de {dim_letra} en {latente} coordenadas.",
        "Activación tanh en las capas ocultas, identidad en la capa latente y sigmoide en la salida.",
        f"Pérdida {perdida} ({nombre_perdida}), optimizador {campo(dat.cfg, 'optimizador')}, "
        f"{campo(dat.cfg, 'epocas')} épocas, semilla {dat.semilla}.",
        "Backpropagation propia sobre ndarray, sin ningún framework de aprendizaje automático.",
    ]
    y_vin = y_diag + alto_caja + 0.25
    y_cfg, alto_cfg = 6.3, 0.65
    d.texto(
        MARGEN_IN, y_vin, ANCHO_UTIL, y_cfg - 0.1 - y_vin,
        [Parrafo(t, vineta=True) for t in viñetas], tam=TAM_CUERPO,
    )
    d.codigo(MARGEN_IN, y_cfg, ANCHO_UTIL, alto_cfg, [dat.cfg], fuente=FUENTE, tam=TAM_CODIGO)


def _tecnicas_tablas(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Técnicas probadas: arquitecturas, optimizadores, activaciones y pérdida")
    columna = 6.0
    x_izq = MARGEN_IN
    x_der = ANCHO_IN - MARGEN_IN - columna
    hueco_tablas = 0.2

    def poner(x: float, y: float, filas: list[list[str]]) -> float:
        alto = d.tabla(
            x, y, columna, [ENCABEZADO] + filas,
            anchos=PESOS_TABLA, tam=TAM_TABLA,
        )
        return y + alto + hueco_tablas

    y = CONTENIDO_Y_IN
    y = poner(x_izq, y, dat.arch)
    y_izq_fin = poner(x_izq, y, dat.optim)
    y = CONTENIDO_Y_IN
    y = poner(x_der, y, dat.act)
    y_der_fin = poner(x_der, y, dat.loss)

    leyenda = [
        f"semillas ok: semillas con todos los patrones dentro de la tolerancia de {dat.tol} {dat.px_tol}",
        "mejor máx. px: mejor máximo de píxeles incorrectos",
        "mediana épocas: hasta entrar en la tolerancia",
        "mediana pérdida: pérdida final",
    ]
    y_pie = y_izq_fin + 0.05
    d.texto(
        x_der, y_der_fin + 0.05, columna, y_pie - 0.1 - (y_der_fin + 0.05),
        [Parrafo(t, vineta=True) for t in leyenda], tam=TAM_PIE,
    )
    variantes = len(dat.arch) + len(dat.optim) + len(dat.act) + len(dat.loss)
    semillas = _k_de_n(dat.fila_elegida[1])[1]
    d.texto(
        MARGEN_IN, y_pie, ANCHO_UTIL, Y_FIN - y_pie,
        f"{dat.corridas} corridas en total: {variantes} variantes con {semillas} semillas cada una, "
        "y un solo factor distinto respecto de una base en cada estudio.",
        tam=TAM_PIE,
    )


def _tecnicas_curvas(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Técnicas probadas: curvas del barrido")
    ancho = 6.0
    alto = 3.6
    x_izq = MARGEN_IN
    x_der = ANCHO_IN - MARGEN_IN - ancho
    y_fig = CONTENIDO_Y_IN
    y_pie = y_fig + alto + 0.15
    alto_pie = Y_FIN - y_pie

    d.imagen(
        salidas.png(EJ, "curves_optim_loss.png"), x_izq, y_fig, ancho, alto,
        alt=salidas.alt(EJ, "curves_optim_loss.png"),
    )
    d.texto(
        x_izq, y_pie, ancho, alto_pie,
        "Pérdida de entrenamiento por optimizador, en escala logarítmica y con la semilla 0: "
        "SGD y Momentum con tasa alta quedan casi planos en torno a la misma pérdida; "
        "las curvas de Adam descienden, con algún salto abrupto del que se recuperan.",
        tam=TAM_PIE,
    )

    n = [f[0] for f in dat.arch]
    d.imagen(
        salidas.png(EJ, "curves_arch_wrongpx.png"), x_der, y_fig, ancho, alto,
        alt=salidas.alt(EJ, "curves_arch_wrongpx.png"),
    )
    d.texto(
        x_der, y_pie, ancho, alto_pie,
        f"Máximo de píxeles incorrectos por época, con la semilla 0: {n[0]}, {n[1]} y {n[2]} "
        f"muestran saltos abruptos y después se recuperan; {n[3]} y {n[4]} llegan antes a cero "
        "y sin saltos.",
        tam=TAM_PIE,
    )


def _resultado(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva(
        f"Ej 1a: Resultados: {dat.ok} patrones con a lo sumo {dat.tol} {dat.px_tol} incorrecto"
    )
    y = CONTENIDO_Y_IN
    ancho_cifra, alto_cifra = 3.0, 1.8
    d.cifra(MARGEN_IN, y, ancho_cifra, alto_cifra, f"{dat.ok}/{dat.total}", "patrones dentro de la tolerancia")
    d.cifra(
        MARGEN_IN + ancho_cifra + 0.3, y, ancho_cifra, alto_cifra,
        dat.maximo, "máximo de píxeles incorrectos en un patrón",
    )
    x_texto = MARGEN_IN + 2 * ancho_cifra + 0.6 + 0.3
    median = dat.fila_elegida[3]
    semillas = dat.fila_elegida[1]
    d.texto(
        x_texto, y, ANCHO_IN - MARGEN_IN - x_texto, alto_cifra,
        f"La configuración se eligió entre las variantes con todas las semillas dentro de la "
        f"tolerancia ({semillas} en {dat.variante}): la de menor mediana de épocas hasta la "
        f"tolerancia, {median}; dentro de ella, la semilla {dat.semilla}, la de menor pérdida final.",
        tam=15,
    )

    y_bloque = y + alto_cifra + 0.25
    alto_linea = 0.45
    d.codigo(MARGEN_IN, y_bloque, ANCHO_UTIL, alto_linea, [dat.resultado], fuente=FUENTE, tam=12)
    subconjunto = salidas.linea(EJ, SALIDA, "SUBCONJUNTO:")
    d.codigo(MARGEN_IN, y_bloque + alto_linea + 0.1, ANCHO_UTIL, alto_linea, [subconjunto], fuente=FUENTE, tam=12)

    # Un renglon por patron del reporte: "caracter: pixeles incorrectos".
    patrones = []
    for linea in salidas.lineas(EJ, SALIDA):
        m = re.match(r"^\s*\d+\s+(\S+)\s+(\d+)\s+(\w+)$", linea)
        if m:
            patrones.append(f"{m.group(1)}: {m.group(2)}")
    if len(patrones) != int(dat.total):
        raise ValueError(
            f"{EJ}/{SALIDA}: la tabla por patron tiene {len(patrones)} filas y RESULTADO dice {dat.total}"
        )
    columnas = 8
    filas = [patrones[i : i + columnas] for i in range(0, len(patrones), columnas)]
    y_pie = y_bloque + 2 * alto_linea + 0.35
    d.texto(
        MARGEN_IN, y_pie, ANCHO_UTIL, 0.4,
        "Píxeles incorrectos por patrón (carácter: píxeles incorrectos), del reporte del ejercicio",
        tam=TAM_PIE,
    )
    d.tabla(MARGEN_IN, y_pie + 0.45, ANCHO_UTIL, filas, tam=12, encabezado=False)


def _latente(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Resultados: espacio latente")
    codigos = salidas.csv(EJ, "latent_codes.csv")
    lado = Y_FIN - CONTENIDO_Y_IN
    d.imagen(
        salidas.png(EJ, "latent.png"), MARGEN_IN, CONTENIDO_Y_IN, lado, lado,
        alt=salidas.alt(EJ, "latent.png"),
    )
    izquierda = ["b", "h", "k", "p", "r", "n"]
    denso = ["t", "x", "z", "g", "y", "s", "j", "{", "}"]
    a, b = _rango(codigos, izquierda, "z1")
    c, e = _rango(codigos, denso, "z1")
    f, g = _rango(codigos, denso, "z2")
    x = MARGEN_IN + lado + 0.4
    ancho = ANCHO_IN - MARGEN_IN - x
    d.texto(
        x, CONTENIDO_Y_IN, ancho, 3.2,
        [
            Parrafo(
                f"Las letras {_enumerar(izquierda)} quedan juntas en el borde izquierdo, con z1 "
                f"entre {a} y {b}: comparten el trazo vertical del lado izquierdo y el encoder "
                "parece agrupar por esa característica.",
                vineta=True,
            ),
            Parrafo(
                f"Un grupo denso rodea el origen ({_enumerar(denso)}), con z1 entre {c} y {e} y z2 "
                f"entre {f} y {g}: es la zona con menor separación entre códigos, y ahí cae la letra nueva.",
                vineta=True,
            ),
        ],
        tam=TAM_CUERPO,
    )
    distintos = len({(r["z1"], r["z2"]) for r in codigos})
    d.cifra(
        x, CONTENIDO_Y_IN + 3.45, 3.6, 1.9,
        f"{distintos}/{len(codigos)}", "códigos latentes distintos entre sí",
    )


def _letra_nueva(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Resultados: letra nueva")
    ancho_img = 5.6
    alto_img = 4.3
    d.imagen(
        salidas.png(EJ, "new_letter.png"), MARGEN_IN, CONTENIDO_Y_IN, ancho_img, alto_img,
        alt=salidas.alt(EJ, "new_letter.png"),
    )
    cercana = numero(dat.cercana, r"entrenamiento: (\S+) \(")
    distancia = numero(dat.cercana, r"distancia Hamming (\d+)\)")
    minimo = numero(dat.letra_nueva, r"en al menos (\d+) pixeles")
    d.texto(
        MARGEN_IN, CONTENIDO_Y_IN + alto_img + 0.15, ancho_img, 0.9,
        f"Izquierda, la letra nueva; derecha, la {cercana} más cercana del entrenamiento "
        f"(distancia de Hamming {distancia}).",
        tam=TAM_PIE,
    )
    x = MARGEN_IN + ancho_img + 0.35
    ancho = ANCHO_IN - MARGEN_IN - x
    lineas = dat.bloque_punto + [dat.cercana, dat.letra_nueva]
    alto_codigo = 2.35
    d.codigo(x, CONTENIDO_Y_IN, ancho, alto_codigo, lineas, fuente=FUENTE, tam=TAM_CODIGO)
    y = CONTENIDO_Y_IN + alto_codigo + 0.2
    d.texto(
        x, y, ancho, Y_FIN - y,
        [
            Parrafo(
                "Se parece a una z deformada: están la barra superior y la inferior, pero la "
                "diagonal queda recargada de píxeles encendidos y la barra inferior pierde la "
                "esquina derecha.",
                vineta=True,
            ),
            Parrafo("Es una forma nueva, pero no una letra legible.", vineta=True),
            Parrafo(
                "Sale de un punto de una zona densa del plano latente que no es el código de "
                f"ningún patrón, y difiere de todos en al menos {minimo} {_px(minimo)}.",
                vineta=True,
            ),
        ],
        tam=TAM_CUERPO,
    )


def _conclusiones(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Conclusiones")
    tam = dat.tamanios
    latente = min(tam)

    sgd = _fila(dat.optim, "optim-sgd")
    sgd_todas = [f for f in dat.optim if f[0].startswith("optim-sgd")]
    mom = [f for f in dat.optim if f[0].startswith("optim-mom")]
    adam = [f for f in dat.optim if f[0].startswith("optim-adam")]
    if len(mom) < 2 or not adam:
        raise ValueError(f"{EJ}/comparison.md: el estudio optim no tiene las filas esperadas")
    sgd_txt = _enumerar([f[1] for f in sgd_todas])
    adam_txt = _enumerar([f[1] for f in adam])

    fallidas = [f for f in dat.arch if _k_de_n(f[1])[0] != _k_de_n(f[1])[1]]
    if len(fallidas) == 1:
        texto_falla = f"{fallidas[0][0]} es la única arquitectura con una semilla fallida ({fallidas[0][1]})"
    elif fallidas:
        texto_falla = "con semillas fallidas: " + _enumerar([f"{f[0]} ({f[1]})" for f in fallidas])
    else:
        texto_falla = "ninguna arquitectura tiene semillas fallidas"
    medianas = [int(f[3]) for f in dat.arch if f[3].isdigit()]
    el = dat.fila_elegida
    menor = "la menor mediana" if int(el[3]) == min(medianas) else "una mediana"

    bce = _fila(dat.loss, "loss-bce")
    mse = _fila(dat.loss, "loss-mse")
    semillas = _k_de_n(el[1])[1]

    puntos = [
        f"Con un espacio latente de {latente} dimensiones, {dat.ok}/{dat.total} patrones quedan con "
        f"a lo sumo {dat.tol} {dat.px_tol} incorrecto (máximo {dat.maximo}).",
        f"El optimizador es el factor que más separa los resultados (semillas que aprenden todos los "
        f"patrones): SGD con las dos tasas, {sgd_txt}; Momentum con la tasa alta, {mom[0][1]}, y con "
        f"la baja, {mom[1][1]}; Adam, {adam_txt} según la tasa.",
        f"La capacidad ayuda a la robustez y a la velocidad: {texto_falla}, y {dat.variante} "
        f"({el[1]}) tiene {menor} de épocas hasta la tolerancia ({el[3]}).",
        f"Entropía cruzada binaria contra error cuadrático medio, con todo lo demás igual: "
        f"{bce[0]} aprende {bce[1]} y {mse[0]} aprende {mse[1]}; el mejor caso con MSE termina con "
        f"{mse[2]} {_px(mse[2])} incorrectos y el criterio admite {dat.tol}.",
        f"Límites: el criterio es reconstruir los mismos patrones del entrenamiento y la selección "
        f"se hizo sobre ellos; la red elegida es una semilla entre las {semillas} de su variante; "
        "la letra nueva se parece a una z deformada y no es una letra legible.",
    ]
    ancho_texto = 8.4
    d.texto(
        MARGEN_IN, CONTENIDO_Y_IN, ancho_texto, Y_FIN - CONTENIDO_Y_IN,
        [Parrafo(t, vineta=True) for t in puntos], tam=15,
    )

    x = MARGEN_IN + ancho_texto + 0.3
    ancho = ANCHO_IN - MARGEN_IN - x
    alto = 1.65
    hueco = (Y_FIN - CONTENIDO_Y_IN - 3 * alto) / 2
    cifras = [
        (f"{dat.ok}/{dat.total}", "patrones dentro de la tolerancia"),
        (sgd[1], f"semillas que aprenden con {sgd[0]}"),
        (mse[1], f"semillas que aprenden con {mse[0]}"),
    ]
    for i, (valor, etiqueta) in enumerate(cifras):
        d.cifra(x, CONTENIDO_Y_IN + i * (alto + hueco), ancho, alto, valor, etiqueta)


def agregar(pres, salidas) -> None:
    dat = _Datos(salidas)
    _arquitectura(pres, dat)
    _resultado(pres, salidas, dat)
    _latente(pres, salidas, dat)
