"""Seccion 20: Ejercicio 1b (denoising autoencoder).

Ocho diapositivas: la arquitectura y su eleccion (1.b.1), el barrido que la respalda y el
estudio de la capacidad de eliminar ruido por nivel (1.b.2). En la diapositiva queda solo lo
comprobable contra output/ o contra la figura; la explicacion va a las notas del orador.

Ninguna cifra se escribe a mano, ni en la diapositiva ni en las notas: todas salen de los
reportes, tablas y CSV que regenera scripts/regenerar.sh, leidos con lector.Salidas.
"""

from __future__ import annotations

import re

from disposicion import (
    ANCHO_UTIL,
    TAM_TABLA,
    destacado,
    diagrama,
    enumerar,
    figura_alta_centrada,
    figura_ancha,
    lineas_md,
    px,
    texto_justo,
    vinetas,
)
from lector import campo, numero
from pptx_minimo import COLOR_AMBAR, CONTENIDO_Y_IN, MARGEN_IN

EJ = "ej1b"
SALIDA = "salida-ej1b.txt"
FUENTE = f"{EJ}/{SALIDA}"
TABLA = "dae_comparison.md"
FUENTE_TABLA = f"{EJ}/{TABLA}"
BARRIDO = "salida-dae-sweep.txt"
FUENTE_BARRIDO = f"{EJ}/{BARRIDO}"

# Columnas de dae_comparison.md que se muestran, en su orden: variante, puntaje y px por nivel.
COLUMNAS = [0, 2, 3, 4, 5]
ANCHOS = [1.4, 1.2, 1.1, 1.1, 1.1]


def _k_de_n(celda: str) -> tuple[int, int]:
    return int(numero(celda, r"^(\d+)/")), int(numero(celda, r"/(\d+)$"))


class _Datos:
    """Todo lo que las diapositivas toman de los reportes y las tablas."""

    def __init__(self, salidas) -> None:
        self.cfg = salidas.linea(EJ, SALIDA, "configuracion:")
        self.evaluacion = salidas.linea(EJ, SALIDA, "evaluacion:")
        self.titular = salidas.linea(EJ, SALIDA, "NIVEL MAXIMO ELIMINADO:")
        self.nombre = campo(self.cfg, "configuracion")
        self.variante = re.sub(r"-s\d+$", "", self.nombre)
        self.semilla = campo(self.cfg, "semilla")
        self.epocas = campo(self.cfg, "epocas")
        self.copias = campo(self.cfg, "copias por epoca")
        self.optimizador = campo(self.cfg, "optimizador")
        self.perdida = campo(self.cfg, "perdida")
        self.ruido = campo(self.cfg, "ruido de entrenamiento")
        self.tamanios = [int(n) for n in campo(self.cfg, "arquitectura").split("-")]
        self.latente = min(self.tamanios)
        self.nivel_max = numero(self.titular, r"ELIMINADO: (\S+) \(")
        self.umbral = numero(self.titular, r"al menos el (\d+) %")
        self.tol = numero(self.titular, r"a lo sumo (\d+) pixel")
        self.px_ok = f"≤ {self.tol} {px(self.tol)} incorrecto"
        self.por_patron = numero(self.evaluacion, r"evaluacion: (\d+) distorsiones")

        # Tabla por nivel: la linea de encabezado y una linea por nivel, literales.
        todas = salidas.lineas(EJ, SALIDA)
        i = next((k for k, l in enumerate(todas) if l.startswith("nivel ")), None)
        if i is None:
            raise ValueError(f"{FUENTE}: no esta la tabla por nivel")
        self.encabezado_niveles = todas[i]
        self.lineas_nivel: dict[str, str] = {}
        for l in todas[i + 1 :]:
            celdas = l.split()
            if len(celdas) != 5 or not re.fullmatch(r"\d\.\d\d", celdas[0]):
                break
            self.lineas_nivel[celdas[0]] = l
        self.niveles = list(self.lineas_nivel)
        if len(self.niveles) < 3:
            raise ValueError(f"{FUENTE}: la tabla por nivel tiene {len(self.niveles)} filas")
        self.csv = {r["level"]: r for r in salidas.csv(EJ, "dae_levels.csv")}
        if list(self.csv) != self.niveles:
            raise ValueError(f"{EJ}/dae_levels.csv: los niveles no coinciden con la tabla del reporte")

        self.encabezados: dict[str, list[str]] = {}
        self.estudios: dict[str, list[list[str]]] = {}
        for estudio in ("lat", "hid", "noise"):
            self.encabezados[estudio], self.estudios[estudio] = salidas.tabla_md(EJ, TABLA, estudio)
        self.seleccion = salidas.linea(EJ, BARRIDO, "SELECCION:")
        self.linea_barrido = salidas.linea(EJ, BARRIDO, "dae-sweep:")
        self.corridas = numero(self.linea_barrido, r"dae-sweep: (\d+) corridas")
        self.puntajes: dict[str, list[str]] = {}
        for l in salidas.lineas(EJ, BARRIDO):
            m = re.match(r"^(\S+)-s\d+ \| puntaje (\d+\.\d+) \|", l)
            if m:
                self.puntajes.setdefault(m.group(1), []).append(m.group(2))
        columnas = list(salidas.csv(EJ, "dae_sweep.csv")[0])
        self.niveles_puntaje = sorted(
            c[len("wrong_p") :] for c in columnas if c.startswith("wrong_p") and float(c[len("wrong_p") :]) > 0
        )

    def celda(self, nivel: str, columna: int) -> str:
        """Celda de la tabla por nivel: 1 entrada_px, 2 salida_px, 3 max_px, 4 dentro_1px_%."""
        return self.lineas_nivel[nivel].split()[columna]

    def fila(self, variante: str) -> list[str]:
        for filas in self.estudios.values():
            for f in filas:
                if f[0] == variante:
                    return f
        raise ValueError(f"{FUENTE_TABLA}: no esta la variante {variante!r}")

    def tabla(self, estudio: str) -> list[list[str]]:
        enc = self.encabezados[estudio]
        return [[enc[i] for i in COLUMNAS]] + [[f[i] for i in COLUMNAS] for f in self.estudios[estudio]]

    def extremos(self, estudio: str) -> tuple[list[str], list[str]]:
        filas = sorted(self.estudios[estudio], key=lambda f: float(f[2]))
        return filas[0], filas[-1]

    def base(self) -> list[str]:
        """Nombres con los que la configuracion base aparece en los estudios (misma fila)."""
        vistas: dict[tuple, list[str]] = {}
        for filas in self.estudios.values():
            for f in filas:
                vistas.setdefault(tuple(f[1:]), []).append(f[0])
        return max(vistas.values(), key=len)

    def ruido_en_palabras(self) -> str:
        m = re.fullmatch(r"fix(\d+\.\d+)", self.ruido)
        if m:
            return f"fijo en {m.group(1)}"
        m = re.fullmatch(r"u(\d+\.\d+)-(\d+\.\d+)", self.ruido)
        if m:
            return f"sorteado entre {m.group(1)} y {m.group(2)}"
        raise ValueError(f"{FUENTE}: ruido de entrenamiento desconocido {self.ruido!r}")

    def primer_nivel(self, columna: str, limite: float) -> str | None:
        """Primer nivel cuya fraccion `columna` del CSV queda por debajo de `limite`."""
        return next((n for n in self.niveles if float(self.csv[n][columna]) < limite), None)

    def cita_tabla(self, niveles: list[str] | None = None) -> list[str]:
        return [self.encabezado_niveles] + [self.lineas_nivel[n] for n in (niveles or self.niveles)]


# --------------------------------------------------------------------------
# Diapositivas
# --------------------------------------------------------------------------


def _arquitectura(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Arquitectura")
    tam = dat.tamanios
    i_lat = tam.index(dat.latente)
    codificador = "-".join(str(n) for n in tam[1:i_lat])
    decodificador = "-".join(str(n) for n in tam[i_lat + 1 : -1])
    y = diagrama(
        d,
        [
            ("Entrada", "con ruido", None),
            ("Encoder", codificador, None),
            ("Latente", f"{dat.latente} neuronas", COLOR_AMBAR),
            ("Decoder", decodificador, None),
            ("Salida", f"{tam[-1]} píxeles", None),
        ],
    )
    # La eleccion del latente se afirma solo si los datos del barrido la sostienen.
    propia = f"lat-l{dat.latente}"
    mayores = [f for f in dat.estudios["lat"] if int(f[0].split("-l")[1]) > dat.latente]
    if propia not in dat.puntajes or not mayores:
        raise ValueError(f"{FUENTE_TABLA}: el estudio lat no tiene {propia} y un latente mayor")
    mayor = mayores[0]
    semillas = [float(p) for p in dat.puntajes[propia]]
    mejora = float(dat.fila(propia)[2]) - float(mayor[2])
    if mejora >= max(semillas) - min(semillas):
        raise ValueError(
            f"{FUENTE_BARRIDO}: {mayor[0]} mejora a {propia} mas que la diferencia entre semillas; "
            "revisar la frase de la diapositiva de arquitectura"
        )
    mejor = min((f for filas in dat.estudios.values() for f in filas), key=lambda f: float(f[2]))
    if mejor[0] != dat.variante:
        raise ValueError(f"{FUENTE_TABLA}: la variante elegida {dat.variante} no es la de menor puntaje")
    mayor_latente = mayor[0].split("-l")[1]
    texto_justo(
        d, MARGEN_IN, y + 0.5, ANCHO_UTIL,
        vinetas(
            [
                "Objetivo del entrenamiento: el patrón limpio.",
                f"Latente {dat.latente}: con {mayor_latente} el puntaje no mejora más que entre semillas.",
                f"Ruido de entrenamiento {dat.ruido_en_palabras()}: el menor puntaje.",
            ]
        ),
    )
    lo, hi = min(dat.puntajes[propia], key=float), max(dat.puntajes[propia], key=float)
    peor = dat.extremos("lat")[1]
    d.notas(
        [
            "Mensaje: es la red del Ej 1a con un latente más ancho. Lo que cambia es qué se le muestra: entra "
            "el patrón con ruido y se le exige el patrón limpio.",
            "Modelo de ruido: cada píxel se invierte de forma independiente con una probabilidad, que llamamos "
            f"nivel de ruido. En el entrenamiento el ruido se sortea de nuevo en cada época, con {dat.copias} "
            f"copias de los patrones por época, y el nivel queda {dat.ruido_en_palabras()}.",
            "Puntaje, que aparece en esta diapositiva y en las dos siguientes: media de píxeles incorrectos de "
            f"la salida sobre distorsiones con ruido de {dat.niveles_puntaje[0]} a {dat.niveles_puntaje[-1]}, "
            "sorteadas con una semilla propia de la selección. Menor es mejor.",
            f"Por qué latente {dat.latente}: la consigna no fija dos dimensiones para el denoising. En el estudio "
            f"del latente el puntaje va de {peor[2]} con {peor[0]} a {dat.fila(propia)[2]} con {propia}; con "
            f"{mayor[0]} es {mayor[2]}, una diferencia menor que la que hay entre las semillas de {propia}, que "
            f"van de {lo} a {hi}.",
            f"Por qué ese ruido de entrenamiento: {dat.variante} tiene el menor puntaje de todo el barrido, "
            f"{mejor[2]}. Está en la diapositiva del ruido de entrenamiento.",
            f"El resto es igual al Ej 1a: capas ocultas {codificador}, pérdida {dat.perdida} y {dat.optimizador}; "
            f"{dat.epocas} épocas, semilla {dat.semilla}.",
            "Interpretación nuestra: como el objetivo no es la entrada, la red no puede limitarse a copiarla; "
            "para acertar tiene que haber aprendido la forma de las letras.",
        ],
        citas=[(FUENTE, [dat.cfg])],
    )


def _tecnicas_latente(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Técnicas: latente y capas ocultas")
    y = CONTENIDO_Y_IN
    y += d.tabla(MARGEN_IN, y, ANCHO_UTIL, dat.tabla("lat"), anchos=ANCHOS, tam=TAM_TABLA) + 0.25
    y += d.tabla(MARGEN_IN, y, ANCHO_UTIL, dat.tabla("hid"), anchos=ANCHOS, tam=TAM_TABLA) + 0.25
    texto_justo(
        d, MARGEN_IN, y, ANCHO_UTIL,
        f"Puntaje: media de píxeles incorrectos de la salida con ruido de {dat.niveles_puntaje[0]} a "
        f"{dat.niveles_puntaje[-1]}; menor es mejor.",
    )
    notas = [
        "Mensaje: estas dos tablas respaldan el tamaño de la red. Arriba cambia solo el latente; abajo, solo "
        "las capas ocultas.",
    ]
    for estudio, que in (("lat", "del latente"), ("hid", "de las capas ocultas")):
        mejor, peor = dat.extremos(estudio)
        notas.append(
            f"Estudio {que}: el puntaje va de {peor[2]} con {peor[0]} a {mejor[2]} con {mejor[0]}."
        )
    base = dat.base()
    elegibles = all(_k_de_n(f[1])[0] == _k_de_n(f[1])[1] for filas in dat.estudios.values() for f in filas)
    semillas = _k_de_n(dat.estudios["lat"][0][1])[1]
    notas += [
        f"Cómo leer las tablas: cada celda es la mediana de {semillas} semillas. Las columnas «px a nivel» son "
        "los píxeles incorrectos medios de la salida con ese nivel de ruido. El nombre de la variante lleva el "
        "tamaño del latente (l) o las capas ocultas del encoder (h).",
        f"La base se repite con otro nombre en los estudios: {enumerar(base)} son la misma configuración y por "
        "eso tienen la misma fila.",
        "Las columnas que no entran en la diapositiva están en las filas citadas abajo: semillas elegibles, "
        f"porcentaje dentro de {dat.tol} {px(dat.tol)} y pérdida final.",
    ]
    if elegibles:
        notas.append(
            "Todas las variantes son elegibles: con todas sus semillas reconstruyen los patrones limpios con "
            f"{dat.px_ok}, el criterio del Ej 1a."
        )
    notas.append(
        "Límite: son tres semillas por variante y un solo factor por vez respecto de la base; no se probó la "
        "grilla completa de combinaciones."
    )
    variantes = [f[0] for f in dat.estudios["lat"] + dat.estudios["hid"]]
    d.notas(notas, citas=[(FUENTE_TABLA, lineas_md(salidas, EJ, TABLA, variantes))])


def _tecnicas_ruido(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Técnicas: ruido de entrenamiento")
    y = CONTENIDO_Y_IN
    y += d.tabla(MARGEN_IN, y, ANCHO_UTIL, dat.tabla("noise"), anchos=ANCHOS, tam=TAM_TABLA) + 0.35
    texto_justo(
        d, MARGEN_IN, y, ANCHO_UTIL,
        vinetas(
            [
                "«fix»: nivel de ruido fijo. «u0-»: nivel sorteado entre 0 y el valor indicado.",
                f"Elegida: {dat.variante}, la de menor puntaje; dentro de ella, la semilla {dat.semilla}.",
            ]
        ),
    )
    por_perdida = sorted(dat.estudios["noise"], key=lambda f: float(f[7]))
    elegida = dat.fila(dat.variante)
    notas = [
        f"Mensaje: entrenar con ruido {dat.ruido_en_palabras()} dio el menor puntaje de todo el barrido, y por "
        "eso es la configuración del ejercicio.",
        "Regla de selección, escrita en el código: una variante es elegible si todas sus semillas reconstruyen "
        "los patrones limpios dentro de la tolerancia; entre las elegibles gana la de menor mediana de puntaje "
        "y, dentro de ella, la semilla de menor puntaje. El resultado es la línea SELECCION citada abajo.",
        f"El barrido son {dat.corridas} corridas entre los tres estudios, con el ruido de selección sorteado "
        "con una semilla distinta de la del entrenamiento y de la de la evaluación final.",
    ]
    if por_perdida[0][0] != dat.variante:
        notas.append(
            f"Pregunta probable sobre las filas completas: la pérdida final de {dat.variante} ({elegida[7]}) es "
            f"mayor que la de {por_perdida[0][0]} ({por_perdida[0][7]}) y aun así corrige mejor. La pérdida se "
            "mide sobre el lote ruidoso del entrenamiento: con más ruido el lote es más difícil y la pérdida "
            "queda más alta. Por eso no se comparan pérdidas entre variantes con distinto ruido y se elige por "
            "puntaje."
        )
    notas.append(
        "Límite: las diferencias de puntaje entre variantes de ruido son chicas y hay tres semillas por "
        "variante; la regla elige, no prueba que una variante sea mejor en general."
    )
    d.notas(
        notas,
        citas=[
            (FUENTE_TABLA, lineas_md(salidas, EJ, TABLA, [f[0] for f in dat.estudios["noise"]])),
            (FUENTE_BARRIDO, [dat.seleccion]),
        ],
    )


def _distorsion(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Resultados: distorsión de las entradas")
    primero, ultimo = dat.niveles[0], dat.niveles[-1]
    figura_ancha(
        d, salidas, EJ, "noise_examples.png",
        f"Cada píxel se invierte con probabilidad igual al nivel de ruido. Niveles: de {primero} a {ultimo}.",
    )
    con_ruido = [n for n in dat.niveles if float(n) > 0]
    mitad = (len(dat.niveles) + 2) // 2 - 1
    notas = [
        "Mensaje: así se distorsionan las entradas. El nivel de ruido es la probabilidad de invertir cada "
        "píxel, de forma independiente.",
        f"Cómo leer la figura: las mismas ocho letras en cada fila. El bloque izquierdo tiene la fila original "
        f"y los niveles {enumerar(dat.niveles[:mitad])}; el derecho, {enumerar(dat.niveles[mitad:])}.",
        f"Píxeles invertidos por entrada, en promedio: {dat.celda(con_ruido[0], 1)} con ruido {con_ruido[0]} y "
        f"{dat.celda(ultimo, 1)} con {ultimo}. Es el nivel por los {dat.tamanios[0]} píxeles.",
    ]
    if float(primero) == 0:
        notas.append(f"Con nivel {primero} la entrada es el patrón limpio: sirve de control.")
    if float(ultimo) == 0.5:
        notas.append(
            f"Con {ultimo} cada píxel de la entrada queda encendido o apagado con igual probabilidad sea cual "
            "sea la letra: la entrada ya no tiene información del patrón y ningún método puede recuperarlo. "
            "Por eso es el último nivel."
        )
    notas.append(
        "Los ejemplos se sortean con una semilla propia, distinta de la del entrenamiento y de la evaluación."
    )
    d.notas(notas, citas=[(FUENTE, dat.cita_tabla([con_ruido[0], ultimo]))])


def _error_por_nivel(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Resultados: error por nivel de ruido")
    n = dat.nivel_max
    muestras = dat.csv[n]["n_samples"]
    figura_ancha(
        d, salidas, EJ, "dae_error_vs_noise.png",
        f"Ruido {n}: en promedio, {dat.celda(n, 1)} píxeles incorrectos en la entrada y {dat.celda(n, 2)} en "
        "la salida.",
    )
    ultimo = dat.niveles[-1]
    d.notas(
        [
            "Mensaje: el autoencoder saca casi todo el ruido en los niveles bajos y deja de poder a medida que "
            "el ruido crece.",
            "Cómo leer la figura: en el eje horizontal el nivel de ruido; en el vertical, los píxeles "
            "incorrectos medios contra el patrón limpio. La curva azul es la entrada con ruido, sin "
            "autoencoder; la naranja es la salida.",
            f"Cómo se evalúa: {dat.por_patron} distorsiones nuevas de cada patrón en cada nivel, {muestras} "
            "muestras por nivel, sorteadas con una semilla que no es la del entrenamiento ni la de la "
            "selección. Se binariza la salida y se cuentan los píxeles que difieren del patrón limpio.",
            f"En el otro extremo, con ruido {ultimo}, la salida tiene {dat.celda(ultimo, 2)} píxeles incorrectos "
            f"contra {dat.celda(ultimo, 1)} de la entrada: ya casi no corrige.",
            "La tabla completa está citada abajo: nivel, píxeles incorrectos medios de la entrada y de la "
            f"salida, peor caso de la salida y porcentaje de muestras con {dat.px_ok}.",
        ],
        citas=[(FUENTE, [dat.evaluacion] + dat.cita_tabla())],
    )


def _nivel_maximo(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Resultados: nivel máximo eliminado")
    figura_ancha(
        d, salidas, EJ, "dae_within_vs_noise.png",
        [
            destacado(f"Nivel máximo eliminado: {dat.nivel_max}"),
            f"Hasta ese nivel, al menos el {dat.umbral} % de las distorsiones queda con {dat.px_ok}.",
        ],
        fraccion=0.905,
    )
    limite = float(dat.umbral) / 100.0
    bajo = dat.primer_nivel("frac_within_1px", limite)
    entrada_bajo_mitad = dat.primer_nivel("input_frac_within_1px", 0.5)
    ultimo = dat.niveles[-1]
    notas = [
        f"Mensaje: con el criterio del {dat.umbral} %, el autoencoder elimina el ruido hasta el nivel "
        f"{dat.nivel_max}.",
        f"Cómo leer la figura: porcentaje de muestras que quedan con {dat.px_ok}, por nivel de ruido. La curva "
        "azul es la entrada con ruido; la naranja, la salida.",
        f"Definición: el nivel máximo eliminado es el mayor nivel tal que, a ese nivel y a todos los menores, "
        f"al menos el {dat.umbral} % de las distorsiones no vistas queda con {dat.px_ok}. La tolerancia es la "
        f"del Ej 1a. El {dat.umbral} % es una convención de este trabajo.",
    ]
    if bajo is not None:
        notas.append(
            f"Si preguntan qué pasa con otro umbral: con ruido {bajo} queda el {dat.celda(bajo, 4)} %, por "
            f"debajo del {dat.umbral} %. Lo que no depende de la convención es la curva completa, que está "
            "en la tabla citada."
        )
    if entrada_bajo_mitad is not None:
        notas.append(
            f"Sin autoencoder, menos de la mitad de las entradas queda dentro de la tolerancia ya con ruido "
            f"{entrada_bajo_mitad}."
        )
    if float(ultimo) == 0.5:
        notas.append(
            f"Con ruido {ultimo} queda dentro de la tolerancia el {dat.celda(ultimo, 4)} %. A ese nivel la "
            "entrada no tiene información del patrón, así que esos aciertos son azar y no eliminación de ruido."
        )
    d.notas(notas, citas=[(FUENTE, [dat.titular] + dat.cita_tabla())])


def _ejemplos(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Resultados: original, entrada ruidosa y salida")
    figura_alta_centrada(d, salidas, EJ, f"dae_examples_p{dat.nivel_max}.png")
    d.notas(
        [
            f"Mensaje: un ejemplo concreto en el nivel máximo eliminado, {dat.nivel_max}.",
            "Cómo leer la figura: arriba las ocho letras originales; en el medio las mismas letras con ruido, "
            "con la cantidad de píxeles invertidos como pie; abajo la salida del autoencoder, con sus píxeles "
            "incorrectos contra el original como pie.",
            "En estos ocho ejemplos todas las salidas coinciden con el original: el pie de la fila de abajo es "
            "cero en las ocho.",
            f"Son ocho ejemplos, no la medida. La medida es la tabla: con ruido {dat.nivel_max} el "
            f"{dat.celda(dat.nivel_max, 4)} % de las muestras queda con {dat.px_ok}, y el peor caso de la "
            f"salida tiene {dat.celda(dat.nivel_max, 3)} píxeles incorrectos.",
            "Hay una figura igual por cada nivel en la carpeta de salidas, por si piden ver un nivel más alto.",
        ],
        citas=[(FUENTE, dat.cita_tabla([dat.nivel_max]))],
    )


def _conclusiones(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1b: Conclusiones")
    bajo_mitad = dat.primer_nivel("frac_within_1px", 0.5)
    ultimo = dat.niveles[-1]
    if bajo_mitad is None or bajo_mitad == ultimo:
        raise ValueError(f"{EJ}/dae_levels.csv: no hay un nivel intermedio bajo la mitad; revisar las conclusiones")
    texto_justo(
        d, MARGEN_IN, CONTENIDO_Y_IN + 0.5, ANCHO_UTIL,
        vinetas(
            [
                f"Hasta ruido {dat.nivel_max}, al menos el {dat.umbral} % de las distorsiones queda con "
                f"{dat.px_ok}.",
                f"Con ruido {bajo_mitad} lo logra el {dat.celda(bajo_mitad, 4)} %; con {ultimo}, el "
                f"{dat.celda(ultimo, 4)} %.",
                f"Configuración elegida: latente {dat.latente} y ruido de entrenamiento {dat.ruido_en_palabras()}.",
            ]
        ),
        tam=24,
    )
    d.notas(
        [
            "Mensaje de cierre del ejercicio: el autoencoder elimina el ruido bajo casi por completo y pierde "
            "capacidad de forma gradual; no hay un nivel a partir del cual deje de servir de golpe.",
            "Las tres frases salen del reporte: la primera es la línea NIVEL MAXIMO ELIMINADO, la segunda son "
            "dos filas de la tabla por nivel y la tercera es la línea de configuración.",
            f"Límites: solo se probó ruido de inversión de píxeles; el {dat.umbral} % es una convención de este "
            "trabajo; la red elegida es una semilla de su variante, elegida por una regla escrita en el código; "
            "la red es un modelo de estas letras y no se probó con otras.",
            "Si preguntan qué haríamos distinto: más semillas por variante y otros tipos de ruido.",
        ],
        citas=[(FUENTE, [dat.cfg, dat.titular] + dat.cita_tabla([bajo_mitad, ultimo]))],
    )


def agregar(pres, salidas) -> None:
    dat = _Datos(salidas)
    _arquitectura(pres, dat)
    _tecnicas_latente(pres, salidas, dat)
    _tecnicas_ruido(pres, salidas, dat)
    _distorsion(pres, salidas, dat)
    _error_por_nivel(pres, salidas, dat)
    _nivel_maximo(pres, salidas, dat)
    _ejemplos(pres, salidas, dat)
    _conclusiones(pres, dat)
