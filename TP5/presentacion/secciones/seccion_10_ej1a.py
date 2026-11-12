"""Seccion 10: Ejercicio 1a (autoencoder basico con latente de dos dimensiones).

Ocho diapositivas, una por pedido de la consigna (1.a.1 a 1.a.4) mas el cierre. En la
diapositiva queda solo lo que se puede comprobar contra un archivo de output/ o contra la
figura de esa misma diapositiva; la explicacion va a las notas del orador.

Ninguna cifra se escribe a mano, ni en la diapositiva ni en las notas: todas salen de los
reportes, tablas y CSV que regenera scripts/regenerar.sh, leidos con lector.Salidas.
"""

from __future__ import annotations

import math
import re

from disposicion import (
    ANCHO_UTIL,
    TAM_TABLA,
    cifra,
    diagrama,
    enumerar,
    figura_alta,
    figura_ancha,
    lineas_md,
    px,
    texto_justo,
    vinetas,
)
from lector import campo, numero
from pptx_minimo import COLOR_AMBAR, CONTENIDO_Y_IN, MARGEN_IN

EJ = "ej1a"
SALIDA = "salida-ej1a.txt"
FUENTE = f"{EJ}/{SALIDA}"
TABLA = "comparison.md"
FUENTE_TABLA = f"{EJ}/{TABLA}"

# Encabezados cortos de las tablas de comparison.md, en el orden de sus columnas.
ENCABEZADO = ["variante", "semillas que aprenden", "mejor máximo de px", "mediana de épocas", "mediana de pérdida"]
ACTIVACION = {"Tanh": "tanh", "Identity": "identidad", "Sigmoid": "sigmoide", "Relu": "ReLU", "ReLU": "ReLU"}
PERDIDA = {"BCE": "entropía cruzada binaria (BCE)", "MSE": "error cuadrático medio (MSE)"}


def _fila(filas: list[list[str]], prefijo: str) -> list[str]:
    for f in filas:
        if f[0].startswith(prefijo):
            return f
    raise ValueError(f"{FUENTE_TABLA}: ninguna variante empieza con {prefijo!r}")


def _estudio(salidas, nombre: str) -> list[list[str]]:
    encabezado, filas = salidas.tabla_md(EJ, TABLA, nombre)
    if len(encabezado) != len(ENCABEZADO):
        raise ValueError(
            f"{FUENTE_TABLA}: la tabla {nombre!r} tiene {len(encabezado)} columnas "
            f"y la diapositiva espera {len(ENCABEZADO)}"
        )
    return filas


def _columnas(filas: list[list[str]], indices: list[int]) -> list[list[str]]:
    return [[ENCABEZADO[i] for i in indices]] + [[f[i] for i in indices] for f in filas]


class _Datos:
    """Todo lo que las diapositivas toman de los reportes y las tablas."""

    def __init__(self, salidas) -> None:
        self.cfg = salidas.linea(EJ, SALIDA, "configuracion elegida:")
        self.resultado = salidas.linea(EJ, SALIDA, "RESULTADO:")
        self.subconjunto = salidas.linea(EJ, SALIDA, "SUBCONJUNTO:")
        self.ok = numero(self.resultado, r"RESULTADO: (\d+)/\d+ patrones")
        self.total = numero(self.resultado, r"RESULTADO: \d+/(\d+) patrones")
        self.tol = numero(self.resultado, r"a lo sumo (\d+) pixel")
        self.maximo = numero(self.resultado, r"\(maximo (\d+)\)")
        self.nombre = campo(self.cfg, "configuracion elegida")
        self.semilla = campo(self.cfg, "semilla")
        self.epocas = campo(self.cfg, "epocas")
        self.optimizador = campo(self.cfg, "optimizador")
        self.perdida = campo(self.cfg, "perdida")
        self.variante = re.sub(r"-s\d+$", "", self.nombre)
        self.tamanios = [int(n) for n in campo(self.cfg, "arquitectura").split("-")]
        self.arch = _estudio(salidas, "arch")
        self.optim = _estudio(salidas, "optim")
        self.act = _estudio(salidas, "act")
        self.loss = _estudio(salidas, "loss")
        self.fila_elegida = _fila(self.arch, self.variante)
        self.barrido = salidas.csv(EJ, "sweep.csv")
        self.corrida = next((r for r in self.barrido if r["name"] == self.nombre), None)
        if self.corrida is None:
            raise ValueError(f"{EJ}/sweep.csv: no esta la corrida elegida {self.nombre}")
        self.bloque_punto = salidas.bloque(EJ, SALIDA, "punto latente:", 8)
        self.cercana = salidas.linea(EJ, SALIDA, "letra mas cercana del entrenamiento:")
        self.letra_nueva = salidas.linea(EJ, SALIDA, "LETRA NUEVA:")
        self.linea_barrido = salidas.linea(EJ, "salida-sweep.txt", "sweep:")
        self.corridas = numero(self.linea_barrido, r"sweep: (\d+) corridas")
        self.codigos = salidas.csv(EJ, "latent_codes.csv")
        self.px_ok = f"≤ {self.tol} {px(self.tol)} incorrecto"

    def activacion(self, columna: str) -> str:
        valor = self.corrida[columna]
        return ACTIVACION.get(valor, valor)

    def base(self) -> list[str]:
        """Nombres con los que la configuracion base aparece en los estudios (misma fila)."""
        ref = self.loss[0][1:]
        return [f[0] for filas in (self.arch, self.optim, self.act, self.loss) for f in filas if f[1:] == ref]


# --------------------------------------------------------------------------
# Diapositivas
# --------------------------------------------------------------------------


def _arquitectura(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Arquitectura")
    tam = dat.tamanios
    i_lat = tam.index(min(tam))
    latente = tam[i_lat]
    codificador = "-".join(str(n) for n in tam[1:i_lat])
    decodificador = "-".join(str(n) for n in tam[i_lat + 1 : -1])
    y = diagrama(
        d,
        [
            ("Entrada", f"{tam[0]} píxeles", None),
            ("Encoder", f"capas {codificador}", None),
            ("Latente", f"{latente} neuronas", COLOR_AMBAR),
            ("Decoder", f"capas {decodificador}", None),
            ("Salida", f"{tam[-1]} píxeles", None),
        ],
    )
    texto_justo(
        d, MARGEN_IN, y + 0.5, ANCHO_UTIL,
        vinetas(
            [
                f"Activaciones: {dat.activacion('hidden_act')} en las capas ocultas, "
                f"{dat.activacion('latent_act')} en el latente y {dat.activacion('output_act')} en la salida.",
                f"Pérdida: {PERDIDA.get(dat.perdida, dat.perdida)}. Optimizador: {dat.optimizador}.",
            ]
        ),
    )
    d.notas(
        [
            f"Mensaje: un autoencoder simétrico que comprime cada letra de {tam[0]} píxeles en "
            f"{latente} números y la vuelve a armar. Las {latente} neuronas del latente las fija la consigna.",
            f"Cómo leer el diagrama: el encoder pasa los {tam[0]} píxeles por capas de "
            f"{codificador.replace('-', ' y ')} neuronas y termina en el latente; el decoder hace el camino "
            "inverso. La salida tiene una neurona por píxel.",
            "Por qué estas activaciones (interpretación nuestra): la salida sigmoide da un valor entre cero y uno "
            "por píxel, que se lee como probabilidad de que esté encendido, y por eso la pérdida es la entropía "
            "cruzada binaria; el latente con identidad no acota los códigos, así que pueden separarse todo lo "
            "que haga falta.",
            f"Entrenamiento: {dat.epocas} épocas con todo el conjunto en cada paso, semilla {dat.semilla}. "
            "La retropropagación y el optimizador son código propio sobre ndarray, sin bibliotecas de redes.",
            "Si preguntan por qué esta configuración y no otra: se eligió por una regla escrita en el código "
            "sobre el barrido de las tres diapositivas siguientes.",
        ],
        citas=[(FUENTE, [dat.cfg])],
    )


def _tecnicas_arquitecturas(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Técnicas: arquitecturas")
    alto = d.tabla(
        MARGEN_IN, CONTENIDO_Y_IN, ANCHO_UTIL, _columnas(dat.arch, [0, 1, 2, 3, 4]),
        anchos=[1.5, 1.2, 1.1, 1.1, 1.1], tam=TAM_TABLA,
    )
    semillas = dat.fila_elegida[1].split("/")[1]
    texto_justo(
        d, MARGEN_IN, CONTENIDO_Y_IN + alto + 0.3, ANCHO_UTIL,
        vinetas(
            [
                f"El nombre indica las capas ocultas del encoder; {semillas} semillas por variante.",
                f"Elegida: {dat.variante}, la de menor mediana de épocas entre las variantes que aprenden "
                "con todas sus semillas.",
            ]
        ),
    )
    base = dat.base()
    d.notas(
        [
            f"Mensaje: se probaron {len(dat.arch)} tamaños de red con todo lo demás igual, y la elegida es la "
            "que entra antes en la tolerancia con todas sus semillas.",
            f"Cómo leer la tabla: «semillas que aprenden» cuenta las semillas que terminan el entrenamiento con "
            f"los {dat.total} patrones con {dat.px_ok}; «mejor máximo de px» es el peor patrón de la mejor "
            "semilla; «mediana de épocas» es la mediana de la primera época en que la corrida entra en esa "
            "tolerancia; «mediana de pérdida» es la pérdida al final.",
            f"Regla de selección, escrita en el código: entre las variantes de los cuatro estudios que aprenden "
            f"con todas sus semillas se toma la de menor mediana de épocas, {dat.variante} con "
            f"{dat.fila_elegida[3]}; dentro de ella, la semilla de menor pérdida final, la {dat.semilla}.",
            f"El barrido completo son {dat.corridas} corridas en cuatro estudios (arquitecturas, optimizadores, "
            "activaciones y pérdida), cada uno cambiando un solo factor respecto de una base. La base se repite "
            f"en los cuatro con otro nombre: {enumerar(base)} son la misma configuración y por eso tienen la "
            "misma fila.",
            "Límite que conviene decir: los estudios de optimizador, activación y pérdida usan la base chica, no "
            "la arquitectura elegida; no se probó la grilla completa de combinaciones.",
            "Interpretación nuestra, con solo tres semillas por variante: más neuronas por capa aceleran el "
            "aprendizaje; la única variante con una semilla que no aprende es la base.",
        ],
        citas=[(FUENTE_TABLA, lineas_md(salidas, EJ, TABLA, [f[0] for f in dat.arch]))],
    )


def _tecnicas_optimizadores(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Técnicas: optimizadores")
    grupo = lambda prefijo: [f for f in dat.optim if f[0].startswith(prefijo)]  # noqa: E731
    sgd, mom, adam = grupo("optim-sgd"), grupo("optim-mom"), grupo("optim-adam")
    if not (sgd and mom and adam):
        raise ValueError(f"{FUENTE_TABLA}: el estudio optim no tiene SGD, Momentum y Adam")
    cuenta = lambda filas: enumerar([f[1] for f in filas])  # noqa: E731
    figura_ancha(
        d, salidas, EJ, "curves_optim_loss.png",
        f"Semillas que aprenden: SGD {cuenta(sgd)}; Momentum {cuenta(mom)}; Adam {cuenta(adam)}.",
    )
    def corridas(filas: list[list[str]]) -> str:
        ok = sum(int(f[1].split("/")[0]) for f in filas)
        total = sum(int(f[1].split("/")[1]) for f in filas)
        return f"{ok} de {total} corridas"

    perdidas: dict[str, list[str]] = {}
    for f in dat.optim:
        perdidas.setdefault(f[4], []).append(f[0])
    repetida = next(((p, v) for p, v in perdidas.items() if len(v) > 1), None)
    parrafos = [
        "Mensaje: con la misma red, el optimizador decide si se aprende o no. Terminan dentro de la "
        f"tolerancia {corridas(sgd)} con SGD, {corridas(mom)} con Momentum y {corridas(adam)} con Adam.",
        "Cómo leer la figura: pérdida de entrenamiento por época en escala logarítmica, una curva por variante, "
        "solo la semilla 0 de cada una. El nombre de la variante lleva la tasa de aprendizaje (lr) y, en "
        "Momentum, el coeficiente b.",
        "Qué se ve: las curvas de Adam bajan durante todo el entrenamiento, con algún salto brusco del que se "
        "recuperan. Las de SGD y Momentum terminan arriba, cerca del valor inicial; algunas bajan un tramo y "
        "vuelven a subir de golpe.",
        "El pie de la diapositiva cuenta semillas que terminan dentro de la tolerancia, en el orden de la "
        "leyenda. La tabla completa del estudio está citada abajo.",
    ]
    if repetida is not None:
        parrafos.append(
            f"Pregunta probable: por qué {enumerar(repetida[1])} tienen la misma mediana de pérdida, "
            f"{repetida[0]}. Porque esas corridas terminan en el mismo punto: la red devuelve la misma salida "
            "para cualquier letra, la frecuencia con que cada píxel está encendido en el conjunto, e ignora la "
            "entrada. Una prueba del repositorio (tests/experiments.rs) calcula la entropía cruzada de esa "
            "salida constante desde font.h y da ese mismo valor. En sweep.csv varias corridas de SGD y "
            "Momentum terminan exactamente ahí."
        )
    parrafos.append(
        "Interpretación nuestra: con tasas altas y sin adaptación del paso, SGD y Momentum caen en esa solución "
        "degenerada y no salen; Adam ajusta el paso de cada peso y la evita. Son tres semillas por variante."
    )
    d.notas(parrafos, citas=[(FUENTE_TABLA, lineas_md(salidas, EJ, TABLA, [f[0] for f in dat.optim]))])


def _tecnicas_activaciones(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Técnicas: activaciones y pérdida")
    indices = [0, 1, 2, 4]
    anchos = [1.5, 1.2, 1.1, 1.1]
    y = CONTENIDO_Y_IN
    y += d.tabla(MARGEN_IN, y, ANCHO_UTIL, _columnas(dat.act, indices), anchos=anchos, tam=TAM_TABLA) + 0.3
    y += d.tabla(MARGEN_IN, y, ANCHO_UTIL, _columnas(dat.loss, indices), anchos=anchos, tam=TAM_TABLA) + 0.3
    bce, mse = _fila(dat.loss, "loss-bce"), _fila(dat.loss, "loss-mse")
    texto_justo(
        d, MARGEN_IN, y, ANCHO_UTIL,
        f"Con BCE aprenden {bce[1]} semillas; con MSE, {mse[1]}: su mejor semilla termina con "
        f"{mse[2]} {px(mse[2])} incorrectos.",
    )
    entran_y_salen = [
        r for r in dat.barrido
        if r["variant"] == mse[0] and r["first_epoch_within_1px"] and int(r["final_max_wrong"]) > int(dat.tol)
    ]
    notas = [
        "Mensaje: la activación cambia poco el resultado y la pérdida lo cambia mucho. Con error cuadrático "
        f"medio terminan dentro de la tolerancia {mse[1]} semillas.",
        "Cómo leer los nombres: act-tanh-latid es tanh en las capas ocultas e identidad en el latente; "
        "act-relu-lattanh es ReLU en las ocultas y tanh en el latente. loss-bce y loss-mse cambian solo la "
        "pérdida.",
        "Las columnas son las mismas de la tabla de arquitecturas, sin la mediana de épocas. La pérdida final "
        "de BCE y la de MSE no se comparan entre sí: son funciones distintas.",
        f"Pregunta probable sobre la tabla completa, citada abajo: loss-mse figura con {mse[1]} semillas y aun "
        f"así con una mediana de épocas de {mse[3]}. No es una contradicción: esa columna mide la primera "
        f"época en que la corrida entra en la tolerancia, y con MSE {len(entran_y_salen)} semillas entran y "
        "después salen (se ve en sweep.csv); la primera columna mira solo el final.",
        "Interpretación nuestra: con salida sigmoide, el gradiente de la entropía cruzada no lleva el factor "
        "de la derivada de la sigmoide, que con MSE frena el aprendizaje cuando la salida satura.",
        "Límite: tres semillas por variante; la diferencia entre dos y tres semillas que aprenden es una sola "
        "corrida.",
    ]
    variantes = [f[0] for f in dat.act + dat.loss]
    d.notas(notas, citas=[(FUENTE_TABLA, lineas_md(salidas, EJ, TABLA, variantes))])


def _resultado(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva(f"Ej 1a: Resultados: {dat.ok}/{dat.total} patrones con {dat.px_ok}")
    ancho = 4.6
    hueco = 0.6
    x = MARGEN_IN + (ANCHO_UTIL - 2 * ancho - hueco) / 2
    y = CONTENIDO_Y_IN + 0.3
    y1 = cifra(d, x, y, ancho, f"{dat.ok}/{dat.total}", f"patrones con {dat.px_ok}")
    y2 = cifra(d, x + ancho + hueco, y, ancho, dat.maximo, "píxeles incorrectos en el peor patrón")
    if int(dat.ok) != int(dat.total) or "no queda ningun patron sin aprender" not in dat.subconjunto:
        raise ValueError(f"{FUENTE}: el reporte no dice que se aprendio el conjunto completo; revisar la diapositiva")
    texto_justo(
        d, MARGEN_IN, max(y1, y2) + 0.5, ANCHO_UTIL,
        f"Se aprendió el conjunto completo de {dat.total} patrones: no hizo falta un subconjunto.",
    )
    d.notas(
        [
            f"Mensaje: se cumple el objetivo de la consigna. Los {dat.total} patrones se reconstruyen con "
            f"{dat.px_ok} pasando por un latente de 2 dimensiones; de hecho el peor patrón tiene {dat.maximo}.",
            "Cómo se mide: se pasa cada patrón por la red, se binariza la salida en un medio y se cuentan los "
            "píxeles que difieren del patrón original.",
            "La consigna pide justificar si se aprende solo un subconjunto. No es el caso: el reporte lo dice "
            "en la línea SUBCONJUNTO citada abajo.",
            f"Red usada: {dat.nombre}, la elegida por la regla del barrido.",
            "Límite que conviene decir sin que lo pregunten: se evalúa sobre los mismos patrones del "
            "entrenamiento, porque eso es lo que pide la consigna. Es memorizar el conjunto, no generalizar.",
        ],
        citas=[(FUENTE, [dat.cfg, dat.resultado, dat.subconjunto])],
    )


def _latente(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Resultados: espacio latente")
    x, ancho = figura_alta(d, salidas, EJ, "latent.png")
    distintos = len({(r["z1"], r["z2"]) for r in dat.codigos})
    y = cifra(d, x, CONTENIDO_Y_IN + 0.4, ancho, f"{distintos}/{len(dat.codigos)}", "códigos latentes distintos")
    texto_justo(d, x, y + 0.4, ancho, "Cada punto es el código (z1, z2) de un patrón.")

    por_z1 = sorted(dat.codigos, key=lambda r: float(r["z1"]))
    por_origen = sorted(dat.codigos, key=lambda r: math.hypot(float(r["z1"]), float(r["z2"])))
    izquierda = [r["label"] for r in por_z1[:6]]
    centro = [r["label"] for r in por_origen[:9]]
    d.notas(
        [
            "Mensaje: este es el gráfico que pide la consigna, los datos de entrada en el espacio latente. "
            "Cada patrón tiene su propio código y ningún par coincide.",
            "Cómo leer la figura: cada punto azul es la salida del encoder para una letra, con la letra al "
            "lado; cuando no entra a la derecha del punto, la etiqueta se corre y se une al punto con una "
            "línea fina. Los ejes no están acotados porque el latente usa identidad.",
            f"Qué se ve, calculado desde latent_codes.csv: las letras más a la izquierda son, en orden, "
            f"{enumerar(izquierda)}; la siguiente es {por_z1[6]['label']}. Los códigos más cercanos al origen "
            f"son, en orden, {enumerar(centro)}: es la zona donde los códigos están más juntos.",
            "Interpretación nuestra, no medida: las letras del borde izquierdo tienen un trazo vertical a la "
            "izquierda, y el encoder parece usar z1 para eso. Es una tendencia que se ve en el gráfico, no una "
            "regla que hayamos probado.",
            "Si preguntan qué significan z1 y z2: no tienen un significado impuesto; son las dos coordenadas "
            "que la red encontró para distinguir los patrones.",
            "La letra nueva de la diapositiva siguiente sale de un punto de la zona apretada cerca del origen.",
        ],
        citas=[(f"{EJ}/latent_codes.csv", [f"{r['label']},{r['z1']},{r['z2']}" for r in por_z1[:7]])],
    )


def _letra_nueva(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Resultados: letra nueva")
    x, ancho = figura_alta(d, salidas, EJ, "new_letter.png")
    punto = numero(dat.bloque_punto[0], r"punto latente: (\(.+\))")
    cercana = numero(dat.cercana, r"entrenamiento: (\S+) \(")
    distancia = numero(dat.cercana, r"distancia Hamming (\d+)\)")
    minimo = numero(dat.letra_nueva, r"en al menos (\d+) pixeles")
    lejos = numero(dat.letra_nueva, r"mas cercano (\d+\.\d+)")
    otros = salidas.lineas_con_prefijo(EJ, SALIDA, "otros candidatos:")
    texto_justo(
        d, x, CONTENIDO_Y_IN + 0.3, ancho,
        vinetas(
            [
                f"Punto latente {punto}: no es el código de ningún patrón.",
                "Se decodifica ese punto y se binariza la salida.",
                f"Letra más cercana del entrenamiento: {cercana}, a distancia de Hamming {distancia}.",
            ]
        ),
    )
    d.notas(
        [
            "Mensaje: el decoder solo, sin encoder, genera un patrón que no está en el conjunto de "
            "entrenamiento a partir de un punto del plano latente.",
            "Cómo se genera: se recorre una grilla regular de puntos sobre la zona del plano que ocupan los "
            "códigos, se decodifica cada punto y se binariza la salida. Se descartan los que dan un patrón del "
            f"entrenamiento o uno a menos de {minimo} píxeles de alguno, para que no pase por una "
            f"reconstrucción con {dat.tol} {px(dat.tol)} de error, que es la tolerancia de la consigna. Entre "
            "los que quedan se elige el más parecido a una letra real.",
            f"Cómo leer la figura: a la izquierda la letra generada, a la derecha la {cercana} del conjunto. "
            f"Difieren en {distancia} píxeles: la generada tiene un píxel encendido de más en la cuarta fila y "
            "le falta la esquina inferior derecha.",
            f"El punto elegido está a {lejos} del código de entrenamiento más cercano, en unidades del plano "
            "latente: no es un código de entrenamiento.",
            f"Juicio nuestro: es una forma nueva pero no una letra legible; se lee como una {cercana} "
            "deformada. Muestra que la red genera patrones fuera del conjunto, no que invente letras con "
            f"sentido. El reporte lista {len(otros)} candidatos más, cada uno con su letra más cercana.",
        ],
        citas=[(FUENTE, dat.bloque_punto + [dat.cercana, dat.letra_nueva])],
    )


def _conclusiones(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 1a: Conclusiones")
    grupo = lambda prefijo: enumerar([f[1] for f in dat.optim if f[0].startswith(prefijo)])  # noqa: E731
    bce, mse = _fila(dat.loss, "loss-bce"), _fila(dat.loss, "loss-mse")
    latente = min(dat.tamanios)
    texto_justo(
        d, MARGEN_IN, CONTENIDO_Y_IN + 0.5, ANCHO_UTIL,
        vinetas(
            [
                f"Con un latente de {latente} dimensiones, {dat.ok}/{dat.total} patrones quedan con {dat.px_ok}.",
                f"Semillas que aprenden: Adam {grupo('optim-adam')}; SGD {grupo('optim-sgd')}.",
                f"Con BCE aprenden {bce[1]} semillas; con MSE, {mse[1]}.",
            ]
        ),
        tam=24,
    )
    semillas = dat.fila_elegida[1].split("/")[1]
    d.notas(
        [
            "Mensaje de cierre del ejercicio: el objetivo de la consigna se cumple con el conjunto completo y, "
            "a nuestro juicio, lo que más pesó para lograrlo fue el optimizador y la pérdida, no el tamaño de "
            "la red.",
            "Las tres frases de la diapositiva salen de los reportes: la primera de la línea RESULTADO, la "
            "segunda de la tabla de optimizadores y la tercera de la tabla de pérdida.",
            f"Límites: el criterio es reconstruir los mismos patrones del entrenamiento; la red elegida es una "
            f"semilla entre las {semillas} de su variante; con {semillas} semillas por variante las diferencias "
            "chicas entre variantes no son concluyentes; la letra nueva no es una letra legible.",
            "Si preguntan qué haríamos distinto: probar la pérdida y el optimizador sobre la arquitectura "
            "elegida y no solo sobre la base, y más semillas por variante.",
        ],
        citas=[(FUENTE, [dat.resultado]), (FUENTE_TABLA, lineas_md(salidas, EJ, TABLA, [f[0] for f in dat.optim + dat.loss]))],
    )


def agregar(pres, salidas) -> None:
    dat = _Datos(salidas)
    _arquitectura(pres, dat)
    _tecnicas_arquitecturas(pres, salidas, dat)
    _tecnicas_optimizadores(pres, salidas, dat)
    _tecnicas_activaciones(pres, salidas, dat)
    _resultado(pres, salidas, dat)
    _latente(pres, salidas, dat)
    _letra_nueva(pres, salidas, dat)
    _conclusiones(pres, salidas, dat)
