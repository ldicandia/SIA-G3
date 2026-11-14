"""Seccion 30: Ejercicio 2 (autoencoder variacional sobre bloques de mena de Minecraft).

Siete diapositivas: el dataset (2.a), el esquema variacional y lo que resuelve (2.b) y el
bloque nuevo con el juicio de los autores (2.c). En la diapositiva queda solo lo comprobable
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
    diagrama,
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
        self.bloques = linea("bloques de mena:")
        self.separacion = linea("separacion")
        self.cfg = linea("configuracion:")
        self.perdida = linea("PERDIDA:")
        self.reconstruccion = linea("RECONSTRUCCION:")
        self.juez = linea("JUEZ:")
        self.manifold = linea("MANIFOLD:")
        self.nuevo = linea("BLOQUE NUEVO:")
        self.galeria = linea("GALERIA:")
        self.prior = linea("VAE PRIOR:")
        self.prior_simple = linea("AE simple PRIOR:")
        self.control = linea("CONTROL NEGATIVO:")
        self.contraste = linea("CONTRASTE:")
        self.resultado = linea("RESULTADO EJ2:")
        self.aceptacion = salidas.lineas_con_prefijo(EJ, SALIDA, "ACEPTACION")
        self.entradas_galeria = salidas.lineas_con_prefijo(EJ, SALIDA, "  nuevo ")

        self.n = numero(self.bloques, r"bloques de mena: (\d+) texturas")
        self.version = numero(self.bloques, r"Minecraft (\S+) de")
        self.lado = numero(self.bloques, r"de (\d+x\d+) RGB")
        self.valores = numero(self.bloques, r"\((\d+) valores")
        self.minerales = numero(self.bloques, r"\| (\d+) minerales")
        self.rocas = numero(self.bloques, r"minerales, (\d+) rocas")
        self.sep_min = numero(self.separacion, r"minima (\d+\.\d+) niveles")
        self.sep_mediana = numero(self.separacion, r"mediana (\d+\.\d+) niveles")

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

        self.rec_niveles = numero(self.reconstruccion, r"error medio (\d+\.\d+) niveles")
        self.rec_media = numero(self.reconstruccion, r"media (\d+\.\d+), maximo")
        self.rec_max = numero(self.reconstruccion, r"maximo (\d+) de")
        self.rec_exactos = numero(self.reconstruccion, r"; (\d+/\d+) bloques sin")

        self.fondo_min = numero(self.juez, r"al menos (\d+) de 256")
        self.peso_min = numero(self.juez, r"al menos (\d+) % del peso")
        self.peso_ajeno = numero(self.juez, r"ajeno en las reconstrucciones: (\d+\.\d+) %")
        self.tope_residuo = numero(self.juez, r"residuo de la mezcla a lo sumo (\d+\.\d+) niveles")
        self.contraste_min = numero(self.juez, r"contraste al menos (\d+\.\d+) niveles")

        self.grilla = numero(self.manifold, r"\((\d+x\d+)\)")
        self.rango = numero(self.manifold, r"z en \[-(\d+\.\d+), ")
        self.celdas_plausibles = numero(self.manifold, r"plausibles (\d+\.\d+) %")
        self.celdas_reconocibles = numero(self.manifold, r"reconocibles (\d+\.\d+) %")
        self.salto = numero(self.contraste, r"vecinas: VAE (\d+\.\d+) contra")
        self.salto_simple = numero(self.contraste, r"vecinas: VAE \S+ contra (\d+\.\d+) niveles")

        self.sorteos = numero(self.prior, r"de (\d+) puntos")
        self.plausibles = numero(self.contraste, r"bloque plausible: VAE (\d+\.\d+) %")
        self.plausibles_simple = numero(self.contraste, r"sin esquema variacional (\d+\.\d+) %")
        self.reconocibles = numero(self.contraste, r"reconocible: VAE (\d+\.\d+) %")
        self.reconocibles_simple = numero(self.contraste, r"reconocible: VAE \S+ % contra (\d+\.\d+) %")
        self.nuevos = numero(self.contraste, r"nuevo: VAE (\d+\.\d+) %")
        self.nuevos_simple = numero(self.contraste, r"nuevo: VAE \S+ % contra (\d+\.\d+) %")
        self.combinaciones = numero(self.prior, r"en (\d+) combinaciones distintas")
        self.combinaciones_simple = numero(self.prior_simple, r"en (\d+) combinaciones distintas")

        self.z = numero(self.nuevo, r"z=(\(.+?\))")
        self.sorteo = numero(self.nuevo, r"sorteo (\d+) de")
        self.combinacion = numero(self.nuevo, r"\| (\S+) \(combinacion ausente")
        self.mezcla = numero(self.nuevo, r"mezcla: (.+?) \|")
        self.fondo = numero(self.nuevo, r"fondo: (\d+) de 256")
        self.residuo = numero(self.nuevo, r"residuo (\d+\.\d+) niveles")
        self.cercano = numero(self.nuevo, r"bloque mas cercano: (\S+) a")
        self.distancia = numero(self.nuevo, r"bloque mas cercano: \S+ a (\d+\.\d+) niveles")
        if numero(self.resultado, r"bloque nuevo (\S+)\)") != self.combinacion:
            raise ValueError(f"{FUENTE}: RESULTADO EJ2 y BLOQUE NUEVO no nombran la misma combinacion")
        self.aceptados = numero(self.galeria, r"GALERIA: (\d+) sorteos aceptados")

        self.control_medias = numero(self.control, r"como nuevas: (\d+/\d+) \(")
        self.control_propias = numero(self.control, r"propia combinacion: (\d+/\d+)\)")
        self.control_muestras = numero(self.control, r"por bloque: (\d+\.\d+) %")
        self.ok = numero(self.resultado, r"\((\d+)/\d+ comprobaciones")
        self.total = numero(self.resultado, r"\(\d+/(\d+) comprobaciones")
        if len(self.aceptacion) != int(self.total) or not all(l.startswith("ACEPTACION OK") for l in self.aceptacion):
            raise ValueError(f"{FUENTE}: las lineas de aceptacion no coinciden con RESULTADO EJ2")

        # Pasajes del documento de analisis, leidos y nunca retipeados.
        combinaciones = _leer_doc("El dataset tiene 18 de las 27 combinaciones")
        self.presentes = numero(combinaciones, r"tiene (\d+) de las")
        self.posibles = numero(combinaciones, r"de las (\d+) combinaciones")
        perdida = _leer_doc("la BCE no puede bajar de la entropía")
        self.piso_bce = numero(perdida, r"es de unos (\d+\.\d+) nats")
        juicio = _leer_doc("En resumen,")
        self.juicio_claras = numero(juicio, r"En resumen, (\d+) de las")
        self.juicio_entradas = numero(juicio, r"de las (\d+) entradas")
        self.juicio_sutiles = numero(juicio, r"(\d+) es sutil")
        self.juicio_debiles = numero(juicio, r"(\d+) son débiles")
        if int(self.juicio_entradas) != len(self.entradas_galeria):
            raise ValueError(f"{DOC.name}: el juicio habla de {self.juicio_entradas} entradas y la galeria tiene otra cantidad")


# --------------------------------------------------------------------------
# Diapositivas
# --------------------------------------------------------------------------


def _dataset(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Dataset")
    figura_ancha(
        d, salidas, EJ, "ores.png",
        f"{dat.n} texturas de Minecraft {dat.version}; {dat.presentes} de las {dat.posibles} combinaciones de "
        "mineral y roca.",
        fraccion=0.866,
    )
    d.notas(
        [
            "Mensaje: la consigna permite elegir el dataset; usamos las texturas reales de los bloques de "
            "mena de Minecraft.",
            f"Cómo leer la figura: son los {dat.n} bloques con su nombre, mineral y roca. Primero los ocho "
            "minerales en piedra, después los mismos en pizarra profunda y al final el oro y el cuarzo del "
            "Nether. El mineral se reconoce por el color de las vetas y la roca por el fondo.",
            f"Cada imagen tiene {dat.lado} píxeles con tres canales de color: {dat.valores} valores entre cero "
            "y uno. Las texturas son de Mojang y no se versionan: un script las baja del cliente oficial del "
            "juego y verifica sus hashes, así el dataset es el mismo en cualquier máquina.",
            f"Distancia entre bloques: el error medio por canal, en niveles de 0 a 255. El par más parecido "
            f"está a {dat.sep_min} niveles y la mediana de la distancia al vecino más cercano es "
            f"{dat.sep_mediana}.",
            f"Lo que importa para el último punto: hay {dat.minerales} minerales y {dat.rocas} rocas, pero el "
            f"dataset tiene solo {dat.presentes} de las {dat.posibles} combinaciones. Un bloque nuevo puede ser "
            "una combinación que falta o un bloque con dos minerales.",
        ],
        citas=[(FUENTE, [dat.bloques, dat.separacion])],
    )


def _arquitectura(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Arquitectura")
    y = diagrama(
        d,
        [
            ("Entrada", f"{dat.encoder[0]} valores", None),
            ("Encoder", "-".join(dat.encoder[1:-1]), None),
            ("μ y log σ²", f"{dat.encoder[-1]} salidas", None),
            ("z = μ + σ·ε", f"{dat.latente} dimensiones", COLOR_AMBAR),
            ("Decoder", "-".join(dat.decoder[1:-1]), None),
            ("Salida", f"{dat.decoder[-1]} valores", None),
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
            f"Cómo leer el diagrama: la entrada son los {dat.encoder[0]} valores de color de un bloque. El "
            f"encoder tiene {dat.encoder[-1]} salidas, una media y un logaritmo de varianza por cada una de las "
            f"{dat.latente} dimensiones latentes. Con ellas se arma z sumando a la media un ruido ε, sorteado de "
            "una normal estándar, multiplicado por el desvío. El decoder recibe z.",
            "Por qué se escribe así, el truco de reparametrización: el sorteo queda fuera de la red, en ε, y el "
            "gradiente puede llegar al encoder a través de la media y del desvío.",
            "La pérdida tiene dos términos. La reconstrucción es la entropía cruzada binaria entre la salida y "
            "la imagen, canal por canal. El KL mide cuánto se aparta la distribución del encoder del prior, una "
            "normal estándar; tiene forma cerrada y se suma con peso uno.",
            "Qué problema resuelve, que es lo que pide la consigna: en un autoencoder simple nada obliga a los "
            "códigos a ocupar una zona conocida del plano, así que no se sabe de dónde sortear para generar. El "
            "término KL empuja los códigos hacia el prior, y entonces sortear del prior da puntos que el "
            "decoder sabe decodificar. La evidencia está en la diapositiva siguiente.",
            f"Entrenamiento: {dat.optimizador}, {dat.epocas} épocas con los {dat.n} bloques en cada paso y ruido "
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
        f"Sorteos del prior que dan un bloque plausible: VAE {dat.plausibles} %; autoencoder simple "
        f"{dat.plausibles_simple} %.",
    )
    d.notas(
        [
            "Mensaje: esta es la evidencia de que el esquema variacional resuelve el problema del espacio "
            "latente. Sorteando del prior, el VAE da bloques plausibles la mayoría de las veces y el "
            "autoencoder simple, bastante menos.",
            "Qué se compara: el autoencoder simple tiene la misma arquitectura, optimizador, pérdida de "
            "reconstrucción, épocas y semilla que el VAE, sin la capa estocástica y sin el término KL.",
            f"Cómo leer las figuras: cada una decodifica una grilla de {dat.grilla} puntos del plano latente. La "
            f"del VAE recorre z entre menos {dat.rango} y {dat.rango} en cada eje, la zona donde cae casi todo "
            "sorteo del prior. La del autoencoder simple recorre la caja que ocupan sus propios códigos, porque "
            "no tiene prior.",
            "Qué se ve en el VAE: el plano se ordena por mineral, azul y celeste arriba, rojo y amarillo a la "
            "izquierda, verde a la derecha y el Nether abajo a la izquierda, con transiciones graduales. El "
            "autoencoder simple también recorre los minerales, pero con zonas de colores lavados y bordes más "
            "duros.",
            f"Cómo se decide si un bloque es plausible: el fondo coincide con la textura de una roca en al menos "
            f"{dat.fondo_min} de 256 píxeles, la imagen se explica como mezcla de bloques del dataset con un "
            f"residuo de a lo sumo {dat.tope_residuo} niveles y tiene al menos {dat.contraste_min} niveles de "
            "contraste, para descartar imágenes borrosas.",
            f"En la grilla del VAE es plausible el {dat.celdas_plausibles} % de las celdas. Los números del pie "
            f"salen de sortear {dat.sorteos} puntos del prior en cada modelo. Reconocibles, es decir plausibles "
            f"y con una combinación del dataset: {dat.reconocibles} % contra {dat.reconocibles_simple} %.",
            "Lo que no favorece al VAE, para decirlo antes de que lo pregunten: el autoencoder simple "
            "reconstruye igual de bien. Reconstruir bien no alcanza para generar.",
            "Límite: el contraste usa una sola arquitectura y una sola semilla.",
        ],
        citas=[(FUENTE, [dat.contraste, dat.prior, dat.prior_simple, dat.manifold, dat.juez])],
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
            f"La reconstrucción baja de {dat.rec_ini} a {dat.rec_fin}. Parece poco, pero los objetivos son "
            "colores y no ceros y unos, y con entropía cruzada la pérdida no puede bajar de la entropía de los "
            f"objetivos, unos {dat.piso_bce} nats por bloque. El final queda a menos de un nat de ese piso.",
            f"El KL empieza en {dat.kl_ini}, sube mientras el decoder aprende a usar el código y se estabiliza "
            f"cerca de {dat.kl_fin}. Que no sea cero indica que el encoder usa el código: distintos bloques "
            "tienen distribuciones distintas.",
            "Las unidades son nats por imagen, sumados sobre todos los valores y promediados sobre el dataset.",
        ],
        citas=[(FUENTE, [dat.perdida])],
    )


def _reconstruccion(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: reconstrucción")
    figura_ancha(
        d, salidas, EJ, "vae_reconstruction.png",
        f"Error medio: {dat.rec_niveles} niveles por canal; {dat.rec_exactos} bloques sin píxeles incorrectos.",
        fraccion=0.713,
    )
    d.notas(
        [
            "Mensaje: el VAE reconstruye los bloques sin errores visibles; se reconoce cuál es cada uno.",
            "Cómo leer la figura: en cada par de filas, arriba están los originales y debajo su "
            "reconstrucción. El pie es la cantidad de píxeles incorrectos.",
            "Un píxel es incorrecto si alguno de sus tres canales difiere en más de 32 niveles del original. "
            f"Ningún bloque tiene píxeles incorrectos, y el error medio es de {dat.rec_niveles} niveles por "
            "canal, que no se ve a simple vista.",
            "La reconstrucción se hace desde la media del encoder, sin sortear ruido.",
            "El autoencoder simple también reconstruye sin píxeles incorrectos: la diferencia entre los dos "
            "modelos no está en reconstruir sino en generar, como se vio en el contraste.",
        ],
        citas=[(FUENTE, [dat.reconstruccion, dat.contraste])],
    )


def _bloque_nuevo(pres, salidas, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Resultados: bloque nuevo")
    figura_ancha(
        d, salidas, EJ, "vae_new_ore.png",
        f"Nuevo 1: {dat.mezcla}, sobre piedra. Los autores lo juzgan parte del conjunto.",
        fraccion=0.88,
    )
    d.notas(
        [
            "Mensaje: sorteando del prior y decodificando, el VAE genera bloques de mena que no están en el "
            "dataset y que juzgamos del mismo conjunto. Es lo que pide el último punto de la consigna.",
            "Cómo leer la figura: ocho bloques generados, cada uno con su número de orden y los minerales que "
            "le encuentra el juez. La roca se ve en el fondo.",
            "Qué llamamos nuevo: un bloque plausible, con el criterio de la diapositiva del contraste, cuya "
            "combinación de roca y minerales no existe en el dataset. Para saber qué minerales tiene, la imagen "
            "se descompone como mezcla de los bloques del dataset, y un mineral cuenta si pesa al menos el "
            f"{dat.peso_min} % de la mezcla.",
            f"El que presentamos es el primero: el sorteo {dat.sorteo} de {dat.sorteos}. Es {dat.combinacion}: "
            f"{dat.mezcla}, una mena que no existe en el dataset ni en el juego. Su fondo coincide con la piedra en {dat.fondo} píxeles y su bloque más cercano, "
            f"{dat.cercano}, está a {dat.distancia} niveles, del orden de lo que separa a dos bloques distintos.",
            f"La galería elige un sorteo por combinación, el de mezcla más equilibrada. De {dat.sorteos} "
            f"sorteos, {dat.aceptados} se aceptan como nuevos.",
            f"El control negativo: de las reconstrucciones del dataset se aceptan como nuevas "
            f"{dat.control_medias}, y se reconocen con su propia combinación {dat.control_propias}. Con ruido, "
            f"se acepta el {dat.control_muestras} % de las reconstrucciones, mucho menos que los sorteos del "
            "prior. El criterio no acepta cualquier salida del decoder.",
            f"Juicio de los autores, que es lo que pide la consigna: a simple vista, {dat.juicio_claras} de las "
            f"{dat.juicio_entradas} se ven como bloques nuevos, {dat.juicio_sutiles} es sutil y "
            f"{dat.juicio_debiles} son débiles. Los minerales con color propio dan bloques nuevos claros; los "
            "apagados, como el carbón y el hierro, dan mezclas que cuesta distinguir.",
            "Límites: el juicio es visual y subjetivo, y el juez solo puede encontrar novedad que sea "
            "combinación de lo que ya hay en el dataset.",
        ],
        citas=[(FUENTE, [dat.nuevo, dat.juez, dat.galeria] + dat.entradas_galeria + [dat.control])],
    )


def _conclusiones(pres, dat: _Datos) -> None:
    d = pres.diapositiva("Ej 2: Conclusiones")
    texto_justo(
        d, MARGEN_IN, CONTENIDO_Y_IN + 0.5, ANCHO_UTIL,
        vinetas(
            [
                f"Sorteos del prior que dan un bloque plausible: VAE {dat.plausibles} %; autoencoder simple "
                f"{dat.plausibles_simple} %.",
                f"Reconstrucción: {dat.rec_niveles} niveles de error medio por canal.",
                f"Bloque nuevo: {dat.combinacion}; los autores lo juzgan parte del conjunto.",
            ]
        ),
        tam=24,
    )
    d.notas(
        [
            "Mensaje de cierre del ejercicio: el esquema variacional vuelve muestreable el espacio latente, y "
            "de ese espacio salen bloques nuevos que juzgamos del conjunto.",
            "Las tres frases salen del reporte: la primera de la línea CONTRASTE, la segunda de la línea "
            "RECONSTRUCCION y la tercera de la línea BLOQUE NUEVO más el juicio del documento de análisis.",
            f"El reporte termina con {dat.total} comprobaciones de aceptación y las {dat.ok} dan bien. Sus "
            "umbrales son convenciones de este trabajo, no medidas externas.",
            f"El VAE también genera más variedad: {dat.combinaciones} combinaciones nuevas distintas contra "
            f"{dat.combinaciones_simple} del autoencoder simple.",
            "Límites: el dataset tiene pocas imágenes; el latente tiene dos dimensiones y no se estudiaron "
            "otras arquitecturas; el contraste con el autoencoder simple usa una sola semilla; los bloques "
            "nuevos son mezclas de minerales en la misma roca y no aparece un mineral en una roca ajena; el "
            "juicio final es visual.",
        ],
        citas=[(FUENTE, [dat.contraste, dat.reconstruccion, dat.prior, dat.prior_simple] + dat.aceptacion + [dat.resultado])],
    )


def agregar(pres, salidas) -> None:
    dat = _Datos(salidas)
    _dataset(pres, salidas, dat)
    _arquitectura(pres, dat)
    _contraste(pres, salidas, dat)
    _perdida(pres, salidas, dat)
    _reconstruccion(pres, salidas, dat)
    _bloque_nuevo(pres, salidas, dat)
    _conclusiones(pres, dat)
