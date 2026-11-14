"""Seccion 00: diapositiva de titulo y diapositiva Resumen.

Ninguna cifra se escribe a mano: todas salen de los reportes impresos que
regenera scripts/regenerar.sh, leidos con lector.Salidas.
"""

from __future__ import annotations

from disposicion import ANCHO_UTIL, cifra, px
from lector import numero
from pptx_minimo import CONTENIDO_Y_IN, MARGEN_IN

TITULO_PORTADA = "TP5: autoencoders (básico, denoising y variacional)"


def agregar(pres, salidas) -> None:
    l1a = salidas.linea("ej1a", "salida-ej1a.txt", "RESULTADO:")
    ok_1a = numero(l1a, r"RESULTADO: (\d+)/\d+ patrones")
    total_1a = numero(l1a, r"RESULTADO: \d+/(\d+) patrones")
    tolerancia = numero(l1a, r"a lo sumo (\d+) pixel")
    px_ok = f"≤ {tolerancia} {px(tolerancia)} incorrecto"

    l1b = salidas.linea("ej1b", "salida-ej1b.txt", "NIVEL MAXIMO ELIMINADO:")
    nivel_1b = numero(l1b, r"NIVEL MAXIMO ELIMINADO: (\d+\.\d+)")
    umbral_1b = numero(l1b, r"al menos el (\d+) %")

    prior = salidas.linea("ej2", "salida-ej2.txt", "VAE PRIOR:")
    plausibles = numero(prior, r"(\d+\.\d+) % decodifican a un bloque plausible")
    sorteos = numero(prior, r"de (\d+) puntos")
    contraste = salidas.linea("ej2", "salida-ej2.txt", "CONTRASTE:")
    simple = numero(contraste, r"sin esquema variacional (\d+\.\d+) %")

    d = pres.diapositiva(TITULO_PORTADA, portada=True)
    d.texto(0.9, 4.2, 11.5, 0.6, "Sistemas de Inteligencia Artificial, ITBA", tam=24)
    d.notas(
        "Presentación del TP5. Son tres ejercicios sobre autoencoders: el básico con latente de dos "
        "dimensiones, el denoising y el variacional. El recorrido sigue los ítems de la consigna, en orden."
    )

    d = pres.diapositiva("Resumen: un resultado por ejercicio")
    hueco = 0.4
    ancho = (ANCHO_UTIL - 2 * hueco) / 3
    columnas = [
        (f"{ok_1a}/{total_1a}", f"Ej 1a: patrones con {px_ok}"),
        (nivel_1b, f"Ej 1b: nivel máximo eliminado (criterio del {umbral_1b} %)"),
        (f"{plausibles} %", "Ej 2: sorteos del prior que dan un bloque plausible"),
    ]
    for i, (valor, etiqueta) in enumerate(columnas):
        cifra(d, MARGEN_IN + i * (ancho + hueco), CONTENIDO_Y_IN + 1.2, ancho, valor, etiqueta)
    d.notas(
        [
            "Mensaje: un número por ejercicio, cada uno con la definición que se usa en el resto de la "
            "presentación.",
            f"Ej 1a, autoencoder básico: los {total_1a} patrones de font.h se reconstruyen con {px_ok} pasando "
            "por un latente de dos dimensiones. Es el objetivo que fija la consigna.",
            f"Ej 1b, denoising: llamamos nivel máximo eliminado al mayor nivel de ruido tal que, a ese nivel y a "
            f"todos los menores, al menos el {umbral_1b} % de las distorsiones no vistas queda con {px_ok}. El "
            "nivel de ruido es la probabilidad de invertir cada píxel. Ese umbral es una convención nuestra.",
            f"Ej 2, variacional: de {sorteos} puntos sorteados del prior, ese porcentaje se decodifica a un "
            "bloque de mena plausible, es decir, con el fondo de una roca del dataset, vetas de minerales del "
            f"dataset y la nitidez de un bloque real. El mismo autoencoder sin esquema variacional llega a {simple} %.",
        ],
        citas=[
            ("ej1a/salida-ej1a.txt", [l1a]),
            ("ej1b/salida-ej1b.txt", [l1b]),
            ("ej2/salida-ej2.txt", [prior, contraste]),
        ],
    )
