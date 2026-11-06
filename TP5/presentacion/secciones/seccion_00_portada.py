"""Seccion 00: diapositiva de titulo y diapositiva Resumen.

Ninguna cifra se escribe a mano: todas salen de los reportes impresos que
regenera scripts/regenerar.sh, leidos con lector.Salidas.
"""

from __future__ import annotations

from lector import campo, numero
from pptx_minimo import CONTENIDO_Y_IN, MARGEN_IN, ANCHO_IN, Parrafo

TITULO_PORTADA = "TP5: autoencoders (básico, denoising y variacional)"


def _pixeles(n: str) -> str:
    return "píxel" if n == "1" else "píxeles"


def agregar(pres, salidas) -> None:
    # --- cifras tomadas de los reportes regenerados -----------------------
    l1a = salidas.linea("ej1a", "salida-ej1a.txt", "RESULTADO:")
    ok_1a = numero(l1a, r"RESULTADO: (\d+)/\d+ patrones")
    total_1a = numero(l1a, r"RESULTADO: \d+/(\d+) patrones")
    tolerancia = numero(l1a, r"a lo sumo (\d+) pixel")
    maximo_1a = numero(l1a, r"\(maximo (\d+)\)")
    cfg_1a = salidas.linea("ej1a", "salida-ej1a.txt", "configuracion elegida:")
    arq_1a = campo(cfg_1a, "arquitectura")

    l1b = salidas.linea("ej1b", "salida-ej1b.txt", "NIVEL MAXIMO ELIMINADO:")
    nivel_1b = numero(l1b, r"NIVEL MAXIMO ELIMINADO: (\d+\.\d+)")
    porcentaje_1b = numero(l1b, r"al menos el (\d+) %")
    cfg_1b = salidas.linea("ej1b", "salida-ej1b.txt", "configuracion:")
    arq_1b = campo(cfg_1b, "arquitectura")
    nombre_1b = campo(cfg_1b, "configuracion")

    l2 = salidas.linea("ej2", "salida-ej2.txt", "RESULTADO EJ2:")
    estado_2 = numero(l2, r"RESULTADO EJ2: (\w+)")
    ok_2 = numero(l2, r"\((\d+)/\d+ comprobaciones")
    total_2 = numero(l2, r"\(\d+/(\d+) comprobaciones")
    dist_2 = numero(l2, r"emoji nuevo a (\d+) px")
    cercano_2 = numero(l2, r"px de ([\w-]+)\)")
    cfg_2 = salidas.linea("ej2", "salida-ej2.txt", "configuracion:")
    nombre_2 = campo(cfg_2, "configuracion")
    decodificador_2 = numero(cfg_2, r"decoder (\S+)")
    n_emojis = numero(salidas.linea("ej2", "salida-ej2.txt", "emojis:"), r"emojis: (\d+) imagenes")

    # --- diapositiva de titulo ---------------------------------------------
    d = pres.diapositiva(TITULO_PORTADA, portada=True)
    d.texto(0.9, 4.2, 11.5, 0.6, "Sistemas de Inteligencia Artificial, ITBA", tam=24)
    d.texto(
        0.9, 4.9, 11.5, 0.5,
        "Código y resultados reproducibles desde TP5 con scripts/regenerar.sh",
        tam=16,
    )

    # --- Resumen ---------------------------------------------------------------
    d = pres.diapositiva("Resumen: tres ejercicios, tres resultados")
    hueco = 0.4
    ancho = (ANCHO_IN - 2 * MARGEN_IN - 2 * hueco) / 3
    y_cifra, alto_cifra = CONTENIDO_Y_IN, 1.9
    y_texto, alto_texto = y_cifra + alto_cifra + 0.25, 3.0

    columnas = [
        (
            f"{ok_1a}/{total_1a}",
            f"patrones con a lo sumo {tolerancia} {_pixeles(tolerancia)} incorrecto",
            f"Ej 1a: autoencoder básico {arq_1a} con cuello de botella bidimensional; "
            f"el peor patrón queda con {maximo_1a} {_pixeles(maximo_1a)} incorrectos.",
        ),
        (
            nivel_1b,
            "nivel de ruido máximo eliminado (criterio del trabajo)",
            f"Ej 1b: autoencoder denoising {arq_1b} ({nombre_1b}); el criterio exige que al menos "
            f"el {porcentaje_1b} % de las distorsiones no vistas quede con a lo sumo "
            f"{tolerancia} {_pixeles(tolerancia)} incorrecto.",
        ),
        (
            f"{ok_2}/{total_2}",
            "comprobaciones de aceptación del VAE",
            f"Ej 2: autoencoder variacional {nombre_2} sobre {n_emojis} emojis, decodificador "
            f"{decodificador_2}; resultado {estado_2}, con un emoji nuevo a {dist_2} px de {cercano_2}.",
        ),
    ]
    for i, (valor, etiqueta, frase) in enumerate(columnas):
        x = MARGEN_IN + i * (ancho + hueco)
        d.cifra(x, y_cifra, ancho, alto_cifra, valor, etiqueta)
        d.texto(x, y_texto, ancho, alto_texto, [Parrafo(frase)], tam=16)
