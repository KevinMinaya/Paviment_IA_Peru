"""Analisis visual preliminar de grietas mediante contraste local."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def analizar_imagen(contenido):
    """Devuelve imagen original, superposicion roja y metricas aproximadas.

    Es una heuristica de contraste, no un modelo entrenado ni una evaluacion PCI.
    """
    original = Image.open(BytesIO(contenido)).convert("RGB")
    # Limita el costo para imagenes muy grandes sin perder detalle practico.
    original.thumbnail((1400, 1400))
    gris_img = original.convert("L")
    gris = np.asarray(gris_img, dtype=np.uint8)
    fondo = np.asarray(gris_img.filter(ImageFilter.GaussianBlur(radius=7)), dtype=np.int16)
    diferencia = fondo - gris.astype(np.int16)

    # Las grietas suelen ser lineas oscuras frente al concreto que las rodea.
    mascara = (diferencia >= 18) & (gris < 190)
    etiquetas, cantidad = ndimage.label(mascara)
    tamanos = np.bincount(etiquetas.ravel())
    validas = tamanos >= 12
    validas[0] = False
    mascara = validas[etiquetas]
    componentes = int(validas.sum())
    porcentaje = float(mascara.mean() * 100)

    if porcentaje >= 5:
        severidad = "Alta"
    elif porcentaje >= 1:
        severidad = "Media"
    else:
        severidad = "Baja"

    rgb = np.asarray(original, dtype=np.uint8).copy()
    rojo = np.zeros_like(rgb)
    rojo[..., 0] = 255
    rgb[mascara] = (rgb[mascara].astype(np.float32) * 0.35 + rojo[mascara] * 0.65).astype(np.uint8)
    return {
        "original": original,
        "superposicion": Image.fromarray(rgb),
        "porcentaje": round(porcentaje, 2),
        "cantidad": componentes,
        "severidad": severidad,
    }
