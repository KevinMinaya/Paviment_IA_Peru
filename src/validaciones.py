"""Validacion de entradas del usuario. Cada funcion devuelve una lista de errores."""
from datetime import date

TIPOS = ["Grietas longitudinales", "Grietas transversales", "Grietas de esquina",
         "Losa dividida", "Despostillamiento", "Escalonamiento"]
SEVERIDADES = ["Baja", "Media", "Alta"]


def validar_prediccion(antiguedad, horizonte):
    errores = []
    if antiguedad is None or not (0 <= antiguedad <= 50):
        errores.append("La antiguedad debe estar entre 0 y 50 anios.")
    if horizonte is None or not (0 <= horizonte <= 20):
        errores.append("El horizonte debe estar entre 0 y 20 anios.")
    return errores


def validar_patologia(tramo, tipo, severidad, fecha):
    errores = []
    if not tramo or not tramo.strip():
        errores.append("El tramo no puede estar vacio.")
    if tipo not in TIPOS:
        errores.append("Tipo de patologia no valido.")
    if severidad not in SEVERIDADES:
        errores.append("Severidad no valida.")
    if fecha is None or fecha > date.today():
        errores.append("La fecha no puede ser futura.")
    return errores
