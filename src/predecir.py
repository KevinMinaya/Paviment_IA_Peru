"""Carga el modelo entrenado y predice el PCI actual y futuro.
Se usa desde app.py:  from src.predecir import predecir
"""
import joblib
import pandas as pd

MODELO = "models/modelo_pci.joblib"
CONDICIONES = [(85, "Bueno"), (70, "Satisfactorio"), (55, "Regular"),
               (40, "Pobre"), (25, "Muy pobre"), (10, "Grave")]


def cargar_modelo(ruta=MODELO):
    return joblib.load(ruta)


def clasificar(pci):
    for limite, nombre in CONDICIONES:
        if pci > limite:
            return nombre
    return "Fallado"


def _pci(modelo, edad):
    valor = float(modelo.predict(pd.DataFrame({"antiguedad": [edad]}))[0])
    return max(0.0, min(100.0, valor))


def predecir(antiguedad, horizonte, info=None):
    info = info or cargar_modelo()
    modelo = info["modelo"]
    futura = antiguedad + horizonte
    pci_actual, pci_futuro = _pci(modelo, antiguedad), _pci(modelo, futura)
    return {
        "pci_actual": round(pci_actual, 1),
        "pci_futuro": round(pci_futuro, 1),
        "condicion_futura": clasificar(pci_futuro),
        "antiguedad_futura": futura,
        "extrapolacion": futura > info["edad_max_entrenamiento"],
        "pendiente": float(modelo.coef_[0]),
        "intercepto": float(modelo.intercept_),
    }
