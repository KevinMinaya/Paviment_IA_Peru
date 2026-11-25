"""Evalua el modelo en el conjunto de prueba: R2, MAE y RMSE.
Ejecutar:  python -m src.evaluar
"""
import json
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.entrenar import cargar_datos, dividir, MODELO


def evaluar():
    info = joblib.load(MODELO)
    _, test = dividir(cargar_datos())
    pred = info["modelo"].predict(test[["antiguedad"]])
    return {
        "R2": round(float(r2_score(test["pci_observado"], pred)), 4),
        "MAE": round(float(mean_absolute_error(test["pci_observado"], pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(test["pci_observado"], pred))), 3),
        "n_prueba": int(len(test)),
    }


if __name__ == "__main__":
    m = evaluar()
    print(m)
    with open("models/metricas.json", "w") as f:
        json.dump(m, f, indent=2)
