"""Genera el dataset SIMULADO de inspecciones (60 registros).
Ejecutar desde la raiz:  python -m src.generar_datos
"""
import numpy as np
import pandas as pd
from pathlib import Path

RUTA = Path("data/pavimento_inspecciones.csv")


def generar(n=60, semilla=42):
    rng = np.random.default_rng(semilla)
    tramos = [f"HCO-{i:02d}" for i in range(1, 13)]
    fallas = ["Grietas longitudinales", "Grietas transversales", "Grietas de esquina",
              "Losa dividida", "Despostillamiento", "Escalonamiento"]
    severidades = ["Baja", "Media", "Alta"]
    transitos = ["Bajo", "Medio", "Alto"]

    antig = rng.integers(1, 11, n)
    transito = rng.choice(transitos, n)
    # PCI simulado: baja ~8.5 puntos por anio + ruido (datos FICTICIOS)
    pci = 103 - 8.5 * antig + rng.normal(0, 3.5, n)
    pci = np.clip(pci, 0, 100).round(1)
    fechas = pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D")

    return pd.DataFrame({
        "tramo": rng.choice(tramos, n),
        "fecha": fechas.strftime("%Y-%m-%d"),
        "antiguedad": antig,
        "transito": transito,
        "tipo_falla": rng.choice(fallas, n),
        "severidad": rng.choice(severidades, n),
        "pci_observado": pci,
        "origen": "simulado",
    })


if __name__ == "__main__":
    RUTA.parent.mkdir(exist_ok=True)
    df = generar()
    df.to_csv(RUTA, index=False)
    print(f"Dataset simulado guardado en {RUTA} ({len(df)} registros)")
