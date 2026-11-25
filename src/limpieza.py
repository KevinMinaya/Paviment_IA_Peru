"""Limpia el dataset: duplicados, nulos, tipos y rangos validos.
Ejecutar:  python -m src.limpieza
"""
import pandas as pd

ENTRADA = "data/pavimento_inspecciones.csv"
SALIDA = "data/pavimento_limpio.csv"


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["antiguedad"] = pd.to_numeric(df["antiguedad"], errors="coerce")
    df["pci_observado"] = pd.to_numeric(df["pci_observado"], errors="coerce")
    df = df.dropna(subset=["fecha", "antiguedad", "pci_observado", "tramo"])
    df = df[(df["pci_observado"].between(0, 100)) & (df["antiguedad"] >= 0)]
    return df.sort_values("fecha").reset_index(drop=True)


if __name__ == "__main__":
    bruto = pd.read_csv(ENTRADA)
    limpio = limpiar(bruto)
    limpio.to_csv(SALIDA, index=False)
    print(f"Registros: {len(bruto)} -> {len(limpio)} despues de limpiar. Guardado en {SALIDA}")
