"""Entrena LinearRegression (antiguedad -> PCI) y guarda el modelo.
Ejecutar:  python -m src.entrenar
"""
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

DATOS = "data/pavimento_limpio.csv"
MODELO = "models/modelo_pci.joblib"


def cargar_datos():
    return pd.read_csv(DATOS, parse_dates=["fecha"])


def dividir(df, prop_prueba=0.2):
    """Division cronologica: 80% mas antiguo = entrenamiento, 20% mas reciente = prueba."""
    df = df.sort_values("fecha").reset_index(drop=True)
    corte = int(len(df) * (1 - prop_prueba))
    return df.iloc[:corte], df.iloc[corte:]


if __name__ == "__main__":
    df = cargar_datos()
    train, _ = dividir(df)
    modelo = LinearRegression().fit(train[["antiguedad"]], train["pci_observado"])
    info = {"modelo": modelo, "edad_max_entrenamiento": int(train["antiguedad"].max())}
    joblib.dump(info, MODELO)
    print(f"Pendiente: {modelo.coef_[0]:.3f} | Intercepto: {modelo.intercept_:.3f}")
    print(f"Modelo guardado en {MODELO}")
