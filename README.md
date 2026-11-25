# PAVEMENT AI - SENATI

Prediccion del PCI de pavimento rigido (Huanuco) con regresion lineal.

## Instalacion
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Pipeline (desde la carpeta raiz)
```
python -m src.generar_datos
python -m src.limpieza
python -m src.entrenar
python -m src.evaluar
python -m streamlit run app.py
```

## Limitaciones
- Datos SIMULADOS (columna `origen`).
- Solo usa la antiguedad como variable; el PCI real depende de tipo/severidad de fallas, transito, drenaje, etc.
- Extrapolar fuera del rango de entrenamiento no es confiable.
- No reemplaza la metodologia PCI ni una inspeccion estructural.
