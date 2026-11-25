"""Aplicacion Streamlit PAVEMENT AI.  Ejecutar:  python -m streamlit run app.py"""
import os
from datetime import datetime
from datetime import date
import pandas as pd
import streamlit as st

from src.entrenar import cargar_datos
from src.deteccion_imagen import analizar_imagen
from src.evaluar import evaluar
from src.predecir import cargar_modelo, predecir
from src.validaciones import validar_prediccion, validar_patologia, TIPOS, SEVERIDADES
from src.graficos import (
    grafico_regresion,
    grafico_patologias,
    grafico_patologias_dataset,
    grafico_condiciones,
    grafico_escenarios,
)

REGISTRO = "data/registro_patologias.csv"
FOTOS = "data/fotos"
COLUMNAS = ["tramo", "fecha", "tipo_falla", "severidad", "foto"]

st.set_page_config(page_title="PAVEMENT AI", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --ink: #172b3a;
    --muted: #637887;
    --teal: #087e78;
    --teal-dark: #075e5a;
    --line: #dce7e9;
    --paper: #f3f7f7;
}

.stApp { background: var(--paper); color: var(--ink); font-family: 'DM Sans', sans-serif; }
[data-testid="stHeader"] { background: rgba(243, 247, 247, .88); }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }

.hero {
    position: relative; overflow: hidden; padding: 2rem 2.25rem;
    margin: 0 0 1.5rem; border-radius: 22px;
    color: white; background: linear-gradient(118deg, #103548 0%, #087e78 72%, #18a18d 100%);
    box-shadow: 0 14px 34px rgba(13, 72, 78, .16);
}
.hero:after { content: ''; position: absolute; width: 270px; height: 270px; border: 1px solid rgba(255,255,255,.16); border-radius: 50%; right: -65px; top: -145px; box-shadow: 0 0 0 28px rgba(255,255,255,.045), 0 0 0 58px rgba(255,255,255,.035); }
.hero-kicker { color: #a8ece0; font-size: .75rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { color: white; margin: .4rem 0 .35rem; font-family: 'Manrope', sans-serif; font-size: clamp(1.7rem, 3vw, 2.55rem); letter-spacing: -.04em; }
.hero p { color: rgba(255,255,255,.82); margin: 0; font-size: 1rem; }
.notice { margin: 0 0 1.4rem; padding: .8rem 1rem; border: 1px solid #f0d9a8; border-radius: 12px; background: #fff8e9; color: #73551e; font-size: .88rem; }

[data-testid="stTabs"] [role="tablist"] { gap: .45rem; border-bottom: 1px solid var(--line); }
[data-testid="stTabs"] button[role="tab"] { padding: .75rem 1rem; border-radius: 10px 10px 0 0; color: var(--muted); font-weight: 600; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal-dark); background: #e4f2ef; }
[data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 16px; padding: 1.1rem 1.2rem; box-shadow: 0 5px 16px rgba(22, 58, 69, .045); }
[data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--teal-dark); font-family: 'Manrope', sans-serif; }
[data-testid="stForm"] { padding: 1.25rem 1.4rem; border: 1px solid var(--line); border-radius: 16px; background: white; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.stButton > button, [data-testid="stFormSubmitButton"] > button, .stDownloadButton > button { border: 0; border-radius: 10px; color: white; background: var(--teal); font-weight: 700; transition: .18s ease; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover, .stDownloadButton > button:hover { color: white; background: var(--teal-dark); border: 0; transform: translateY(-1px); }
input, textarea, [data-baseweb="select"] > div { border-radius: 9px !important; }
h2, h3 { color: var(--ink); font-family: 'Manrope', sans-serif; letter-spacing: -.025em; }
@media (max-width: 700px) { .block-container { padding: 1rem 1rem 2rem; } .hero { padding: 1.5rem; border-radius: 16px; } }
</style>
<header class="hero">
  <div class="hero-kicker">Pavement AI · Huánuco</div>
  <h1>Monitoreo del pavimento rígido</h1>
  <p>Analiza el estado PCI, registra patologías y explora escenarios de deterioro.</p>
</header>
<div class="notice"><strong>Nota:</strong> los datos de entrenamiento son simulados. Esta herramienta no reemplaza una inspección técnica PCI.</div>
""", unsafe_allow_html=True)

if not os.path.exists("models/modelo_pci.joblib"):
    st.error("Falta el modelo. Ejecute: python -m src.entrenar")
    st.stop()


@st.cache_resource
def modelo():
    return cargar_modelo()


def leer_registro():
    if os.path.exists(REGISTRO):
        return pd.read_csv(REGISTRO)
    return pd.DataFrame(columns=COLUMNAS)


info = modelo()
datos = cargar_datos()
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dashboard", "Registro de patologias", "Simulador PCI", "Escenarios", "Evaluar foto"]
)

with tab1:
    m = evaluar()
    graf1, graf2 = st.columns(2)
    with graf1:
        st.plotly_chart(
            grafico_patologias_dataset(datos),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with graf2:
        st.plotly_chart(
            grafico_condiciones(datos),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2 (prueba)", m["R2"])
    c2.metric("MAE", m["MAE"])
    c3.metric("RMSE", m["RMSE"])
    c4.metric("Pendiente (PCI/anio)", f"{info['modelo'].coef_[0]:.2f}")
    st.plotly_chart(grafico_regresion(datos, info), use_container_width=True)
    reg = leer_registro()
    if not reg.empty:
        st.plotly_chart(grafico_patologias(reg), use_container_width=True)
        st.download_button("Exportar registro (CSV)", reg.to_csv(index=False), "registro_patologias.csv")

with tab2:
    with st.form("form_patologia", clear_on_submit=True):
        tramo = st.text_input("Tramo (ej. HCO-01)")
        fecha = st.date_input("Fecha de inspeccion", value=date.today())
        tipo = st.selectbox("Tipo de patologia", TIPOS)
        sev = st.selectbox("Severidad", SEVERIDADES)
        foto = st.file_uploader("Fotografia", type=["jpg", "jpeg", "png"])
        enviar = st.form_submit_button("Guardar patologia")
    if enviar:
        errores = validar_patologia(tramo, tipo, sev, fecha)
        if errores:
            for e in errores:
                st.error(e)
        else:
            nombre = ""
            if foto is not None:
                os.makedirs(FOTOS, exist_ok=True)
                nombre = f"{tramo.strip()}_{fecha}_{foto.name}"
                with open(os.path.join(FOTOS, nombre), "wb") as f:
                    f.write(foto.getbuffer())
            fila = pd.DataFrame([[tramo.strip(), fecha, tipo, sev, nombre]], columns=COLUMNAS)
            fila.to_csv(REGISTRO, mode="a", header=not os.path.exists(REGISTRO), index=False)
            st.success("Patologia registrada.")
    st.dataframe(leer_registro(), use_container_width=True)

with tab3:
    a = st.number_input("Antiguedad actual (anios)", 0.0, 50.0, 5.0, 1.0)
    h = st.number_input("Horizonte (anios)", 0.0, 20.0, 3.0, 1.0)
    if st.button("Ejecutar prediccion"):
        errores = validar_prediccion(a, h)
        if errores:
            for e in errores:
                st.error(e)
        else:
            r = predecir(a, h, info)
            c1, c2, c3 = st.columns(3)
            c1.metric("PCI actual", r["pci_actual"])
            c2.metric("PCI futuro", r["pci_futuro"])
            c3.metric("Condicion futura", r["condicion_futura"])
            st.write(f"PCI = {r['intercepto']:.2f} + ({r['pendiente']:.2f}) x antiguedad")
            if r["extrapolacion"]:
                st.warning("Extrapolacion: la antiguedad futura supera el rango de los datos de entrenamiento.")
            st.plotly_chart(grafico_regresion(datos, info, (r["antiguedad_futura"], r["pci_futuro"])),
                            use_container_width=True)

with tab4:
    st.write("Tres escenarios simulados con el modelo entrenado.")
    esc = [("Optimista", 3, 2), ("Base", 5, 3), ("Pesimista", 8, 5)]
    filas = []
    for nombre, ed, hor in esc:
        r = predecir(ed, hor, info)
        filas.append({"escenario": nombre, "antiguedad": ed, "horizonte": hor,
                      "pci_futuro": r["pci_futuro"], "condicion_futura": r["condicion_futura"],
                      "extrapolacion": r["extrapolacion"]})
    df_esc = pd.DataFrame(filas)
    st.dataframe(df_esc, use_container_width=True)
    st.plotly_chart(grafico_escenarios(df_esc), use_container_width=True)

with tab5:
    st.subheader("Deteccion de fallas en imagenes")
    st.caption("Carga una foto del pavimento para ver las zonas oscuras resaltadas y registrar la inspeccion.")
    st.warning(
        "La zona roja se calcula con una regla visual de contraste. Es una estimacion preliminar, "
        "no un modelo entrenado ni un calculo PCI oficial. Sombras, manchas y juntas pueden marcarse como fallas."
    )
    imagen = st.file_uploader(
        "Subir imagen del pavimento", type=["jpg", "jpeg", "png"], key="foto_condicion"
    )
    if imagen is not None:
        analisis = analizar_imagen(imagen.getvalue())
        foto_original, foto_marcada = st.columns(2)
        with foto_original:
            st.image(analisis["original"], caption="Imagen original", use_container_width=True)
        with foto_marcada:
            st.image(analisis["superposicion"], caption="Zonas resaltadas en rojo", use_container_width=True)

        metrica1, metrica2, metrica3 = st.columns(3)
        metrica1.metric("Area resaltada", f"{analisis['porcentaje']:.2f}%")
        metrica2.metric("Regiones detectadas", analisis["cantidad"])
        metrica3.metric("Severidad estimada", analisis["severidad"])

        st.markdown("#### Registrar patologia")
        with st.form("registro_foto_deteccion", clear_on_submit=False):
            tramo = st.text_input("Tramo (ej. HCO-01)")
            fecha = st.date_input("Fecha de inspeccion", value=date.today(), key="fecha_foto_deteccion")
            tipo = st.selectbox("Tipo de patologia", TIPOS, key="tipo_foto_deteccion")
            severidad = st.selectbox(
                "Severidad observada", SEVERIDADES,
                index=SEVERIDADES.index(analisis["severidad"]), key="sev_foto_deteccion",
            )
            guardar = st.form_submit_button("Guardar inspeccion")

        if guardar:
            errores = validar_patologia(tramo, tipo, severidad, fecha)
            if errores:
                for error in errores:
                    st.error(error)
            else:
                os.makedirs(FOTOS, exist_ok=True)
                nombre_seguro = os.path.basename(imagen.name).replace("\\", "_").replace("/", "_")
                nombre = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{nombre_seguro}"
                with open(os.path.join(FOTOS, nombre), "wb") as archivo:
                    archivo.write(imagen.getvalue())
                fila = pd.DataFrame(
                    [[tramo.strip(), fecha, tipo, severidad, nombre]], columns=COLUMNAS
                )
                fila.to_csv(REGISTRO, mode="a", header=not os.path.exists(REGISTRO), index=False)
                st.success("Inspeccion guardada con la fotografia.")

    st.markdown("#### Registro de patologias")
    registro = leer_registro()
    st.dataframe(registro, use_container_width=True)
