"""Graficos con Plotly. Cada funcion devuelve una figura para st.plotly_chart."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def grafico_patologias_dataset(df):
    """Cuenta las patologias del dataset de inspecciones."""
    cuenta = df["tipo_falla"].value_counts().rename_axis("Patologia").reset_index(name="Cantidad")
    fig = px.bar(cuenta, x="Patologia", y="Cantidad", text="Cantidad",
                 color_discrete_sequence=["#22aee8"])
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Patologias en el dataset", xaxis_title="Patologia", yaxis_title="Cantidad",
                      xaxis_tickangle=-32, showlegend=False, margin=dict(t=25, b=70))
    return fig


def grafico_condiciones(df):
    """Distribuye los tramos por condicion PCI observada."""
    limites = [(85, "Bueno"), (70, "Satisfactorio"), (55, "Regular"),
               (40, "Pobre"), (25, "Muy pobre"), (10, "Grave")]

    def condicion(pci):
        for limite, nombre in limites:
            if pci > limite:
                return nombre
        return "Fallado"

    conteo = df["pci_observado"].map(condicion).value_counts()
    orden = ["Muy pobre", "Regular", "Pobre", "Satisfactorio", "Bueno", "Grave", "Fallado"]
    resumen = conteo.reindex(orden, fill_value=0).rename_axis("Condicion").reset_index(name="Tramos")
    resumen = resumen[resumen["Tramos"] > 0]
    colores = {"Muy pobre": "#22aee8", "Regular": "#f5a623", "Pobre": "#f45468",
               "Satisfactorio": "#30c99a", "Bueno": "#9b7bea", "Grave": "#ffbf24",
               "Fallado": "#52616b"}
    fig = px.pie(resumen, names="Condicion", values="Tramos", hole=0,
                 color="Condicion", color_discrete_map=colores)
    fig.update_traces(textposition="inside", textinfo="percent", hovertemplate="%{label}<br>%{value} tramos<extra></extra>")
    fig.update_layout(title="Condicion de los tramos", legend_title_text="",
                      margin=dict(t=45, b=20, l=10, r=10))
    return fig


def grafico_regresion(df, info, punto=None):
    modelo = info["modelo"]
    xs = np.arange(0, 16)
    ys = np.clip(modelo.intercept_ + modelo.coef_[0] * xs, 0, 100)
    fig = go.Figure()
    fig.add_scatter(x=df["antiguedad"], y=df["pci_observado"], mode="markers", name="Observado")
    fig.add_scatter(x=xs, y=ys, mode="lines", name="Regresion lineal", line=dict(dash="dash"))
    if punto:
        fig.add_scatter(x=[punto[0]], y=[punto[1]], mode="markers", name="Prediccion",
                        marker=dict(size=14, color="red"))
    fig.update_layout(xaxis_title="Antiguedad (anios)", yaxis_title="PCI", yaxis_range=[0, 105])
    return fig


def grafico_patologias(df_reg):
    cuenta = df_reg.groupby(["tipo_falla", "severidad"]).size().reset_index(name="cantidad")
    return px.bar(cuenta, x="tipo_falla", y="cantidad", color="severidad", barmode="stack",
                  labels={"tipo_falla": "Patologia"})


def grafico_escenarios(df_esc):
    return px.bar(df_esc, x="escenario", y="pci_futuro", color="condicion_futura",
                  range_y=[0, 100], labels={"pci_futuro": "PCI futuro"})
