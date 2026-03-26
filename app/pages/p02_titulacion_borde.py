# =============================================================================
# p02_titulacion.py
# Pestaña 2 — Análisis por titulación
#
# QUÉ HACE:
#   Vista detallada del abandono en una titulación concreta.
#   El usuario elige la titulación → ve métricas, distribución de riesgo,
#   factores más influyentes (SHAP), y tabla de alumnos de riesgo alto.
#
# PERFIL DE USUARIO:
#   Profesores, coordinadores de titulación y tribunal evaluador del TFM.
#   El tribunal mirará esta pestaña en profundidad — explicativa y técnica.
#
# REQUISITOS:
#   - meta_test.parquet con prob_abandono (generado en f6_m00_preparacion)
#   - shap_global.pkl (valores SHAP del modelo CatBoost)
#   - pipeline_preprocesamiento.pkl y modelo CatBoost cargados vía loaders
#
# GENERA:
#   Visualización interactiva en Streamlit (sin ficheros de salida)
#
# FLUJO:
#   loaders.py → cargar_datos_app() → df con prob_abandono, titulacion, rama
#   Selector titulación → filtrar df → KPIs → gráficos → tabla
#
# SIGUIENTE: p03_prospecto.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ROOT robusto — igual que en todos los notebooks del proyecto
# ---------------------------------------------------------------------------
def _detectar_root() -> Path:
    """Sube niveles desde este fichero hasta encontrar src/."""
    ruta = Path(__file__).resolve()
    for padre in ruta.parents:
        if (padre / "src").exists():
            return padre
    raise FileNotFoundError(
        f"No se encontró src/ subiendo desde {ruta}. "
        "Asegúrate de que el proyecto tiene la carpeta src/ en la raíz."
    )

ROOT = _detectar_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_app import (
    COLORES, COLORES_RAMAS, RAMAS_NOMBRES, UMBRALES, NOMBRES_VARIABLES
)
from utils.loaders import cargar_meta_test_app, cargar_modelo, cargar_pipeline


# =============================================================================
# CONSTANTES LOCALES
# =============================================================================

# Columnas que son metadatos — NO son features del modelo
# Se usan para filtrar antes de pasar datos al pipeline
_COLS_META = {
    "abandono", "titulacion", "rama", "rama_meta", "anio_cohorte",
    "sexo", "nivel_riesgo", "prob_abandono", "per_id_ficticio",
    "cupo", "curso_aca_ini", "flag_cautela", "n_titulaciones",
    "vive_fuera", "pais_nombre", "provincia", "via_acceso"
}

# Colores para niveles de riesgo
_COLOR_BAJO   = "#27AE60"   # verde
_COLOR_MEDIO  = "#F39C12"   # naranja
_COLOR_ALTO   = "#E53E3E"   # rojo abandono (paleta proyecto)

# Número de alumnos a mostrar en la tabla de riesgo alto
_MAX_TABLA = 50


# =============================================================================
# CARGA Y PREPARACIÓN
# =============================================================================

@st.cache_data(show_spinner=False)
def _cargar_y_preparar() -> pd.DataFrame:
    """
    Carga los datos de la app y añade columna nivel_riesgo si no existe.
    Cachea el resultado para no recargar en cada interacción.
    """
              # dict con 'df', 'modelo', 'pipeline', etc.
    df    = cargar_meta_test_app().copy()

    # Calcular prob_abandono si no existe — aplicar modelo sobre features
    if "prob_abandono" not in df.columns:
        modelo   = cargar_modelo()
        pipeline = cargar_pipeline()
        cols_features = [c for c in pipeline.feature_names_in_ if c in df.columns]
        X = df[cols_features]
        df["prob_abandono"] = modelo.predict_proba(X)[:, 1]

    # Añadir nivel de riesgo categórico si no existe
    if "nivel_riesgo" not in df.columns and "prob_abandono" in df.columns:
        umbral_alto  = UMBRALES["riesgo_medio"]   # ≥ 0.60 → alto
        umbral_medio = UMBRALES["riesgo_bajo"]     # ≥ 0.30 → medio

        condiciones = [
            df["prob_abandono"] >= umbral_alto,
            df["prob_abandono"] >= umbral_medio,
        ]
        df["nivel_riesgo"] = np.select(
            condiciones,
            ["Alto", "Medio"],
            default="Bajo"
        )

    return df


def _lista_titulaciones(df: pd.DataFrame) -> list[str]:
    """
    Devuelve la lista de titulaciones ordenadas: primero por rama,
    luego alfabéticamente dentro de cada rama.
    """
    if "titulacion" not in df.columns:
        return []
    return (
        df[["titulacion", "rama_meta"]]
        .drop_duplicates()
        .sort_values(["rama_meta", "titulacion"])["titulacion"]
        .tolist()
    )


# =============================================================================
# BLOQUES DE CONTENIDO
# =============================================================================

def _bloque_kpis(df_tit: pd.DataFrame):
    """
    Fila de KPIs rápidos para la titulación seleccionada.
    5 métricas: total, tasa real, tasa predicha, riesgo alto, F1 del modelo.
    """
    n_total        = len(df_tit)
    n_abandono_real = df_tit["abandono"].sum() if "abandono" in df_tit.columns else None
    tasa_real       = (n_abandono_real / n_total * 100) if n_abandono_real is not None else None
    tasa_predicha   = df_tit["prob_abandono"].mean() * 100 if "prob_abandono" in df_tit.columns else None
    n_riesgo_alto   = (df_tit["nivel_riesgo"] == "Alto").sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Alumnos en test", f"{n_total:,}")
    with col2:
        if tasa_real is not None:
            st.metric(
                "Tasa abandono real",
                f"{tasa_real:.1f}%",
                help="Porcentaje de alumnos que realmente abandonaron en el conjunto de test."
            )
        else:
            st.metric("Tasa abandono real", "N/D")
    with col3:
        if tasa_predicha is not None:
            delta_val = tasa_predicha - tasa_real if tasa_real is not None else None
            delta_str = f"{delta_val:+.1f}pp" if delta_val is not None else None
            st.metric(
                "Riesgo medio predicho",
                f"{tasa_predicha:.1f}%",
                delta=delta_str,
                delta_color="inverse",
                help="Probabilidad media de abandono según el modelo."
            )
        else:
            st.metric("Riesgo medio predicho", "N/D")
    with col4:
        pct_alto = n_riesgo_alto / n_total * 100 if n_total > 0 else 0
        st.metric(
            "Alumnos en riesgo alto",
            f"{n_riesgo_alto:,}",
            delta=f"{pct_alto:.1f}% del total",
            delta_color="off",
            help=f"Alumnos con probabilidad de abandono ≥ {UMBRALES['riesgo_medio']:.0%}."
        )
    with col5:
        # F1 global del modelo — constante, no depende de la titulación
        st.metric(
            "F1 modelo (global)",
            "0.799",
            help="F1-score del modelo Stacking sobre el conjunto de test completo."
        )


def _bloque_distribucion_riesgo(df_tit: pd.DataFrame, nombre_tit: str):
    """
    Donut de distribución de riesgo + histograma de probabilidades.
    Dos gráficos en columnas para visión compacta.
    """
    col_izq, col_der = st.columns([1, 1.6])

    # --- Donut ---
    with col_izq:
        st.subheader("Distribución del riesgo")

        conteo = df_tit["nivel_riesgo"].value_counts().reindex(
            ["Bajo", "Medio", "Alto"], fill_value=0
        ).reset_index()
        conteo.columns = ["nivel", "n"]

        fig_donut = go.Figure(go.Pie(
            labels=conteo["nivel"],
            values=conteo["n"],
            hole=0.55,
            marker_colors=[_COLOR_BAJO, _COLOR_MEDIO, _COLOR_ALTO],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} alumnos (%{percent})<extra></extra>",
            sort=False
        ))
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # --- Histograma ---
    with col_der:
        st.subheader("Probabilidad de abandono")

        if "prob_abandono" in df_tit.columns:
            fig_hist = px.histogram(
                df_tit,
                x="prob_abandono",
                nbins=30,
                color="nivel_riesgo",
                color_discrete_map={
                    "Bajo": _COLOR_BAJO,
                    "Medio": _COLOR_MEDIO,
                    "Alto": _COLOR_ALTO
                },
                category_orders={"nivel_riesgo": ["Bajo", "Medio", "Alto"]},
                labels={
                    "prob_abandono": "Probabilidad de abandono",
                    "nivel_riesgo": "Nivel de riesgo"
                }
            )
            # Líneas verticales de umbral
            for umbral, etiqueta, color in [
                (UMBRALES["riesgo_bajo"],  "Umbral bajo",  _COLOR_MEDIO),
                (UMBRALES["riesgo_medio"], "Umbral alto",  _COLOR_ALTO)
            ]:
                fig_hist.add_vline(
                    x=umbral,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=etiqueta,
                    annotation_position="top right"
                )
            fig_hist.update_layout(
                margin=dict(t=10, b=30, l=0, r=0),
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Riesgo"
            )
            st.plotly_chart(fig_hist, use_container_width=True)


def _bloque_evolucion_temporal(df_tit: pd.DataFrame):
    """
    Línea temporal: tasa de abandono real vs riesgo predicho por año de cohorte.
    Permite ver si el problema ha mejorado o empeorado en esta titulación.
    """
    st.subheader("Evolución temporal")

    col_anio = None
    for candidato in ["anio_cohorte", "curso_aca_ini"]:
        if candidato in df_tit.columns:
            col_anio = candidato
            break

    if col_anio is None or "prob_abandono" not in df_tit.columns:
        st.info("No hay datos temporales disponibles para esta titulación.")
        return

    # Agregar por año
    evol = (
        df_tit
        .groupby(col_anio)
        .agg(
            tasa_real    = ("abandono",     "mean"),
            riesgo_medio = ("prob_abandono", "mean"),
            n_alumnos    = ("prob_abandono", "count")
        )
        .reset_index()
        .rename(columns={col_anio: "anio"})
        .sort_values("anio")
    )

    fig = go.Figure()

    # Línea tasa real
    if "tasa_real" in evol.columns and evol["tasa_real"].notna().any():
        fig.add_trace(go.Scatter(
            x=evol["anio"],
            y=evol["tasa_real"] * 100,
            mode="lines+markers",
            name="Abandono real (%)",
            line=dict(color=COLORES["abandono"], width=2.5),
            marker=dict(size=7),
            hovertemplate="Año %{x}<br>Abandono real: %{y:.1f}%<extra></extra>"
        ))

    # Línea riesgo predicho
    fig.add_trace(go.Scatter(
        x=evol["anio"],
        y=evol["riesgo_medio"] * 100,
        mode="lines+markers",
        name="Riesgo predicho medio (%)",
        line=dict(color=COLORES["primario"], width=2.5, dash="dot"),
        marker=dict(size=7),
        hovertemplate="Año %{x}<br>Riesgo predicho: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Año de cohorte",
        yaxis_title="Porcentaje (%)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=30, b=30, l=0, r=0),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


def _bloque_factores_shap(df_tit: pd.DataFrame, titulacion: str):
    """
    Variables más influyentes en esta titulación.
    Usa los valores SHAP medios del subconjunto de la titulación.
    Si SHAP no está disponible, muestra diferencia de medias como proxy.
    """
    st.subheader("Factores más influyentes en esta titulación")
    st.caption(
        "Importancia media de cada variable (SHAP) para los alumnos "
        f"de {titulacion}. Barras positivas → aumentan el riesgo de abandono."
    )

    # Intentar cargar SHAP desde sesión (precargado en loaders)
    datos_shap = st.session_state.get("shap_values_catboost", None)

    if datos_shap is not None and "indices_tit" in st.session_state:
        # Usar SHAP real para los alumnos de esta titulación
        indices = st.session_state["indices_tit"].get(titulacion, None)
        if indices is not None and len(indices) > 0:
            shap_tit = datos_shap[indices]
            importancia_media = np.abs(shap_tit).mean(axis=0)
            shap_medio        = shap_tit.mean(axis=0)

            # Nombres legibles de las features
            feature_names = [
                NOMBRES_VARIABLES.get(f, f)
                for f in st.session_state.get("feature_names", range(len(importancia_media)))
            ]

            df_shap = pd.DataFrame({
                "variable":    feature_names,
                "importancia": importancia_media,
                "shap_medio":  shap_medio
            }).sort_values("importancia", ascending=True).tail(12)

            fig = go.Figure(go.Bar(
                x=df_shap["shap_medio"],
                y=df_shap["variable"],
                orientation="h",
                marker_color=[
                    _COLOR_ALTO if v > 0 else _COLOR_BAJO
                    for v in df_shap["shap_medio"]
                ],
                hovertemplate="%{y}: SHAP medio = %{x:.3f}<extra></extra>"
            ))
            fig.add_vline(x=0, line_color="gray", line_width=1)
            fig.update_layout(
                xaxis_title="Impacto SHAP medio",
                margin=dict(t=10, b=10, l=0, r=0),
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
            return

    # --- Fallback: diferencia de medias (proxy de importancia) ---
    # Funciona aunque no haya SHAP cargado en sesión
    cols_num = [
        c for c in df_tit.select_dtypes(include=[np.number]).columns
        if c not in _COLS_META and "prob" not in c
    ]

    if not cols_num:
        st.info("No hay variables numéricas disponibles para este análisis.")
        return

    if "abandono" not in df_tit.columns:
        st.info("La columna 'abandono' no está disponible en este subconjunto.")
        return

    df_abandon = df_tit[df_tit["abandono"] == 1][cols_num].mean()
    df_continu = df_tit[df_tit["abandono"] == 0][cols_num].mean()
    diferencia  = df_abandon - df_continu

    df_proxy = pd.DataFrame({
        "variable":   [NOMBRES_VARIABLES.get(c, c) for c in cols_num],
        "diferencia":  diferencia.values
    }).sort_values("diferencia")

    fig = go.Figure(go.Bar(
        x=df_proxy["diferencia"],
        y=df_proxy["variable"],
        orientation="h",
        marker_color=[
            _COLOR_ALTO if v > 0 else _COLOR_BAJO
            for v in df_proxy["diferencia"]
        ],
        hovertemplate="%{y}: diferencia = %{x:.3f}<extra></extra>"
    ))
    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_layout(
        xaxis_title="Diferencia de medias (abandono − no abandono)",
        margin=dict(t=10, b=10, l=0, r=0),
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("ℹ️ Nota técnica", expanded=False):
        st.caption(
            "Los valores SHAP no están cargados en sesión. "
            "Se muestra diferencia de medias como proxy de importancia. "
            "Ejecuta `f6_m01a_shap_global.ipynb` para activar SHAP completo."
        )


def _bloque_tabla_riesgo_alto(df_tit: pd.DataFrame):
    """
    Tabla de alumnos con riesgo alto, ordenada de mayor a menor probabilidad.
    Columnas seleccionadas para ser útiles al profesor coordinador.
    """
    st.subheader(f"Alumnos en riesgo alto (≥ {UMBRALES['riesgo_medio']:.0%})")

    df_alto = df_tit[df_tit["nivel_riesgo"] == "Alto"].copy()

    if df_alto.empty:
        st.success("✅ No hay alumnos en riesgo alto en esta titulación.")
        return

    # Columnas a mostrar — las que existan en el dataframe
    cols_mostrar_candidatas = [
        "per_id_ficticio", "prob_abandono", "nota_acceso", "nota_1er_anio",
        "n_anios_beca", "tasa_rendimiento", "situacion_laboral",
        "anio_cohorte", "sexo_meta"
    ]
    cols_mostrar = [c for c in cols_mostrar_candidatas if c in df_alto.columns]

    df_tabla = (
        df_alto[cols_mostrar]
        .sort_values("prob_abandono", ascending=False)
        .head(_MAX_TABLA)
        .rename(columns={
            c: NOMBRES_VARIABLES.get(c, c) for c in cols_mostrar
        })
    )

    # Formatear columna de probabilidad
    col_prob = NOMBRES_VARIABLES.get("prob_abandono", "prob_abandono")
    if col_prob not in df_tabla.columns:
        # Buscar por nombre original formateado
        col_prob_display = "Probabilidad de abandono"
        if col_prob_display in df_tabla.columns:
            df_tabla[col_prob_display] = df_tabla[col_prob_display].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
    else:
        df_tabla[col_prob] = df_tabla[col_prob].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )

    st.dataframe(
        df_tabla,
        use_container_width=True,
        height=min(400, 40 + len(df_tabla) * 35)
    )

    if len(df_alto) > _MAX_TABLA:
        st.caption(f"Mostrando los {_MAX_TABLA} alumnos de mayor riesgo de {len(df_alto)} totales.")


def _bloque_contexto_titulacion(df: pd.DataFrame, titulacion_sel: str, rama_tit: str):
    """
    Contexto de la titulación seleccionada dentro de su rama.
    Muestra todas las titulaciones de la misma rama ordenadas por riesgo predicho,
    resaltando la seleccionada en rojo abandono.
    """
    st.subheader("Comparativa con el resto de titulaciones")
    st.caption(
        f"Todas las titulaciones de la rama '{rama_tit}', "
        "ordenadas por riesgo predicho medio. La seleccionada aparece resaltada."
    )

    col_rama = "rama_meta" if "rama_meta" in df.columns else "rama"
    if "titulacion" not in df.columns or "prob_abandono" not in df.columns:
        return

    # Filtrar a la rama de la titulación seleccionada
    df_rama = df[df[col_rama] == rama_tit] if rama_tit and col_rama in df.columns else df

    por_tit = (
        df_rama.groupby("titulacion")["prob_abandono"]
        .mean()
        .reset_index()
        .rename(columns={"prob_abandono": "riesgo_medio"})
        .sort_values("riesgo_medio", ascending=True)
    )

    if por_tit.empty:
        return

    colores = [
        _COLOR_ALTO if t == titulacion_sel else COLORES["primario"]
        for t in por_tit["titulacion"]
    ]

    fig = go.Figure(go.Bar(
        x=por_tit["riesgo_medio"] * 100,
        y=por_tit["titulacion"],
        orientation="h",
        marker_color=colores,
        text=[f"{v*100:.1f}%" for v in por_tit["riesgo_medio"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        xaxis_title="Riesgo medio predicho (%)",
        xaxis=dict(range=[0, max(por_tit["riesgo_medio"].max() * 115, 10)]),
        margin=dict(t=10, b=30, l=0, r=60),
        height=max(200, 40 + len(por_tit) * 38),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


def _comparativa_construir_tabla(df: pd.DataFrame, titulaciones_sel: list[str], col_rama: str) -> pd.DataFrame:
    """Construye el DataFrame resumen para el modo comparativo."""
    filas = []
    for tit in titulaciones_sel:
        df_t = df[df["titulacion"] == tit]
        if df_t.empty:
            continue
        n         = len(df_t)
        tasa_real = df_t["abandono"].mean() * 100 if "abandono" in df_t.columns else None
        riesgo    = df_t["prob_abandono"].mean() * 100 if "prob_abandono" in df_t.columns else None
        n_alto    = (df_t["nivel_riesgo"] == "Alto").sum()
        pct_alto  = n_alto / n * 100 if n > 0 else 0
        rama      = df_t[col_rama].mode()[0] if col_rama in df_t.columns and not df_t.empty else "—"
        filas.append({
            "Titulación":          tit,
            "Rama":                rama,
            "Alumnos":             n,
            "Abandono real (%)":   round(tasa_real, 1) if tasa_real is not None else None,
            "Riesgo predicho (%)": round(riesgo, 1) if riesgo is not None else None,
            "En riesgo alto":      n_alto,
            "% riesgo alto":       round(pct_alto, 1),
        })
    return pd.DataFrame(filas)


# Paleta discreta para distinguir titulaciones en líneas/barras
_PALETA_COMP = [
    "#3182CE", "#E53E3E", "#38A169", "#D69E2E", "#805AD5",
    "#DD6B20", "#319795", "#E91E8C", "#2C7BB6", "#1A9850"
]


def _bloque_comparativa_titulaciones(df: pd.DataFrame, titulaciones_sel: list[str]):
    """
    Vista comparativa cuando el usuario selecciona varias titulaciones.
    Mismos 4 bloques que el modo detalle, adaptados para comparar N titulaciones.
    """
    col_rama = "rama_meta" if "rama_meta" in df.columns else "rama"
    df_comp  = _comparativa_construir_tabla(df, titulaciones_sel, col_rama)

    if df_comp.empty:
        st.warning("No hay datos para las titulaciones seleccionadas.")
        return

    # Asignar un color fijo a cada titulación para coherencia entre gráficos
    color_tit = {
        tit: _PALETA_COMP[i % len(_PALETA_COMP)]
        for i, tit in enumerate(titulaciones_sel)
    }

    # -------------------------------------------------------------------------
    # CABECERA + TABLA RESUMEN
    # -------------------------------------------------------------------------
    st.subheader(f"Comparativa de {len(titulaciones_sel)} titulaciones")
    st.caption("Haz clic en el encabezado de cualquier columna para ordenar.")
    st.dataframe(df_comp, use_container_width=True, hide_index=True,
                 height=min(450, 50 + len(df_comp) * 38))

    st.divider()

    # =========================================================================
    # BLOQUE 1 — Distribución del riesgo (barras apiladas horizontales)
    # =========================================================================
    filas_riesgo = []
    for tit in titulaciones_sel:
        df_t = df[df["titulacion"] == tit].copy()
        n    = len(df_t)
        if n == 0 or "nivel_riesgo" not in df_t.columns:
            continue
        conteo = df_t["nivel_riesgo"].value_counts().reindex(
            ["Bajo", "Medio", "Alto"], fill_value=0
        )
        for nivel, cnt in conteo.items():
            filas_riesgo.append({
                "Titulación": tit,
                "Nivel":      nivel,
                "Porcentaje": cnt / n * 100,
                "N":          int(cnt),
            })

    usar_columnas = len(titulaciones_sel) <= 2
    col_dist, col_hist = (st.columns(2) if usar_columnas else (None, None))
    if filas_riesgo:
        df_r = pd.DataFrame(filas_riesgo)
        fig_dist = go.Figure()
        def _partir_nombre(nombre, max_chars=22):
            """Parte un nombre largo en 2 líneas usando <br> por el espacio más cercano al centro."""
            if len(nombre) <= max_chars:
                return nombre
            mid = len(nombre) // 2
            # Buscar el espacio más cercano al centro
            izq = nombre.rfind(" ", 0, mid)
            der = nombre.find(" ", mid)
            if izq == -1 and der == -1:
                return nombre
            if izq == -1:
                corte = der
            elif der == -1:
                corte = izq
            else:
                corte = izq if (mid - izq) <= (der - mid) else der
            return nombre[:corte] + "<br>" + nombre[corte+1:]

        for nivel, color in [("Bajo", _COLOR_BAJO), ("Medio", _COLOR_MEDIO), ("Alto", _COLOR_ALTO)]:
            df_n = df_r[df_r["Nivel"] == nivel].copy()
            df_n["Titulación"] = df_n["Titulación"].apply(_partir_nombre)
            fig_dist.add_trace(go.Bar(
                y=df_n["Titulación"],
                x=df_n["Porcentaje"],
                name=nivel,
                orientation="h",
                marker_color=color,
                text=[f"{v:.1f}%" for v in df_n["Porcentaje"]],
                textposition="inside",
                hovertemplate="%{y} — " + nivel + ": %{x:.1f}% (%{customdata} alumnos)<extra></extra>",
                customdata=df_n["N"],
            ))
        fig_dist.update_layout(
            barmode="stack",
            xaxis_title="Porcentaje (%)",
            xaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        traceorder="normal"),
            margin=dict(t=30, b=20, l=0, r=10),
            height=min(280, 60 + len(titulaciones_sel) * 45),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        contenedor_dist = col_dist if usar_columnas else st.container()
        with contenedor_dist:
            st.subheader("Distribución por riesgo")
            st.caption("Proporción de alumnos en cada nivel de riesgo por titulación.")
            st.plotly_chart(fig_dist, use_container_width=True)

    # --- Histograma (<=2 titulaciones) o Boxplot (>2 titulaciones) ---
    if "prob_abandono" in df.columns:
        contenedor_hist = col_hist if usar_columnas else st.container()
        with contenedor_hist:
            st.subheader("Probabilidad de abandono")
            if len(titulaciones_sel) <= 2:
                # Histograma superpuesto
                fig_prob = go.Figure()
                for tit in titulaciones_sel:
                    df_t = df[df["titulacion"] == tit]
                    if df_t.empty:
                        continue
                    fig_prob.add_trace(go.Histogram(
                        x=df_t["prob_abandono"],
                        name=tit,
                        nbinsx=25,
                        marker_color=color_tit[tit],
                        opacity=0.6,
                        hovertemplate=f"{tit}<br>Probabilidad: %{{x:.2f}}<br>Alumnos: %{{y}}<extra></extra>"
                    ))
                for umbral, etiqueta, color in [
                    (UMBRALES["riesgo_bajo"],  "Umbral bajo",  _COLOR_MEDIO),
                    (UMBRALES["riesgo_medio"], "Umbral alto",  _COLOR_ALTO)
                ]:
                    fig_prob.add_vline(
                        x=umbral, line_dash="dash", line_color=color,
                        annotation_text=etiqueta, annotation_position="top right"
                    )
                fig_prob.update_layout(
                    barmode="overlay",
                    xaxis_title="Probabilidad de abandono",
                    yaxis_title="Nº alumnos",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(t=50, b=20, l=0, r=0),
                    height=max(220, 50 + len(titulaciones_sel) * 45),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
            else:
                # Boxplot — más limpio con muchas titulaciones
                # Nombres cortos para eje X: quitar "Grado en " al inicio
                def _nombre_corto(nombre, max_chars=18):
                    for prefijo in ["Grado en Ingeniería en ", "Grado en Ingeniería ",
                                    "Grado en ", "Grado "]:
                        if nombre.startswith(prefijo):
                            nombre = nombre[len(prefijo):]
                            break
                    # Partir en 2 líneas por el espacio más cercano al centro
                    if len(nombre) <= max_chars:
                        return nombre
                    mid = len(nombre) // 2
                    izq = nombre.rfind(" ", 0, mid)
                    der = nombre.find(" ", mid)
                    if izq == -1 and der == -1:
                        return nombre
                    if izq == -1:
                        corte = der
                    elif der == -1:
                        corte = izq
                    else:
                        corte = izq if (mid - izq) <= (der - mid) else der
                    return nombre[:corte] + "<br>" + nombre[corte+1:]

                fig_prob = go.Figure()
                for tit in titulaciones_sel:
                    df_t = df[df["titulacion"] == tit]
                    if df_t.empty:
                        continue
                    fig_prob.add_trace(go.Box(
                        y=df_t["prob_abandono"],
                        name=_nombre_corto(tit),
                        marker_color=color_tit[tit],
                        boxmean=True,
                        hovertemplate=f"{tit}<br>P(abandono): %{{y:.2f}}<extra></extra>"
                    ))
                for umbral, etiqueta, color_ann in [
                    (UMBRALES["riesgo_bajo"],  "Umbral bajo",  _COLOR_MEDIO),
                    (UMBRALES["riesgo_medio"], "Umbral alto",  _COLOR_ALTO)
                ]:
                    fig_prob.add_hline(
                        y=umbral, line_dash="dash", line_color=color_ann,
                        annotation_text=etiqueta,
                        annotation_position="right",
                        annotation_font_color=color_ann
                    )
                fig_prob.update_layout(
                    yaxis_title="Probabilidad<br>de abandono",
                    yaxis=dict(range=[0, 1]),
                    xaxis=dict(tickangle=-30),
                    showlegend=False,
                    margin=dict(t=20, b=20, l=0, r=80),
                    height=max(280, 50 + len(titulaciones_sel) * 45),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
            st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    # =========================================================================
    # BLOQUE 2 — Evolución temporal (una línea por titulación, abandono real)
    # =========================================================================
    st.subheader("Evolución temporal")
    st.caption("Tasa de abandono real por año de cohorte. Una línea por titulación.")

    col_anio = next((c for c in ["anio_cohorte", "curso_aca_ini"] if c in df.columns), None)

    if col_anio and "abandono" in df.columns:
        fig_evol = go.Figure()
        for tit in titulaciones_sel:
            df_t = df[df["titulacion"] == tit]
            evol = (
                df_t.groupby(col_anio)["abandono"]
                .mean()
                .reset_index()
                .rename(columns={col_anio: "anio", "abandono": "tasa_real"})
                .sort_values("anio")
            )
            if evol.empty:
                continue
            fig_evol.add_trace(go.Scatter(
                x=evol["anio"],
                y=evol["tasa_real"] * 100,
                mode="lines+markers",
                name=tit,
                line=dict(color=color_tit[tit], width=2.5),
                marker=dict(size=6),
                hovertemplate=f"{tit}<br>Año %{{x}}: %{{y:.1f}}%<extra></extra>"
            ))
        fig_evol.update_layout(
            xaxis_title="Año de cohorte",
            yaxis_title="Tasa de abandono real (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=30, b=30, l=0, r=0),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_evol, use_container_width=True)
    else:
        st.info("No hay datos temporales disponibles.")

    st.divider()

    # =========================================================================
    # BLOQUE 3 — Factores influyentes (diferencia de medias, barras agrupadas)
    # =========================================================================
    st.subheader("Factores más influyentes")
    st.caption(
        "Top variables por diferencia de medias entre alumnos que abandonan y los que no. "
        "Barras positivas → el factor aumenta el riesgo."
    )

    cols_num = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in _COLS_META and "prob" not in c
    ]

    if cols_num and "abandono" in df.columns:
        # Calcular diferencia de medias por titulación
        importancias = {}
        for tit in titulaciones_sel:
            df_t      = df[df["titulacion"] == tit]
            cols_ok   = [c for c in cols_num if c in df_t.columns]
            if df_t["abandono"].nunique() < 2:
                continue
            diff = (
                df_t[df_t["abandono"] == 1][cols_ok].mean()
                - df_t[df_t["abandono"] == 0][cols_ok].mean()
            )
            importancias[tit] = diff

        if importancias:
            # Top 10 variables por importancia media absoluta entre titulaciones
            df_imp   = pd.DataFrame(importancias)
            top_vars = df_imp.abs().mean(axis=1).nlargest(10).index.tolist()
            df_top   = df_imp.loc[top_vars].copy()

            fig_fact = go.Figure()
            for tit in titulaciones_sel:
                if tit not in df_top.columns:
                    continue
                fig_fact.add_trace(go.Bar(
                    y=[NOMBRES_VARIABLES.get(v, v) for v in top_vars],
                    x=df_top[tit].values,
                    name=tit,
                    orientation="h",
                    marker_color=color_tit[tit],
                    opacity=0.85,
                    hovertemplate=f"{tit}<br>%{{y}}: %{{x:.3f}}<extra></extra>"
                ))
            fig_fact.add_vline(x=0, line_color="gray", line_width=1)
            fig_fact.update_layout(
                barmode="group",
                xaxis_title="Diferencia de medias (abandono − no abandono)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=30, b=20, l=0, r=10),
                height=max(400, 50 + len(top_vars) * 42),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_fact, use_container_width=True)
            with st.expander("ℹ️ Nota técnica", expanded=False):
                st.caption(
                    "Los valores SHAP no están cargados en sesión. "
                    "Se muestra diferencia de medias como proxy de importancia. "
                    "Ejecuta `f6_m01a_shap_global.ipynb` para activar SHAP completo."
                )
    else:
        st.info("No hay variables numéricas disponibles para este análisis.")

    st.divider()

    # =========================================================================
    # BLOQUE 4 — Contexto ramas (resaltando las ramas de las titulaciones sel.)
    # =========================================================================
    st.subheader("Comparativa con el resto de titulaciones")
    st.caption(
        "Las ramas de las titulaciones seleccionadas aparecen resaltadas. "
        "Sirve para contextualizar si el abandono es alto o bajo respecto al resto."
    )

    if "titulacion" in df.columns and "prob_abandono" in df.columns:
        tits_sel_set = set(titulaciones_sel)
        por_tit_ctx = (
            df.groupby("titulacion")["prob_abandono"]
            .mean()
            .reset_index()
            .rename(columns={"prob_abandono": "riesgo_medio"})
            .sort_values("riesgo_medio", ascending=True)
        )
        colores_ctx = [
            _COLOR_ALTO if t in tits_sel_set else COLORES["primario"]
            for t in por_tit_ctx["titulacion"]
        ]
        fig_ctx = go.Figure(go.Bar(
            x=por_tit_ctx["riesgo_medio"] * 100,
            y=por_tit_ctx["titulacion"],
            orientation="h",
            marker_color=colores_ctx,
            text=[f"{v*100:.1f}%" for v in por_tit_ctx["riesgo_medio"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
        ))
        fig_ctx.update_layout(
            xaxis_title="Riesgo medio predicho (%)",
            xaxis=dict(range=[0, max(por_tit_ctx["riesgo_medio"].max() * 115, 10)]),
            margin=dict(t=10, b=30, l=0, r=60),
            height=max(300, 40 + len(por_tit_ctx) * 22),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ctx, use_container_width=True)


def _nota_metodologica_p02():
    """
    Bloque plegable con explicación metodológica orientada al tribunal.
    Responde a preguntas que un evaluador podría hacer al ver esta pestaña.
    """
    with st.expander("📋 Nota metodológica — haz clic para ampliar", expanded=False):
        st.markdown("""
        **¿De dónde vienen las probabilidades?**

        El modelo utilizado es un **Stacking Classifier** (AUC = 0.931, F1 = 0.799 sobre test)
        que combina CatBoost, XGBoost y LightGBM como estimadores base con Regresión Logística
        como meta-learner. La probabilidad de abandono que se muestra es la salida de
        `predict_proba()[:, 1]` del modelo entrenado en Fase 5.

        **¿Qué significa el nivel de riesgo?**

        | Nivel | Umbral de probabilidad |
        |-------|------------------------|
        | Bajo  | < 30%  |
        | Medio | 30% – 60% |
        | Alto  | ≥ 60%  |

        Estos umbrales son configurables en `config_app.py` y han sido seleccionados
        para equilibrar sensibilidad (detectar abandonos reales) y especificidad
        (evitar falsas alarmas). Un análisis de calibración completo se encuentra
        en `f6_m05b_calibracion.ipynb`.

        **¿Por qué la tasa predicha difiere de la real?**

        La diferencia entre tasa real y riesgo medio predicho es normal: el modelo predice
        probabilidades continuas, no etiquetas binarias. La tasa real es la frecuencia observada
        en el conjunto de test; el riesgo medio predicho es la media de probabilidades del modelo.
        Una diferencia pequeña indica buena calibración.

        **Limitaciones de esta vista**

        - Los datos provienen del **conjunto de test** (20% del total, 6.725 alumnos).
          Titulaciones con pocos alumnos en test pueden mostrar tasas inestables.
        - No se muestra información personal identificable de ningún alumno.
          El campo `per_id_ficticio` es un identificador anonimizado.
        """)


# =============================================================================
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# =============================================================================

def mostrar():
    """
    Punto de entrada principal. Llamada desde main.py cuando se selecciona
    la pestaña 'Análisis por titulación'.
    """
    st.title("📚 Análisis por titulación")
    st.markdown(
        "Selecciona una titulación para ver el perfil de abandono, "
        "los factores de riesgo y los alumnos que requieren más atención."
    )

    # --- Carga de datos ---
    with st.spinner("Cargando datos..."):
        try:
            df = _cargar_y_preparar()
        except Exception as e:
            st.error(f"Error cargando los datos: {e}")
            st.stop()

    if df.empty:
        st.warning("El conjunto de datos está vacío.")
        st.stop()

    if "titulacion" not in df.columns:
        st.error(
            "La columna 'titulacion' no está disponible. "
            "Comprueba que `meta_test.parquet` incluye la titulación "
            "y que `loaders.py` hace el join correctamente."
        )
        st.stop()

    # --- Selectores en la página ---
    col_rama = "rama_meta" if "rama_meta" in df.columns else "rama"
    ramas_disponibles = sorted(df[col_rama].dropna().unique().tolist())

    col_sel1, col_sel2 = st.columns([1, 2])

    # Selector 1: filtro por rama (acota la lista de titulaciones)
    with col_sel1:
        ramas_sel = st.multiselect(
            "Rama",
            options=ramas_disponibles,
            default=[],
            placeholder="Todas las ramas",
            key="filtro_rama_p02",
            help="Deja vacío para ver todas. Selecciona para acotar las titulaciones."
        )
    df_filtrado = df[df[col_rama].isin(ramas_sel)] if ramas_sel else df

    titulaciones = _lista_titulaciones(df_filtrado)
    if not titulaciones:
        st.warning("No hay titulaciones disponibles.")
        st.stop()

    # Selector 2: multiselect de titulaciones (como p01 con ramas)
    with col_sel2:
        tits_sel = st.multiselect(
            "Titulación",
            options=titulaciones,
            default=[],
            placeholder="Selecciona una o varias titulaciones",
            key="filtro_tit_p02",
            help="Una titulación → análisis detallado. Varias → comparativa."
        )

    # Pantalla neutra si no ha seleccionado nada
    if not tits_sel:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#718096;">
            <div style="font-size:3rem;">🎓</div>
            <div style="font-size:1.1rem; margin-top:0.5rem;">
                Selecciona una titulación para ver el análisis detallado,<br>
                o varias para comparar entre ellas.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # =========================================================================
    # MODO COMPARATIVO — varias titulaciones seleccionadas
    # =========================================================================
    if len(tits_sel) > 1:
        _bloque_comparativa_titulaciones(df, tits_sel)
        st.divider()
        _nota_metodologica_p02()
        return

    # =========================================================================
    # MODO DETALLE — una sola titulación seleccionada
    # =========================================================================
    titulacion_sel = tits_sel[0]

    # Filtrar al subconjunto de la titulación seleccionada
    df_tit = df[df["titulacion"] == titulacion_sel].copy()
    rama_tit = df_tit[col_rama].mode()[0] if col_rama in df_tit.columns and not df_tit.empty else ""

    # -------------------------------------------------------------------------
    # CONTENIDO PRINCIPAL
    # -------------------------------------------------------------------------
    st.caption(f"Rama: {rama_tit} · {len(df_tit):,} alumnos en el conjunto de test")
    st.divider()

    # Bloque 1 — KPIs
    _bloque_kpis(df_tit)

    st.divider()

    # Bloque 2 — Distribución de riesgo
    _bloque_distribucion_riesgo(df_tit, titulacion_sel)

    st.divider()

    # Bloque 3 — Evolución temporal
    _bloque_evolucion_temporal(df_tit)

    st.divider()

    # Bloque 4 — Factores más influyentes (SHAP o proxy)
    _bloque_factores_shap(df_tit, titulacion_sel)

    st.divider()

    # Bloque 5 — Tabla quitada (datos anonimizados + escalados, sin utilidad)

    # Bloque 6 — Contexto: titulaciones de la misma rama
    _bloque_contexto_titulacion(df, titulacion_sel, rama_tit)

    st.divider()

    # Nota metodológica para el tribunal
    _nota_metodologica_p02()
# Alias para compatibilidad con main.py
show = mostrar
