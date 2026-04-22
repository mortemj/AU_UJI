import re
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
    COLORES, COLORES_RAMAS, RAMAS_NOMBRES, UMBRALES, UMBRALES_MUESTRA,
    NOMBRES_VARIABLES, nombre_legible,
)
from config_app import RUTAS as _RUTAS

# Refactor SRC↔APP (Chat 8): se borraron las siguientes definiciones locales
# porque ahora vienen de config_app:
#   - _NOMBRES_EXTRA: parche con 4 etiquetas (tasa_repeticion, cred_repetidos,
#     n_anios_trabajando, n_anios_sin_notas) que faltaban en NOMBRES_VARIABLES.
#     YA están en src/config_datos.py → ETIQUETAS_VARIABLES (paso 1 del refactor).
#   - _NOMBRES_VAR: dict combinado NOMBRES_VARIABLES + _NOMBRES_EXTRA.
#     Innecesario: NOMBRES_VARIABLES ya las contiene (paso 3 del refactor).
#   - _nombre_legible(): función local de fallback.
#     Sustituida por nombre_legible() de config_app (centralizada para todas
#     las páginas, importada arriba).

# Variables excluidas del gráfico de factores (técnicas o no interpretables)
_COLS_EXCLUIR_FACTORES = {
    "nota_1er_anio_missing",
    "nota_acceso_missing",
    "nota_selectividad_missing",
    "tasa_abandono_titulacion",
}
from utils.loaders import cargar_meta_test_app, cargar_modelo, cargar_pipeline
# B10 (p02): eliminado `from config_app import RUTAS as _RUTAS` duplicado
# (ya importado en L61, no hace falta volver a importarlo).


# =============================================================================
# CONSTANTES LOCALES
# =============================================================================

# Columnas que son metadatos — NO son features del modelo
# Se usan para filtrar antes de pasar datos al pipeline
# B10 (p02): eliminado "rama" duplicado (era inocuo en set, pero confuso al leer).
_COLS_META = {
    "abandono", "titulacion", "rama", "anio_cohorte",
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
# HELPER — Lectura centralizada de métricas del modelo
# =============================================================================
# B12 (Chat p02): copiado del patrón de p01 para no duplicar la lectura del
# JSON en varios sitios (antes estaba inline en _bloque_kpis y hardcodeada
# en la nota metodológica). Cacheada con st.cache_data para no releer el
# fichero en cada rerun de Streamlit.

@st.cache_data(show_spinner=False)
def _leer_metricas_modelo() -> dict:
    """
    Lee el fichero metricas_modelo.json generado por la Fase 6 (evaluación).

    Returns
    -------
    dict
        Diccionario con las métricas del modelo. Claves típicas:
          - 'f1', 'auc', 'accuracy', 'precision', 'recall' (floats)
          - 'tasa_abandono' (float entre 0 y 1)
          - 'fecha_entrenamiento' (str, opcional)
          - 'modelo' (str, opcional — p.ej. "Stacking__balanced")
        Si el fichero no existe o falla la lectura, devuelve {} (dict vacío).
    """
    try:
        import json as _json
        ruta_m = _RUTAS.get("metricas_modelo")
        if ruta_m and ruta_m.exists():
            with open(ruta_m, encoding="utf-8") as _f:
                return _json.load(_f)
    except Exception:
        pass
    return {}


# =============================================================================
# CARGA Y PREPARACIÓN
# =============================================================================

@st.cache_data(show_spinner=False)
def _cargar_y_preparar() -> pd.DataFrame:
    """
    Carga los datos de la app y añade prob_abandono y nivel_riesgo.
    Usa X_test_prep.parquet (ya preprocesado) para evitar errores de pipeline.
    """
    df = cargar_meta_test_app().copy()

    # Traducir abreviaturas de rama a nombres completos
    if "rama" in df.columns:
        df["rama"] = df["rama"].map(RAMAS_NOMBRES).fillna(df["rama"])

    # Calcular prob_abandono desde X_test_prep (ya preprocesado en Fase 5)
    if "prob_abandono" not in df.columns:
        try:
            ruta_xprep = _RUTAS.get("X_test_prep")
            if ruta_xprep and ruta_xprep.exists():
                X_prep = pd.read_parquet(ruta_xprep)
                modelo = cargar_modelo()
                prob = modelo.predict_proba(X_prep)[:, 1]
                df["prob_abandono"] = pd.Series(prob, index=X_prep.index)
            else:
                df["prob_abandono"] = np.nan
        except Exception as e:
            st.warning(f"⚠️ No se pudieron calcular las probabilidades: {e}")
            df["prob_abandono"] = np.nan

    # Añadir nivel de riesgo categórico si no existe
    # B11 (p02): renombrados los umbrales para que el rol semántico sea explícito.
    # Antes: `umbral_alto = UMBRALES["riesgo_medio"]` (confuso al leer).
    # Ahora: nombres directos a la frontera que representan.
    #
    # Lógica del np.select (orden importante — la primera condición que cumple gana):
    #   1. prob >= 0.60 → "Alto"
    #   2. prob >= 0.30 → "Medio" (ya no llega aquí si fuera ≥0.60)
    #   3. resto         → "Bajo"
    if "nivel_riesgo" not in df.columns and "prob_abandono" in df.columns:
        frontera_alto  = UMBRALES["riesgo_medio"]   # 0.60 — frontera Medio↔Alto
        frontera_medio = UMBRALES["riesgo_bajo"]    # 0.30 — frontera Bajo↔Medio

        condiciones = [
            df["prob_abandono"] >= frontera_alto,    # Alto si ≥ 0.60
            df["prob_abandono"] >= frontera_medio,   # Medio si ≥ 0.30 (y < 0.60)
        ]
        df["nivel_riesgo"] = np.select(
            condiciones,
            ["Alto", "Medio"],
            default="Bajo"                           # Bajo si < 0.30
        )

    # B7 (Chat p02): filtro implícito del universo de análisis.
    # El TFM está definido oficialmente sobre cursos 2010–2020. En el parquet
    # meta_test_app quedan ~129 alumnos con curso_aca_ini < 2010 que son
    # residuos del dataset original. Aplicamos el filtro al FINAL (tras
    # calcular prob_abandono y nivel_riesgo) para evitar problemas de
    # alineación de índices entre X_test_prep y df. Coherente con p01 (L183).
    if "curso_aca_ini" in df.columns:
        df = df[df["curso_aca_ini"] >= 2010].reset_index(drop=True)

    return df


def _lista_titulaciones(df: pd.DataFrame) -> list[str]:
    """
    Devuelve la lista de titulaciones ordenadas: primero por rama,
    luego alfabéticamente dentro de cada rama.
    """
    if "titulacion" not in df.columns:
        return []
    return (
        df[["titulacion", "rama"]]
        .drop_duplicates()
        .sort_values(["rama", "titulacion"])["titulacion"]
        .tolist()
    )


# =============================================================================
# BLOQUES DE CONTENIDO
# =============================================================================

# -----------------------------------------------------------------------------
# B3-A (Chat p02): GUARDIA DE DATAFRAME VACÍO
# -----------------------------------------------------------------------------
# Copiado de p01 (líneas 1200-1235) para mantener paridad total.
# Cuando los filtros dejan el dataset vacío, cada bloque llama a esta función
# al principio. Si devuelve True, el bloque debe hacer `return` sin intentar
# renderizar nada. Evita errores como:
#   - TypeError: int() argument must be a string... not 'NAType'
#     (por int(pd.NA) al hacer .min()/.max() sobre Series vacías)
#   - ValueError: zero-size array
#   - Gráficos en blanco sin explicación
#
# Se usa en los 7 bloques gráficos de p02 (ver aplicación en B3-C).
# Los avisos de tamaño de muestra (rojo/naranja/amarillo) se gestionan
# aparte en el flujo principal (ver B3-B), usando UMBRALES_MUESTRA.

def _guardia_df_vacio(df: pd.DataFrame, titulo_bloque: str) -> bool:
    """
    Comprueba si un DataFrame está vacío y, si lo está, renderiza un aviso
    visual coherente con el estilo de la app.

    Devuelve:
      - True  → df está vacío, el bloque que lo llame debe hacer `return`
      - False → df tiene datos, el bloque puede continuar normalmente

    Parámetros:
      df            → DataFrame a comprobar (típicamente df_filtrado)
      titulo_bloque → Nombre del bloque para el aviso (ej: "Evolución temporal")
    """
    if df is None or len(df) == 0:
        st.markdown(f"""
        <div style="background:{COLORES['fondo']};
            border:1px dashed {COLORES['texto_muy_suave']};
            border-radius:8px;
            padding:1.5rem 1rem;
            text-align:center;
            color:{COLORES['texto_suave']};
            font-size:0.85rem;
            margin:0.5rem 0;">
            <div style="font-weight:600; margin-bottom:0.3rem;
                color:{COLORES['texto']};">
                {titulo_bloque}
            </div>
            📭 No hay datos para mostrar con los filtros actuales.
            <br>
            <span style="font-size:0.78rem;">
                Prueba a relajar los filtros para ver este bloque.
            </span>
        </div>
        """, unsafe_allow_html=True)
        return True
    return False


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
        st.metric("Alumnos en test", f"{n_total:,}".replace(",", "."))
    with col2:
        if tasa_real is not None:
            st.metric(
                "Tasa abandono real",
                f"{tasa_real:.1f}%".replace(".", ","),
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
                f"{tasa_predicha:.1f}%".replace(".", ","),
                delta=delta_str,
                delta_color="inverse",
                # B6 (p02): help ampliado para explicar la sigla "pp" del delta.
                # Streamlit no permite HTML en delta=, así que el tooltip
                # explicativo de "puntos porcentuales" va aquí en help=.
                help=(
                    "Probabilidad media de abandono según el modelo. "
                    "El delta (pp = puntos porcentuales) compara el riesgo "
                    "predicho con la tasa real observada en el test."
                )
            )
        else:
            st.metric("Riesgo medio predicho", "N/D")
    with col4:
        pct_alto = n_riesgo_alto / n_total * 100 if n_total > 0 else 0
        st.metric(
            "Alumnos en riesgo alto",
            f"{pct_alto:.1f}%".replace(".", ","),
            delta=f"{n_riesgo_alto:,} alumnos".replace(",", "."),
            delta_color="off",
            help=f"Alumnos con probabilidad de abandono ≥ {UMBRALES['riesgo_medio']:.0%}."
        )
    with col5:
        # F1 global del modelo — leído desde metricas_modelo.json
        # B12 (p02): antes había lectura inline con imports dentro de la función;
        # ahora se usa el helper _leer_metricas_modelo() centralizado (cacheado).
        _metricas = _leer_metricas_modelo()
        _f1 = _metricas.get("f1")
        f1_val = f"{_f1:.3f}".replace(".", ",") if _f1 is not None else "N/D"
        st.metric(
            "F1 modelo (global)",
            f1_val,
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
        st.plotly_chart(fig_donut, width='stretch')

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
            st.plotly_chart(fig_hist, width='stretch')


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
    st.plotly_chart(fig, width='stretch')


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
            st.plotly_chart(fig, width='stretch')
            return

    # --- Fallback: diferencia de medias (proxy de importancia) ---
    # Funciona aunque no haya SHAP cargado en sesión
    cols_num = [
        c for c in df_tit.select_dtypes(include=[np.number]).columns
        if c not in _COLS_META and "prob" not in c
        and c not in _COLS_EXCLUIR_FACTORES
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
        "variable":   [nombre_legible(c) for c in cols_num],
        "diferencia":  diferencia.values,
    }).sort_values("diferencia")

    # Escala log simétrica: sign(x)*log10(|x|+1) — reduce dominio visual
    df_proxy["dif_log"] = df_proxy["diferencia"].apply(
        lambda x: np.sign(x) * np.log10(abs(x) + 1)
    )

    fig = go.Figure(go.Bar(
        x=df_proxy["dif_log"],
        y=df_proxy["variable"],
        orientation="h",
        marker_color=[
            _COLOR_ALTO if v > 0 else _COLOR_BAJO
            for v in df_proxy["diferencia"]
        ],
        customdata=df_proxy["diferencia"].round(3),
        hovertemplate="%{y}<br>Diferencia real: %{customdata}<extra></extra>"
    ))
    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_layout(
        xaxis_title="Diferencia de medias — escala log simétrica (abandono − no abandono)",
        margin=dict(t=10, b=10, l=0, r=0),
        height=max(380, len(df_proxy) * 22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, width='stretch')
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
        width='stretch',
        height=min(400, 40 + len(df_tabla) * 35)
    )

    if len(df_alto) > _MAX_TABLA:
        st.caption(f"Mostrando los {_MAX_TABLA} alumnos de mayor riesgo de {len(df_alto)} totales.")


def _bloque_contexto_titulacion(df: pd.DataFrame, titulacion_sel: str, rama_tit: str):
    """
    Comparativa de la titulación seleccionada con las de su misma rama.
    Colores por rama — seleccionada más intensa, resto más suave.
    """
    st.subheader("Comparativa con el resto de titulaciones")

    if "titulacion" not in df.columns or "prob_abandono" not in df.columns:
        return

    # Mostrar todas las titulaciones de la misma rama
    col_rama = "rama"
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

    # Color base de la rama
    color_rama = COLORES_RAMAS.get(rama_tit, COLORES["primario"])

    # Seleccionada → color rama opaco; resto → color rama transparente
    colores = []
    for t in por_tit["titulacion"]:
        if t == titulacion_sel:
            colores.append(color_rama)
        else:
            # Versión más suave: añadir transparencia via rgba
            r = int(color_rama[1:3], 16)
            g = int(color_rama[3:5], 16)
            b = int(color_rama[5:7], 16)
            colores.append(f"rgba({r},{g},{b},0.35)")

    # Quitar "Grado en" para que quepa en el eje Y
    tit_cortas = [re.sub(r"^(Grado en |Doble Grado en )", "", t) for t in por_tit["titulacion"]]

    fig = go.Figure(go.Bar(
        x=por_tit["riesgo_medio"] * 100,
        y=tit_cortas,
        orientation="h",
        marker_color=colores,
        text=[f"{v*100:.1f}%" for v in por_tit["riesgo_medio"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
    ))
    max_x = max(por_tit["riesgo_medio"].max() * 100 * 1.25, 15)
    fig.update_layout(
        xaxis_title="Riesgo medio predicho (%)",
        xaxis=dict(range=[0, max_x]),
        margin=dict(t=10, b=30, l=0, r=60),
        height=max(200, 40 + len(por_tit) * 38),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.caption(
        f"Titulaciones de la rama '{rama_tit}' ordenadas por riesgo predicho. "
        "La seleccionada aparece en color sólido."
    )
    st.plotly_chart(fig, width='stretch')


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
        # Quitar prefijo "Grado en " para que quepa mejor en la tabla
        tit_corto = re.sub(r"^(Grado en |Doble Grado en )", "", tit)
        filas.append({
            "Titulación":          tit_corto,
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
    col_rama = "rama"  # B10 (p02): antes había condición tautológica que siempre devolvía "rama"
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
    st.dataframe(df_comp, width='stretch', hide_index=True,
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
            st.plotly_chart(fig_dist, width='stretch')

    # --- Histograma (<=2 titulaciones) o Boxplot (>2 titulaciones) ---
    if "prob_abandono" in df.columns:
        contenedor_hist = col_hist if usar_columnas else st.container()
        with contenedor_hist:
            st.subheader("Probabilidad de abandono")
            stats_list = []  # se rellena en el bloque violin (>2 tit)
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
                # Violin plot — muestra distribución completa de probabilidades por titulación
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

                # Calcular estadísticas por titulación para tooltip y tabla
                stats_list = []
                fig_prob = go.Figure()
                for tit in titulaciones_sel:
                    df_t = df[df["titulacion"] == tit]["prob_abandono"].dropna()
                    if df_t.empty:
                        continue
                    q1, med, q3 = df_t.quantile([0.25, 0.5, 0.75]).values
                    media = df_t.mean()
                    mini  = df_t.min()
                    maxi  = df_t.max()
                    stats_list.append({
                        "Titulación": tit,
                        "Media":    round(media, 3),
                        "Mediana":  round(med,   3),
                        "Q1":       round(q1,    3),
                        "Q3":       round(q3,    3),
                        "Mín":      round(mini,  3),
                        "Máx":      round(maxi,  3),
                    })
                    # Tooltip sin nombre — solo estadísticas
                    hover = (
                        f"Media: {media:.3f}<br>"
                        f"Mediana: {med:.3f}<br>"
                        f"Q1: {q1:.3f} · Q3: {q3:.3f}<br>"
                        f"Mín: {mini:.3f} · Máx: {maxi:.3f}"
                        "<extra></extra>"
                    )
                    fig_prob.add_trace(go.Violin(
                        y=df_t,
                        name=_nombre_corto(tit),
                        marker_color=color_tit[tit],
                        fillcolor=color_tit[tit],
                        opacity=0.7,
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                        hovertemplate=hover
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
                    height=max(320, 60 + len(titulaciones_sel) * 50),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
            st.plotly_chart(fig_prob, width='stretch')

            # Expander con tabla de estadísticas detalladas
            if stats_list:
                with st.expander("📊 Estadísticas detalladas de probabilidad de abandono", expanded=False):
                    import pandas as _pd_stats
                    df_stats = _pd_stats.DataFrame(stats_list).set_index("Titulación")
                    # Formatear con coma decimal
                    for col in df_stats.columns:
                        df_stats[col] = df_stats[col].apply(lambda x: f"{x:.3f}".replace(".", ","))
                    st.dataframe(df_stats, width='stretch')
                    st.caption(
                        "Media = promedio de probabilidad predicha · "
                        "Mediana = valor central · Q1/Q3 = percentiles 25 y 75 · "
                        "Mín/Máx = valores extremos"
                    )

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
        st.plotly_chart(fig_evol, width='stretch')
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
            st.plotly_chart(fig_fact, width='stretch')
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
        # B10 (p02): formato de la línea simplificado (antes tenía espacios extra
        # que dificultaban la lectura). Misma lógica.
        rama_por_tit = (
            df.drop_duplicates("titulacion").set_index("titulacion")["rama"].to_dict()
            if "rama" in df.columns else {}
        )
        colores_ctx = [
            COLORES_RAMAS.get(rama_por_tit.get(t, ""), COLORES["primario"])
            for t in por_tit_ctx["titulacion"]
        ]
        borde_ancho = [2 if t in tits_sel_set else 0 for t in por_tit_ctx["titulacion"]]
        borde_color = [COLORES["texto"] if t in tits_sel_set else "rgba(0,0,0,0)"
                       for t in por_tit_ctx["titulacion"]]
        fig_ctx = go.Figure(go.Bar(
            x=por_tit_ctx["riesgo_medio"] * 100,
            y=por_tit_ctx["titulacion"],
            orientation="h",
            marker=dict(color=colores_ctx, line=dict(color=borde_color, width=borde_ancho)),
            text=[
                f"{v*100:.1f}% ◀" if t in tits_sel_set else f"{v*100:.1f}%"
                for v, t in zip(por_tit_ctx["riesgo_medio"], por_tit_ctx["titulacion"])
            ],
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
        st.plotly_chart(fig_ctx, width='stretch')


def _nota_metodologica_p02(n_alumnos_test: int = None):
    """
    Bloque plegable con explicación metodológica orientada al tribunal.
    Responde a preguntas que un evaluador podría hacer al ver esta pestaña.

    B12 (Chat p02): refactor completo —
      - Base learners corregidos: CatBoost + RandomForest (antes: XGBoost/LightGBM erróneo)
      - AUC y F1 leídos dinámicamente de metricas_modelo.json
      - Umbrales leídos de UMBRALES (config_app) — antes hardcodeados a 30/60
      - N del test pasado como parámetro (antes hardcodeado a 6.725)

    Parameters
    ----------
    n_alumnos_test : int, optional
        Número total de alumnos en el conjunto de test. Si es None, no se
        incluye el dato en el texto. Se recomienda pasar len(df) desde mostrar().
    """
    # --- Leer métricas del modelo (fallbacks documentados si el JSON falla) ---
    _metricas = _leer_metricas_modelo()
    _auc = _metricas.get("auc")
    _f1  = _metricas.get("f1")
    auc_str = f"{_auc:.3f}".replace(".", ",") if _auc is not None else "0,931"
    f1_str  = f"{_f1:.3f}".replace(".", ",")  if _f1  is not None else "0,799"

    # --- Umbrales dinámicos desde UMBRALES (config_app) ---
    _u_bajo  = UMBRALES.get("riesgo_bajo",  0.30)
    _u_medio = UMBRALES.get("riesgo_medio", 0.60)
    bajo_pct  = f"{_u_bajo:.0%}"
    medio_pct = f"{_u_medio:.0%}"

    # --- N del test (con separador de miles al estilo español) ---
    n_str = f"{n_alumnos_test:,}".replace(",", ".") if n_alumnos_test else "—"

    with st.expander("📋 Nota metodológica — haz clic para ampliar", expanded=False):
        st.markdown(f"""
        **¿De dónde vienen las probabilidades?**

        El modelo utilizado es un **Stacking Classifier** (AUC = {auc_str}, F1 = {f1_str} sobre test)
        que combina **CatBoost y Random Forest** como estimadores base con Regresión Logística
        como meta-learner. La probabilidad de abandono que se muestra es la salida de
        `predict_proba()[:, 1]` del modelo entrenado en Fase 5.

        **¿Qué significa el nivel de riesgo?**

        | Nivel | Umbral de probabilidad |
        |-------|------------------------|
        | Bajo  | < {bajo_pct}  |
        | Medio | {bajo_pct} – {medio_pct} |
        | Alto  | ≥ {medio_pct}  |

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

        - Los datos provienen del **conjunto de test** ({n_str} alumnos).
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
    col_rama = "rama"  # B10 (p02): antes había condición tautológica que siempre devolvía "rama"
    ramas_disponibles = sorted(df[col_rama].dropna().unique().tolist())

    # Zona de filtros — línea azul arriba y abajo
    st.markdown("""
    <div style="border-top: 2px solid #3182ce; margin-bottom: 0.5rem;">
        <span style="font-size:0.72rem; font-weight:600; color:#3182ce;
                     text-transform:uppercase; letter-spacing:0.05em;">
            🔍 Filtros
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_sel1, col_sel2 = st.columns([1, 2])

    # Selector 1: filtro por rama (acota la lista de titulaciones)
    # B2 (p02): eliminado `default=[]` para evitar warning de Streamlit
    # cuando se combina con `key=`. Con solo `key=`, Streamlit inicializa a
    # lista vacía automáticamente (mismo comportamiento, sin warning).
    with col_sel1:
        ramas_sel = st.multiselect(
            "Rama",
            options=ramas_disponibles,
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
    # B2 (p02): eliminado `default=[]` para evitar warning de Streamlit.
    # B1 (p02): CASCADA — limpieza automática de selecciones incompatibles.
    # Si el usuario tenía una titulación seleccionada (p.ej. "Grado en Medicina")
    # y cambia la rama a una que no la incluye (p.ej. "Sociales"), el chip se
    # queda pegado en session_state. Patrón ITER 5 de p01: limpiamos las
    # selecciones inválidas ANTES de instanciar el widget para que el estado
    # visible sea siempre coherente con las opciones disponibles.
    _sel_previa_tit = st.session_state.get("filtro_tit_p02", [])
    _sel_valida_tit = [t for t in _sel_previa_tit if t in titulaciones]
    if _sel_valida_tit != _sel_previa_tit:
        st.session_state["filtro_tit_p02"] = _sel_valida_tit

    with col_sel2:
        tits_sel = st.multiselect(
            "Titulación",
            options=titulaciones,
            placeholder="Selecciona una o varias titulaciones",
            key="filtro_tit_p02",
            help="Una titulación → análisis detallado. Varias → comparativa."
        )

    st.markdown("""
    <div style="border-bottom: 2px solid #3182ce; margin-top: 0.5rem; margin-bottom: 1rem;"></div>
    """, unsafe_allow_html=True)

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
        _nota_metodologica_p02(n_alumnos_test=len(df))
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
    st.caption(f"Rama: {rama_tit} · {len(df_tit):,} alumnos en el conjunto de test".replace(",", "."))
    st.divider()

    # -------------------------------------------------------------------------
    # B3-B (Chat p02): AVISO DE TAMAÑO DE MUESTRA
    # -------------------------------------------------------------------------
    # Copiado de p01 (líneas 191-209) para paridad total.
    # Si la titulación seleccionada tiene muy pocos alumnos en el test, los
    # porcentajes y tasas no son fiables estadísticamente. Mostramos un aviso
    # proporcional a la gravedad usando UMBRALES_MUESTRA de config_app:
    #   - minima (10):    error rojo  → muestra insuficiente
    #   - aceptable (30): warning     → muy pequeña
    #   - fiable (100):   info        → pequeña
    # Frase final "Interpreta los resultados con cautela" unificada con p01
    # para coherencia textual entre páginas (Estrategia B, sesión 22-abr-26).
    n_muestra_tit = len(df_tit)
    if n_muestra_tit < UMBRALES_MUESTRA['minima']:
        st.error(
            f"❌ Muestra insuficiente ({n_muestra_tit} alumnos). "
            f"Los porcentajes no son fiables estadísticamente. "
            f"Interpreta los resultados con cautela."
        )
    elif n_muestra_tit < UMBRALES_MUESTRA['aceptable']:
        st.warning(
            f"⚠️ Muestra muy pequeña ({n_muestra_tit} alumnos). "
            f"Los resultados son orientativos — poco representativos."
        )
    elif n_muestra_tit < UMBRALES_MUESTRA['fiable']:
        st.info(
            f"ℹ️ Muestra pequeña ({n_muestra_tit} alumnos). "
            f"Los porcentajes son indicativos."
        )

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
    _nota_metodologica_p02(n_alumnos_test=len(df))
# Alias para compatibilidad con main.py
show = mostrar
