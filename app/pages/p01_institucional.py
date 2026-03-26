# =============================================================================
# p01_institucional.py
# Pestaña 1 — Visión institucional
#
# ¿QUÉ HACE ESTE FICHERO?
#   Muestra una visión global del abandono universitario en la UJI.
#   Pensada para gestores y dirección académica que necesitan ver el
#   panorama completo de un vistazo, sin entrar en detalle de alumnos
#   concretos.
#
# ESTRUCTURA DE LA PÁGINA:
#   0. Filtros globales (sidebar) — afectan a TODOS los bloques
#   1. KPIs rápidos — métricas clave según filtros activos
#   2. Evolución temporal — tasa de abandono por año de cohorte
#   3. Abandono por rama — barras horizontales comparativas
#   4. Top titulaciones — tabla ordenable con más detalle
#   5. Distribución de riesgo predicho — donut bajo/medio/alto
#
# DATOS QUE USA:
#   - meta_test.parquet (vía loaders.cargar_meta_test)
#   - modelo + pipeline (vía loaders.cargar_modelo / cargar_pipeline)
#     para calcular probabilidades de abandono predichas
#
# REQUISITOS:
#   - config_app.py accesible (un nivel arriba)
#   - utils/loaders.py disponible
#   - Fase 5 ejecutada (modelo y pipeline en data/05_modelado/)
#   - Fase 6 ejecutada (meta_test en data/06_evaluacion/)
#
# GENERA:
#   Página HTML interactiva renderizada por Streamlit. No genera ficheros.
#
# SIGUIENTE:
#   pages/p02_titulacion.py — análisis por titulación individual
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Imports internos
# ---------------------------------------------------------------------------
# _path_setup añade app/ a sys.path de forma robusta en Windows/OneDrive
import _path_setup  # noqa: F401

from config_app import COLORES, COLORES_RAMAS, COLORES_RIESGO, NOMBRES_VARIABLES, UMBRALES
from utils.loaders import cargar_meta_test, cargar_modelo, cargar_pipeline


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def show():
    """Renderiza la pestaña de visión institucional completa."""

    # Título de la página
    st.markdown(f"""
    <h2 style="color: {COLORES['primario']}; margin-bottom: 0.2rem;">
        🏛️ Visión institucional
    </h2>
    <p style="color: {COLORES['texto_suave']}; margin-top: 0; font-size: 0.95rem;">
        Panorama global de abandono universitario en la UJI · Datos de test (2010–2020)
    </p>
    """, unsafe_allow_html=True)

    # --- Carga de datos ---
    # Intentamos cargar. Si algo falla, mostramos error amigable y paramos.
    with st.spinner("Cargando datos..."):
        try:
            df_raw = cargar_meta_test()
            modelo  = cargar_modelo()
            pipeline = cargar_pipeline()
        except FileNotFoundError as e:
            st.error(f"❌ No se pudieron cargar los datos:\n\n{e}")
            st.stop()

    # --- Calcular probabilidades predichas ---
    # El modelo necesita los datos preprocesados por el pipeline.
    # Separamos las columnas que son features de las que son metadatos.
    # Las features son las que el pipeline sabe transformar.
    df = _añadir_probabilidades(df_raw, modelo, pipeline)

    # --- Añadir columna de nivel de riesgo (bajo / medio / alto) ---
    # Basada en los umbrales definidos en config_app.py
    df = _añadir_nivel_riesgo(df)

    # --- Filtros en sidebar ---
    # Los filtros se aplican ANTES de pasar los datos a cada bloque.
    # Así todos los bloques ven exactamente los mismos datos filtrados.
    df_filtrado = _aplicar_filtros_sidebar(df)

    # --- Aviso si los filtros dejan muy pocos datos ---
    if len(df_filtrado) < 10:
        st.warning(
            f"⚠️ La combinación de filtros seleccionada solo contiene "
            f"{len(df_filtrado)} registros. Los resultados pueden no ser representativos."
        )

    st.divider()

    # --- Bloques de contenido ---
    # Cada bloque recibe el DataFrame ya filtrado.
    _bloque_kpis(df_filtrado)
    st.divider()
    _bloque_evolucion_temporal(df_filtrado)
    st.divider()
    _bloque_abandono_por_rama(df_filtrado)
    st.divider()
    _bloque_top_titulaciones(df_filtrado)
    st.divider()
    _bloque_distribucion_riesgo(df_filtrado)

    st.divider()

    # Nota metodológica — igual que en p00 y p02
    with st.expander("📋 Nota metodológica — haz clic para ampliar", expanded=False):
        st.markdown("""
        **Dataset:** 109.568 registros · 30.872 alumnos únicos · Universitat Jaume I · Cursos 2010–2020.

        **Variable objetivo:** abandono definitivo del grado (definición estricta). Tasa de abandono en test: **29,2 %**.

        **Modelo final:** Stacking con CatBoost, XGBoost, LightGBM y Random Forest como modelos base,
        y regresión logística como meta-learner. AUC = 0.931 · F1 = 0.799.

        **Sobre los gráficos:** los porcentajes de riesgo son predicciones del modelo, no valores reales observados.
        La tasa de abandono real corresponde a los datos históricos del conjunto de test (6.725 observaciones).

        **Limitaciones:** el modelo está entrenado con datos hasta 2020. Las predicciones para cohortes
        posteriores deben interpretarse con cautela.
        """)


# =============================================================================
# HELPERS DE DATOS
# =============================================================================

def _añadir_probabilidades(df_raw: pd.DataFrame, modelo, pipeline) -> pd.DataFrame:
    """
    Calcula la probabilidad de abandono predicha por el modelo para cada
    alumno del conjunto de test y la añade como columna 'prob_abandono'.

    El pipeline transforma las features brutas al formato que espera el modelo.
    Solo aplicamos el pipeline a las columnas que él conoce (las features),
    no a los metadatos (titulacion, rama, etc.).
    """
    df = df_raw.copy()

    # Usamos pipeline.feature_names_in_ para saber exactamente qué columnas
    # necesita el pipeline — robusto, sin listas manuales que queden desfasadas
    cols_features   = list(pipeline.feature_names_in_)
    cols_disponibles = [c for c in cols_features if c in df.columns]
    X = df[cols_disponibles]

    # Transformamos con el pipeline y predecimos probabilidades
    # predict_proba devuelve [[prob_no_abandono, prob_abandono], ...]
    # nos quedamos con la columna 1 (prob de abandono)
    try:
        X_prep = pipeline.transform(X)
        df['prob_abandono'] = modelo.predict_proba(X_prep)[:, 1]
    except Exception as e:
        # Si falla la predicción, ponemos NaN y avisamos
        st.warning(f"⚠️ No se pudieron calcular las probabilidades: {e}")
        df['prob_abandono'] = np.nan

    return df


def _añadir_nivel_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica cada alumno en nivel de riesgo según su probabilidad predicha.
    Añade la columna 'nivel_riesgo' con valores: 'Bajo', 'Medio', 'Alto'.
    """
    df = df.copy()

    def _clasificar(prob):
        if pd.isna(prob):
            return 'Desconocido'
        if prob < UMBRALES['riesgo_bajo']:
            return 'Bajo'
        elif prob < UMBRALES['riesgo_medio']:
            return 'Medio'
        else:
            return 'Alto'

    df['nivel_riesgo'] = df['prob_abandono'].apply(_clasificar)
    return df


def _aplicar_filtros_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtros en fila horizontal debajo del título de la página.
    Encadenados: sexo → rama → año de cohorte.
    """
    col_f1, col_f2, col_f3, col_info = st.columns([1, 1.5, 1.5, 1])
    df_f = df.copy()

    with col_f1:
        # Usar sexo_meta (texto legible) en lugar de sexo (numérico del pipeline)
        col_sexo = 'sexo_meta' if 'sexo_meta' in df.columns else 'sexo'
        opciones_sexo = ["Todos"] + sorted(df[col_sexo].dropna().unique().tolist()) \
            if col_sexo in df.columns else ["Todos"]
        sexo_sel = st.selectbox("Sexo", options=opciones_sexo, index=0,
                                help="Filtra por sexo", key="filtro_sexo_p01")
        if sexo_sel != "Todos":
            df_f = df_f[df_f[col_sexo] == sexo_sel]

    with col_f2:
        col_rama_s = 'rama_meta' if 'rama_meta' in df_f.columns else 'rama'
        ramas_disp = sorted(df_f[col_rama_s].dropna().unique().tolist()) \
            if col_rama_s in df_f.columns else []
        ramas_sel = st.multiselect(
            "Rama", options=ramas_disp,
            default=[],
            placeholder="Todas las ramas",
            key="filtro_rama_p01",
            help="Deja vacío para ver todas. Selecciona para filtrar."
        )
        # Si no selecciona nada → todas las ramas
        if ramas_sel:
            df_f = df_f[df_f[col_rama_s].isin(ramas_sel)]
    with col_f3:
        if 'curso_aca_ini' in df_f.columns and df_f['curso_aca_ini'].notna().any():
            a_min = int(df_f['curso_aca_ini'].min())
            a_max = int(df_f['curso_aca_ini'].max())
            if a_min < a_max:
                rango = st.slider("Cohorte", min_value=a_min, max_value=a_max,
                                  value=(a_min, a_max), step=1, key="filtro_anios_p01")
                df_f = df_f[
                    (df_f['curso_aca_ini'] >= rango[0]) &
                    (df_f['curso_aca_ini'] <= rango[1])
                ]

    with col_info:
        st.markdown(f"""
        <div style="font-size:0.78rem; color:{COLORES['texto_suave']};
            background:{COLORES['fondo']}; border-radius:6px;
            padding:0.5rem 0.75rem; margin-top:1.6rem;">
            📋 <strong>{len(df_f):,}</strong> de {len(df):,}
        </div>
        """, unsafe_allow_html=True)

    return df_f


# =============================================================================
# BLOQUE 1: KPIs rápidos
# =============================================================================

def _bloque_kpis(df: pd.DataFrame):
    """Fila de métricas clave calculadas sobre los datos filtrados."""

    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        📌 Indicadores clave
    </h4>
    """, unsafe_allow_html=True)

    # Calculamos las métricas
    n_total        = len(df)
    n_abandono     = df['abandono'].sum() if 'abandono' in df.columns else 0
    tasa_abandono  = (n_abandono / n_total * 100) if n_total > 0 else 0
    n_riesgo_alto  = (df['nivel_riesgo'] == 'Alto').sum()
    pct_riesgo_alto = (n_riesgo_alto / n_total * 100) if n_total > 0 else 0
    n_titulaciones = df['titulacion'].nunique() if 'titulacion' in df.columns else 0
    col_rama_kpi   = 'rama_meta' if 'rama_meta' in df.columns else 'rama'
    n_ramas        = df[col_rama_kpi].nunique() if col_rama_kpi in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            label="👥 Alumnos (test)",
            value=f"{n_total:,}",
            help="Número de registros en el conjunto de test con los filtros activos"
        )
    with c2:
        st.metric(
            label="📉 Tasa abandono real",
            value=f"{tasa_abandono:.1f} %",
            help="Porcentaje de abandono observado en los datos (etiqueta real, no predicción)"
        )
    with c3:
        st.metric(
            label="🔴 En riesgo alto",
            value=f"{n_riesgo_alto:,}",
            delta=f"{pct_riesgo_alto:.1f} % del total",
            delta_color="inverse",  # rojo si sube (más riesgo = peor)
            help=f"Alumnos con probabilidad predicha ≥ {UMBRALES['riesgo_medio']:.0%}"
        )
    with c4:
        st.metric(
            label="🎓 Titulaciones",
            value=f"{n_titulaciones}",
            help="Número de titulaciones distintas con los filtros activos"
        )
    with c5:
        st.metric(
            label="📚 Ramas",
            value=f"{n_ramas}",
            help="Número de ramas de conocimiento con los filtros activos"
        )


# =============================================================================
# BLOQUE 2: Evolución temporal
# =============================================================================

def _bloque_evolucion_temporal(df: pd.DataFrame):
    """Gráfico de línea: tasa de abandono real por año de cohorte."""

    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        📈 Evolución temporal del abandono
    </h4>
    """, unsafe_allow_html=True)

    if 'curso_aca_ini' not in df.columns or 'abandono' not in df.columns:
        st.info("No hay datos de cohorte disponibles con los filtros actuales.")
        return

    # Agrupamos por año y calculamos tasa de abandono
    evolucion = (
        df.groupby('curso_aca_ini')
        .agg(
            n_total=('abandono', 'count'),
            n_abandono=('abandono', 'sum'),
            prob_media=('prob_abandono', 'mean')
        )
        .reset_index()
    )
    evolucion = evolucion.rename(columns={'curso_aca_ini': 'anio_cohorte'})
    evolucion['tasa_abandono_pct'] = (
        evolucion['n_abandono'] / evolucion['n_total'] * 100
    ).round(1)
    evolucion['prob_media_pct'] = (evolucion['prob_media'] * 100).round(1)

    # Gráfico con dos líneas: abandono real y probabilidad media predicha
    fig = go.Figure()

    # Línea 1: abandono real observado
    fig.add_trace(go.Scatter(
        x=evolucion['anio_cohorte'],
        y=evolucion['tasa_abandono_pct'],
        mode='lines+markers',
        name='Abandono real (%)',
        line=dict(color=COLORES['abandono'], width=2.5),
        marker=dict(size=8),
        hovertemplate=(
            "<b>Cohorte %{x}</b><br>"
            "Abandono real: %{y:.1f}%<br>"
            "N alumnos: %{customdata}<extra></extra>"
        ),
        customdata=evolucion['n_total']
    ))

    # Línea 2: probabilidad media predicha por el modelo
    fig.add_trace(go.Scatter(
        x=evolucion['anio_cohorte'],
        y=evolucion['prob_media_pct'],
        mode='lines+markers',
        name='Riesgo predicho medio (%)',
        line=dict(color=COLORES['primario'], width=2, dash='dash'),
        marker=dict(size=6),
        hovertemplate=(
            "<b>Cohorte %{x}</b><br>"
            "Riesgo predicho medio: %{y:.1f}%<extra></extra>"
        )
    ))

    # Anotación de cautela para la cohorte más reciente (truncamiento temporal)
    # Las cohortes recientes tienen tasa baja no porque abandonen menos
    # sino porque aún no han tenido tiempo suficiente para abandonar
    anio_max = int(evolucion['anio_cohorte'].max())
    tasa_max = evolucion[evolucion['anio_cohorte'] == anio_max]['tasa_abandono_pct'].values[0]
    fig.add_annotation(
        x=anio_max, y=tasa_max,
        text="⚠️ Truncamiento<br>temporal",
        showarrow=True, arrowhead=2,
        arrowcolor=COLORES['advertencia'],
        font=dict(size=11, color=COLORES['advertencia']),
        bgcolor="white",
        bordercolor=COLORES['advertencia'],
        borderwidth=1,
        ax=-70, ay=-40,
    )

    fig.update_layout(
        xaxis_title="Año de cohorte",
        yaxis_title="Porcentaje (%)",
        yaxis=dict(range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
            font=dict(size=13),
            bgcolor="rgba(255,255,255,0.0)",
            borderwidth=0,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=30, b=80),
        height=400,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORES['borde'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💡 La caída en cohortes recientes (2018-2020) refleja truncamiento temporal: "
        "los alumnos más recientes aún no han tenido tiempo suficiente para abandonar. "
        "La línea azul (riesgo predicho) es más estable porque el modelo no depende del tiempo observado."
    )


# =============================================================================
# BLOQUE 3: Abandono por rama
# =============================================================================

def _bloque_abandono_por_rama(df: pd.DataFrame):
    """Barras horizontales: tasa de abandono real por rama de conocimiento."""

    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        📚 Abandono por rama de conocimiento
    </h4>
    """, unsafe_allow_html=True)

    # Usamos rama_meta (legible) si existe, sino rama (puede ser numérica del pipeline)
    col_rama = 'rama_meta' if 'rama_meta' in df.columns else 'rama'
    if col_rama not in df.columns or 'abandono' not in df.columns:
        st.info("No hay datos de rama disponibles con los filtros actuales.")
        return

    por_rama = (
        df.groupby(col_rama)
        .agg(
            n_total=('abandono', 'count'),
            n_abandono=('abandono', 'sum'),
            prob_media=('prob_abandono', 'mean')
        )
        .reset_index()
    )
    por_rama = por_rama.rename(columns={col_rama: 'rama'})
    por_rama['tasa_pct'] = (
        por_rama['n_abandono'] / por_rama['n_total'] * 100
    ).round(1)
    por_rama = por_rama.sort_values('tasa_pct', ascending=True)  # orden ascendente para barras horizontales

    # Asignamos color fijo por rama usando paleta Opción C
    colores_barras = [COLORES_RAMAS.get(r, COLORES['primario']) for r in por_rama['rama']]
    fig = px.bar(
        por_rama,
        x='tasa_pct',
        y='rama',
        orientation='h',
        color='rama',
        color_discrete_map=COLORES_RAMAS,
        text='tasa_pct',
        custom_data=['n_total', 'n_abandono'],
        labels={'tasa_pct': 'Tasa abandono (%)', 'rama': ''},
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        showlegend=True,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Tasa abandono: %{x:.1f}%<br>"
            "Alumnos totales: %{customdata[0]}<br>"
            "Abandonos: %{customdata[1]}<extra></extra>"
        )
    )
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=60, t=20, b=40),
        height=max(250, len(por_rama) * 55),  # altura dinámica según nº ramas
        xaxis=dict(range=[0, por_rama['tasa_pct'].max() * 1.15], showgrid=True, gridcolor=COLORES['borde']),
        yaxis=dict(showgrid=False),
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# BLOQUE 4: Top titulaciones
# =============================================================================

def _bloque_top_titulaciones(df: pd.DataFrame):
    """Tabla ordenable con las titulaciones y sus métricas de abandono."""

    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        🎓 Abandono por titulación
    </h4>
    """, unsafe_allow_html=True)

    if 'titulacion' not in df.columns or 'abandono' not in df.columns:
        st.info("No hay datos de titulación disponibles con los filtros actuales.")
        return

    # Controles: filtro rama + nº titulaciones (sin "Ordenar por" — se ordena pulsando cabecera)
    col_f, col_n = st.columns([2, 1])

    with col_f:
        col_rama_disp = 'rama_meta' if 'rama_meta' in df.columns else 'rama'
        ramas_tabla   = ["Todas"] + sorted(df[col_rama_disp].dropna().unique().tolist())
        rama_sel_t    = st.selectbox(
            "Filtrar por rama",
            options=ramas_tabla, index=0,
            key="rama_filtro_tabla",
            help="Filtra las titulaciones por rama de conocimiento"
        )

    with col_n:
        n_mostrar = st.slider(
            "Titulaciones",
            min_value=5,
            max_value=min(40, df['titulacion'].nunique()),
            value=10, step=5,
            help="Cuántas titulaciones ver"
        )

    df_t = df.copy()
    if rama_sel_t != "Todas":
        df_t = df_t[df_t[col_rama_disp] == rama_sel_t]

    col_rama_t2 = 'rama_meta' if 'rama_meta' in df_t.columns else 'rama'
    por_titulacion = (
        df_t.groupby(['titulacion', col_rama_t2] if col_rama_t2 in df_t.columns else ['titulacion'])
        .agg(
            Alumnos=('abandono', 'count'),
            Abandonos=('abandono', 'sum'),
            Riesgo_medio=('prob_abandono', 'mean')
        )
        .reset_index()
    )
    por_titulacion['tasa_pct'] = (por_titulacion['Abandonos'] / por_titulacion['Alumnos'] * 100).round(1)
    por_titulacion['riesgo_pct'] = (por_titulacion['Riesgo_medio'] * 100).round(1)
    por_titulacion = por_titulacion.sort_values('tasa_pct', ascending=False).head(n_mostrar)

    # Acortar nombres de titulación
    por_titulacion['titulacion_corta'] = (
        por_titulacion['titulacion']
        .str.replace(r'^Grado en ', '', regex=True)
        .str.replace(r'^Doble Grado en ', 'Doble: ', regex=True)
    )

    # Función semáforo
    def _color_tasa(v):
        if v < UMBRALES["riesgo_bajo"] * 100:
            return COLORES_RIESGO["bajo"], "🟢"
        elif v < UMBRALES["riesgo_medio"] * 100:
            return COLORES_RIESGO["medio"], "🟡"
        return COLORES_RIESGO["alto"], "🔴"

    # Construir tabla HTML propia
    filas_html = ""
    for _, row in por_titulacion.iterrows():
        color, emoji = _color_tasa(row['tasa_pct'])
        rama_txt = str(row.get(col_rama_t2, ""))
        tit = str(row['titulacion_corta'])

        # Barra de abandono
        barra_ab = f"""
        <div style="display:flex; align-items:center; gap:6px;">
            <span>{emoji}</span>
            <div style="flex:1; background:#eee; border-radius:4px; height:8px;">
                <div style="width:{min(row['tasa_pct'],100)}%; background:{color};
                            border-radius:4px; height:8px;"></div>
            </div>
            <span style="font-size:0.8rem; min-width:38px;">{row['tasa_pct']:.1f}%</span>
        </div>"""

        # Barra de riesgo predicho
        barra_riesgo = f"""
        <div style="display:flex; align-items:center; gap:6px;">
            <div style="flex:1; background:#eee; border-radius:4px; height:8px;">
                <div style="width:{min(row['riesgo_pct'],100)}%; background:{COLORES['primario']};
                            border-radius:4px; height:8px;"></div>
            </div>
            <span style="font-size:0.8rem; min-width:38px;">{row['riesgo_pct']:.1f}%</span>
        </div>"""

        filas_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding:8px 10px; font-size:0.82rem; line-height:1.3;
                       min-width:180px; max-width:240px; word-wrap:break-word;
                       white-space:normal;">{tit}</td>
            <td style="padding:8px 10px; font-size:0.8rem; color:#666;
                       min-width:120px; max-width:160px; word-wrap:break-word;
                       white-space:normal;">{rama_txt}</td>
            <td style="padding:8px 10px; font-size:0.82rem; text-align:right;">{int(row['Alumnos'])}</td>
            <td style="padding:8px 10px; font-size:0.82rem; text-align:right;">{int(row['Abandonos'])}</td>
            <td style="padding:8px 10px; min-width:160px;">{barra_ab}</td>
            <td style="padding:8px 10px; min-width:160px;">{barra_riesgo}</td>
        </tr>"""

    tabla_html = f"""
    <div style="overflow-x:auto; border:1px solid #e2e8f0; border-radius:8px;">
    <table style="width:100%; border-collapse:collapse; background:white;">
        <thead>
            <tr style="background:#f7fafc; border-bottom:2px solid #e2e8f0;">
                <th style="padding:10px; text-align:left; font-size:0.82rem;
                           color:#4a5568; font-weight:600;">Titulación</th>
                <th style="padding:10px; text-align:left; font-size:0.82rem;
                           color:#4a5568; font-weight:600;">Rama</th>
                <th style="padding:10px; text-align:right; font-size:0.82rem;
                           color:#4a5568; font-weight:600;">Alumnos</th>
                <th style="padding:10px; text-align:right; font-size:0.82rem;
                           color:#4a5568; font-weight:600;">Abandonos</th>
                <th style="padding:10px; text-align:left; font-size:0.82rem;
                           color:#4a5568; font-weight:600;">Abandono (%)</th>
                <th style="padding:10px; text-align:left; font-size:0.82rem;
                           color:#4a5568; font-weight:600;">Riesgo predicho (%)</th>
            </tr>
        </thead>
        <tbody>{filas_html}</tbody>
    </table>
    </div>
    """
    st.markdown(tabla_html, unsafe_allow_html=True)
    st.caption("💡 Haz clic en el filtro de rama para ver titulaciones por área de conocimiento.")


# =============================================================================
# BLOQUE 5: Distribución del riesgo predicho
# =============================================================================

def _bloque_distribucion_riesgo(df: pd.DataFrame):
    """Donut + histograma: distribución de probabilidades predichas."""

    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        🔮 Distribución del riesgo predicho
    </h4>
    """, unsafe_allow_html=True)

    if 'prob_abandono' not in df.columns or df['prob_abandono'].isna().all():
        st.info("No hay probabilidades predichas disponibles.")
        return

    col_donut, col_hist = st.columns([1, 2])

    with col_donut:
        # --- Gráfico donut: proporción bajo / medio / alto ---
        conteo_riesgo = df['nivel_riesgo'].value_counts()
        orden = ['Bajo', 'Medio', 'Alto']
        valores = [conteo_riesgo.get(nivel, 0) for nivel in orden]
        colores_donut = [COLORES_RIESGO['bajo'], COLORES_RIESGO['medio'], COLORES_RIESGO['alto']]

        fig_donut = go.Figure(go.Pie(
            labels=orden,
            values=valores,
            hole=0.55,             # el hueco central que hace el donut
            marker_colors=colores_donut,
            textinfo='percent',
            hovertemplate="<b>%{label}</b><br>%{value} alumnos (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.05),
            margin=dict(l=10, r=10, t=30, b=10),
            height=280,
            annotations=[dict(
                text=f"<b>{len(df):,}</b><br>alumnos",
                x=0.5, y=0.5,
                font_size=13,
                showarrow=False
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_hist:
        # --- Histograma: distribución completa de probabilidades ---
        fig_hist = px.histogram(
            df,
            x='prob_abandono',
            nbins=40,
            color_discrete_sequence=[COLORES['primario']],
            labels={'prob_abandono': 'Probabilidad de abandono predicha'},
            opacity=0.8,
        )

        # Colorear barras por nivel de riesgo usando COLORES_RIESGO
        import numpy as _np
        probs = df['prob_abandono'].dropna()
        bins  = _np.linspace(0, 1, 41)
        _, edges = _np.histogram(probs, bins=bins)
        nuevos_colores = []
        for e in edges[:-1]:
            mid = e + 0.0125
            if mid < UMBRALES['riesgo_bajo']:
                nuevos_colores.append(COLORES_RIESGO['bajo'])
            elif mid < UMBRALES['riesgo_medio']:
                nuevos_colores.append(COLORES_RIESGO['medio'])
            else:
                nuevos_colores.append(COLORES_RIESGO['alto'])
        fig_hist.update_traces(marker_color=nuevos_colores)

        # Líneas verticales para marcar los umbrales
        for umbral, color, etiqueta in [
            (UMBRALES['riesgo_bajo'],  COLORES_RIESGO['bajo'],  'Umbral bajo'),
            (UMBRALES['riesgo_medio'], COLORES_RIESGO['medio'], 'Umbral medio'),
        ]:
            fig_hist.add_vline(
                x=umbral,
                line_dash="dash",
                line_color=color,
                line_width=2,
                annotation_text=etiqueta,
                annotation_position="top right",
                annotation_font_size=11,
            )

        fig_hist.update_layout(
            xaxis=dict(range=[0, 1], tickformat='.0%'),
            yaxis_title="Nº alumnos",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=20, t=20, b=40),
            height=280,
            bargap=0.05,
        )
        fig_hist.update_xaxes(showgrid=True, gridcolor=COLORES['borde'])
        fig_hist.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            f"Las líneas de corte marcan los umbrales de riesgo: "
            f"bajo < {UMBRALES['riesgo_bajo']:.0%} · "
            f"medio < {UMBRALES['riesgo_medio']:.0%} · "
            f"alto ≥ {UMBRALES['riesgo_medio']:.0%}"
        )


# =============================================================================
# FIN DE p01_institucional.py
# =============================================================================
