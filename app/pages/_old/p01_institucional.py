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
_DIR_APP = Path(__file__).resolve().parent.parent  # sube de pages/ a app/
if str(_DIR_APP) not in sys.path:
    sys.path.insert(0, str(_DIR_APP))

from config_app import COLORES, NOMBRES_VARIABLES, UMBRALES
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

    # Nota aclaratoria al pie
    st.markdown(f"""
    <div style="font-size: 0.75rem; color: {COLORES['texto_suave']}; margin-top: 1rem;">
        ℹ️ <em>Los datos mostrados corresponden al conjunto de evaluación del modelo
        (6.725 observaciones). Las probabilidades de riesgo son predicciones del modelo
        Stacking entrenado en Fase 5, no valores reales observados.</em>
    </div>
    """, unsafe_allow_html=True)


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

    # Columnas que son metadatos (no features del modelo)
    # Estas columnas existen en meta_test pero el pipeline no las conoce
    cols_meta = ['abandono', 'titulacion', 'rama', 'anio_cohorte', 'sexo']
    cols_meta_presentes = [c for c in cols_meta if c in df.columns]

    # Features: todo lo que no es metadato
    cols_features = [c for c in df.columns if c not in cols_meta_presentes]
    X = df[cols_features]

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
    Muestra los filtros en la barra lateral y devuelve el DataFrame filtrado.

    Los filtros son encadenados: cada uno actúa sobre el resultado del anterior.
    El orden es: sexo → rama → año de cohorte.
    Cambiar cualquier filtro actualiza todos los bloques de la página.
    """
    with st.sidebar:
        st.divider()
        st.markdown(f"""
        <p style="font-size: 0.82rem; font-weight: bold; color: {COLORES['texto']};">
            🔽 Filtros — Visión institucional
        </p>
        """, unsafe_allow_html=True)

        # --- Filtro 1: Sexo ---
        # st.selectbox() crea un desplegable de selección única
        opciones_sexo = ["Todos"] + sorted(df['sexo'].dropna().unique().tolist()) \
            if 'sexo' in df.columns else ["Todos"]

        sexo_sel = st.selectbox(
            label="Sexo",
            options=opciones_sexo,
            index=0,   # empieza en "Todos"
            help="Filtra por sexo del estudiante"
        )

        # Aplicamos filtro de sexo al DataFrame
        df_f = df.copy()
        if sexo_sel != "Todos":
            df_f = df_f[df_f['sexo'] == sexo_sel]

        # --- Filtro 2: Rama de conocimiento ---
        # st.multiselect() permite seleccionar varias opciones a la vez.
        # Por defecto todas seleccionadas (lista completa).
        ramas_disponibles = sorted(df_f['rama'].dropna().unique().tolist()) \
            if 'rama' in df_f.columns else []

        ramas_sel = st.multiselect(
            label="Rama de conocimiento",
            options=ramas_disponibles,
            default=ramas_disponibles,  # todas seleccionadas por defecto
            help="Puedes seleccionar una o varias ramas. "
                 "Se aplica sobre el filtro de sexo ya activo."
        )

        if ramas_sel:
            df_f = df_f[df_f['rama'].isin(ramas_sel)]

        # --- Filtro 3: Rango de años de cohorte ---
        # st.slider() con dos valores crea un rango (mínimo, máximo).
        if 'anio_cohorte' in df_f.columns and df_f['anio_cohorte'].notna().any():
            anio_min = int(df_f['anio_cohorte'].min())
            anio_max = int(df_f['anio_cohorte'].max())

            # Solo mostramos el slider si hay más de un año disponible
            if anio_min < anio_max:
                rango_anios = st.slider(
                    label="Años de cohorte",
                    min_value=anio_min,
                    max_value=anio_max,
                    value=(anio_min, anio_max),  # rango completo por defecto
                    step=1,
                    help="Filtra por año de inicio de estudios del alumno"
                )
                df_f = df_f[
                    (df_f['anio_cohorte'] >= rango_anios[0]) &
                    (df_f['anio_cohorte'] <= rango_anios[1])
                ]

        # Contador de registros activos — útil para que el usuario vea
        # cuántos datos quedan con los filtros aplicados
        st.markdown(f"""
        <div style="
            font-size: 0.78rem;
            color: {COLORES['texto_suave']};
            background: {COLORES['fondo']};
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin-top: 0.5rem;
        ">
            📋 <strong>{len(df_f):,}</strong> registros seleccionados<br>
            de {len(df):,} totales
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
    n_ramas        = df['rama'].nunique() if 'rama' in df.columns else 0

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

    if 'anio_cohorte' not in df.columns or 'abandono' not in df.columns:
        st.info("No hay datos de cohorte disponibles con los filtros actuales.")
        return

    # Agrupamos por año y calculamos tasa de abandono
    evolucion = (
        df.groupby('anio_cohorte')
        .agg(
            n_total=('abandono', 'count'),
            n_abandono=('abandono', 'sum'),
            prob_media=('prob_abandono', 'mean')
        )
        .reset_index()
    )
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

    fig.update_layout(
        xaxis_title="Año de cohorte",
        yaxis_title="Porcentaje (%)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=380,
        hovermode="x unified",   # al pasar el ratón muestra ambas líneas a la vez
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORES['borde'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

    # use_container_width=True → el gráfico ocupa todo el ancho disponible
    st.plotly_chart(fig, use_container_width=True)


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

    if 'rama' not in df.columns or 'abandono' not in df.columns:
        st.info("No hay datos de rama disponibles con los filtros actuales.")
        return

    por_rama = (
        df.groupby('rama')
        .agg(
            n_total=('abandono', 'count'),
            n_abandono=('abandono', 'sum'),
            prob_media=('prob_abandono', 'mean')
        )
        .reset_index()
    )
    por_rama['tasa_pct'] = (
        por_rama['n_abandono'] / por_rama['n_total'] * 100
    ).round(1)
    por_rama = por_rama.sort_values('tasa_pct', ascending=True)  # orden ascendente para barras horizontales

    fig = px.bar(
        por_rama,
        x='tasa_pct',
        y='rama',
        orientation='h',           # barras horizontales
        color='tasa_pct',
        color_continuous_scale=[   # degradado: verde (bajo) → rojo (alto)
            [0.0, COLORES['exito']],
            [0.5, COLORES['advertencia']],
            [1.0, COLORES['abandono']]
        ],
        text='tasa_pct',
        custom_data=['n_total', 'n_abandono'],
        labels={'tasa_pct': 'Tasa abandono (%)', 'rama': ''},
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Tasa abandono: %{x:.1f}%<br>"
            "Alumnos totales: %{customdata[0]}<br>"
            "Abandonos: %{customdata[1]}<extra></extra>"
        )
    )
    fig.update_layout(
        coloraxis_showscale=False,  # ocultamos la leyenda de color (ya está en el texto)
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=60, t=20, b=40),
        height=max(250, len(por_rama) * 55),  # altura dinámica según nº ramas
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor=COLORES['borde']),
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

    # Control para mostrar N titulaciones
    # st.columns() aquí nos sirve para poner el slider al lado del título
    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        n_mostrar = st.slider(
            "Nº titulaciones a mostrar",
            min_value=5,
            max_value=min(40, df['titulacion'].nunique()),
            value=10,
            step=5,
            help="Ajusta cuántas titulaciones ver en la tabla"
        )

    por_titulacion = (
        df.groupby(['titulacion', 'rama'] if 'rama' in df.columns else ['titulacion'])
        .agg(
            Alumnos=('abandono', 'count'),
            Abandonos=('abandono', 'sum'),
            Riesgo_medio=('prob_abandono', 'mean')
        )
        .reset_index()
    )
    por_titulacion['Tasa_abandono_pct'] = (
        por_titulacion['Abandonos'] / por_titulacion['Alumnos'] * 100
    ).round(1)
    por_titulacion['Riesgo_medio_pct'] = (
        por_titulacion['Riesgo_medio'] * 100
    ).round(1)

    # Ordenamos por tasa de abandono descendente y tomamos las N primeras
    tabla = (
        por_titulacion
        .sort_values('Tasa_abandono_pct', ascending=False)
        .head(n_mostrar)
        .rename(columns={
            'titulacion':         'Titulación',
            'rama':               'Rama',
            'Tasa_abandono_pct':  'Abandono real (%)',
            'Riesgo_medio_pct':   'Riesgo predicho medio (%)',
        })
        [['Titulación', 'Rama', 'Alumnos', 'Abandonos',
          'Abandono real (%)', 'Riesgo predicho medio (%)']]
        if 'rama' in df.columns else
        por_titulacion
        .sort_values('Tasa_abandono_pct', ascending=False)
        .head(n_mostrar)
        .rename(columns={
            'titulacion':        'Titulación',
            'Tasa_abandono_pct': 'Abandono real (%)',
            'Riesgo_medio_pct':  'Riesgo predicho medio (%)',
        })
        [['Titulación', 'Alumnos', 'Abandonos',
          'Abandono real (%)', 'Riesgo predicho medio (%)']]
    )

    # st.dataframe() con column_config permite personalizar cada columna:
    # barras de progreso, formatos, anchos, etc.
    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Abandono real (%)": st.column_config.ProgressColumn(
                "Abandono real (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                help="Porcentaje de abandono observado en los datos reales"
            ),
            "Riesgo predicho medio (%)": st.column_config.ProgressColumn(
                "Riesgo predicho medio (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                help="Probabilidad media de abandono según el modelo"
            ),
        }
    )

    st.caption(
        "💡 Haz clic en el encabezado de cualquier columna para ordenar la tabla. "
        "Los filtros de la barra lateral afectan a estas titulaciones."
    )


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
        colores_donut = [COLORES['exito'], COLORES['advertencia'], COLORES['abandono']]

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
            nbins=40,              # número de barras del histograma
            color_discrete_sequence=[COLORES['primario']],
            labels={'prob_abandono': 'Probabilidad de abandono predicha'},
            opacity=0.8,
        )

        # Líneas verticales para marcar los umbrales
        for umbral, color, etiqueta in [
            (UMBRALES['riesgo_bajo'],  COLORES['exito'],      'Umbral bajo'),
            (UMBRALES['riesgo_medio'], COLORES['advertencia'], 'Umbral medio'),
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
