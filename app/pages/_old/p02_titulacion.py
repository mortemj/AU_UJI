# =============================================================================
# p02_titulacion.py
# Pestaña 2 — Análisis por titulación
#
# ¿QUÉ HACE ESTE FICHERO?
#   Permite explorar en detalle el abandono en una titulación concreta.
#   El usuario selecciona su grado y toda la página se actualiza con
#   datos específicos de esa titulación.
#
# ESTRUCTURA DE LA PÁGINA:
#   0. Selector de titulación — filtro principal visible en la página
#   1. KPIs de la titulación vs media UJI
#   2. Evolución temporal con referencia UJI
#   3. Importancia de variables (SHAP) para esa titulación
#   4. Alumnos del test ordenados por riesgo
#   5. Perfil comparativo: abandona vs no abandona
#
# DATOS QUE USA:
#   - meta_test.parquet (vía loaders.cargar_meta_test)
#   - modelo + pipeline (vía loaders.cargar_modelo / cargar_pipeline)
#   - shap_global (vía loaders.cargar_shap_global) — opcional
#
# REQUISITOS:
#   - config_app.py accesible (un nivel arriba)
#   - utils/loaders.py disponible
#
# GENERA:
#   Página HTML interactiva renderizada por Streamlit. No genera ficheros.
#
# SIGUIENTE:
#   pages/p03_prospecto.py — pronóstico para alumnos antes de matricularse
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
_DIR_APP = Path(__file__).resolve().parent.parent
if str(_DIR_APP) not in sys.path:
    sys.path.insert(0, str(_DIR_APP))

from config_app import COLORES, NOMBRES_VARIABLES, UMBRALES
from utils.loaders import (cargar_meta_test, cargar_modelo, cargar_pipeline,
                           cargar_shap_global)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def show():
    """Renderiza la pestaña de análisis por titulación completa."""

    st.markdown(f"""
    <h2 style="color: {COLORES['primario']}; margin-bottom: 0.2rem;">
        📚 Análisis por titulación
    </h2>
    <p style="color: {COLORES['texto_suave']}; margin-top: 0; font-size: 0.95rem;">
        Explora el abandono en detalle para un grado universitario concreto
    </p>
    """, unsafe_allow_html=True)

    # --- Carga de datos ---
    with st.spinner("Cargando datos..."):
        try:
            df_raw   = cargar_meta_test()
            modelo   = cargar_modelo()
            pipeline = cargar_pipeline()
        except FileNotFoundError as e:
            st.error(f"❌ No se pudieron cargar los datos:\n\n{e}")
            st.stop()

    shap_values = cargar_shap_global()  # opcional, puede ser None

    # --- Añadir probabilidades y nivel de riesgo ---
    df = _añadir_probabilidades(df_raw, modelo, pipeline)
    df = _añadir_nivel_riesgo(df)

    # --- Calcular métricas globales UJI (referencia para comparativas) ---
    tasa_uji = (df['abandono'].sum() / len(df) * 100) if 'abandono' in df.columns else 0
    prob_uji  = df['prob_abandono'].mean() * 100 if 'prob_abandono' in df.columns else 0

    # --- Selector de titulación ---
    titulacion_sel = _selector_titulacion(df)
    if titulacion_sel is None:
        st.info("Selecciona una titulación para ver el análisis detallado.")
        return

    # --- Filtrar datos por titulación seleccionada ---
    df_tit = df[df['titulacion'] == titulacion_sel].copy()

    if len(df_tit) < 5:
        st.warning(
            f"⚠️ La titulación **{titulacion_sel}** solo tiene {len(df_tit)} registros "
            "en el conjunto de test. Los resultados pueden no ser representativos."
        )

    st.divider()

    # --- Bloques de contenido ---
    _bloque_kpis_titulacion(df_tit, tasa_uji, prob_uji, titulacion_sel)
    st.divider()
    _bloque_evolucion_titulacion(df_tit, df, titulacion_sel)
    st.divider()
    _bloque_importancia_variables(df_tit, shap_values, titulacion_sel)
    st.divider()
    _bloque_alumnos_riesgo(df_tit, titulacion_sel)
    st.divider()
    _bloque_perfil_comparativo(df_tit, titulacion_sel)


# =============================================================================
# SELECTOR DE TITULACIÓN
# =============================================================================

def _selector_titulacion(df: pd.DataFrame) -> str | None:
    """
    Muestra un desplegable con todas las titulaciones disponibles.
    Lo ponemos en la página principal (no en sidebar) porque es el
    filtro central de esta pestaña — el usuario debe verlo claramente.

    Devuelve el nombre de la titulación seleccionada, o None si no hay datos.
    """
    if 'titulacion' not in df.columns:
        st.error("❌ La columna 'titulacion' no está disponible en meta_test.")
        return None

    titulaciones = sorted(df['titulacion'].dropna().unique().tolist())

    if not titulaciones:
        st.error("❌ No se encontraron titulaciones en los datos.")
        return None

    # Añadimos información de rama junto al nombre de la titulación
    # para facilitar la búsqueda cuando hay muchas opciones
    col_sel, col_info = st.columns([2, 1])

    with col_sel:
        titulacion_sel = st.selectbox(
            label="🎓 Selecciona una titulación",
            options=titulaciones,
            index=0,
            help=(
                "Selecciona el grado que quieres analizar. "
                "Puedes escribir para buscar dentro de la lista."
            )
        )

    with col_info:
        # Mostramos la rama y nº de alumnos de la titulación seleccionada
        if 'rama' in df.columns:
            rama = df[df['titulacion'] == titulacion_sel]['rama'].iloc[0] \
                if len(df[df['titulacion'] == titulacion_sel]) > 0 else "—"
            n_alumnos = len(df[df['titulacion'] == titulacion_sel])
            st.markdown(f"""
            <div style="
                background: {COLORES['fondo']};
                border-left: 3px solid {COLORES['primario']};
                border-radius: 4px;
                padding: 0.6rem 1rem;
                margin-top: 1.6rem;
                font-size: 0.85rem;
            ">
                <strong>Rama:</strong> {rama}<br>
                <strong>Alumnos en test:</strong> {n_alumnos:,}
            </div>
            """, unsafe_allow_html=True)

    return titulacion_sel


# =============================================================================
# BLOQUE 1: KPIs de la titulación vs media UJI
# =============================================================================

def _bloque_kpis_titulacion(df_tit: pd.DataFrame, tasa_uji: float,
                             prob_uji: float, titulo: str):
    """
    Métricas clave de la titulación con comparativa respecto a la media UJI.
    El delta (flecha arriba/abajo) muestra si está por encima o debajo de la media.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        📌 Indicadores de <em>{titulo}</em>
    </h4>
    """, unsafe_allow_html=True)

    n_total       = len(df_tit)
    n_abandono    = df_tit['abandono'].sum() if 'abandono' in df_tit.columns else 0
    tasa_tit      = (n_abandono / n_total * 100) if n_total > 0 else 0
    prob_media    = df_tit['prob_abandono'].mean() * 100 \
                    if 'prob_abandono' in df_tit.columns else 0
    n_riesgo_alto = (df_tit['nivel_riesgo'] == 'Alto').sum()
    pct_alto      = (n_riesgo_alto / n_total * 100) if n_total > 0 else 0

    # delta = diferencia respecto a la media UJI
    # Positivo = peor que la media (más abandono) → rojo
    # Negativo = mejor que la media (menos abandono) → verde
    delta_tasa = tasa_tit - tasa_uji
    delta_prob = prob_media - prob_uji

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="👥 Alumnos en test",
            value=f"{n_total:,}",
            help="Número de registros de esta titulación en el conjunto de test"
        )
    with c2:
        st.metric(
            label="📉 Abandono real",
            value=f"{tasa_tit:.1f} %",
            delta=f"{delta_tasa:+.1f} pp vs UJI ({tasa_uji:.1f}%)",
            delta_color="inverse",  # inverse: positivo=rojo, negativo=verde
            help="Tasa de abandono observada. El delta compara con la media de toda la UJI."
        )
    with c3:
        st.metric(
            label="🔮 Riesgo predicho medio",
            value=f"{prob_media:.1f} %",
            delta=f"{delta_prob:+.1f} pp vs UJI",
            delta_color="inverse",
            help="Probabilidad media de abandono según el modelo para esta titulación"
        )
    with c4:
        st.metric(
            label="🔴 En riesgo alto",
            value=f"{n_riesgo_alto:,}",
            delta=f"{pct_alto:.1f} % del total",
            delta_color="inverse",
            help=f"Alumnos con probabilidad predicha ≥ {UMBRALES['riesgo_medio']:.0%}"
        )


# =============================================================================
# BLOQUE 2: Evolución temporal con referencia UJI
# =============================================================================

def _bloque_evolucion_titulacion(df_tit: pd.DataFrame, df_total: pd.DataFrame,
                                  titulo: str):
    """
    Gráfico de línea con la evolución temporal de la titulación.
    Incluye línea de referencia con la media UJI para contextualizar.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        📈 Evolución temporal
    </h4>
    """, unsafe_allow_html=True)

    if 'anio_cohorte' not in df_tit.columns or 'abandono' not in df_tit.columns:
        st.info("No hay datos de cohorte disponibles para esta titulación.")
        return

    # Evolución de la titulación
    ev_tit = (
        df_tit.groupby('anio_cohorte')
        .agg(n=('abandono', 'count'), ab=('abandono', 'sum'))
        .reset_index()
    )
    ev_tit['tasa'] = (ev_tit['ab'] / ev_tit['n'] * 100).round(1)

    # Evolución media UJI (referencia)
    ev_uji = (
        df_total.groupby('anio_cohorte')
        .agg(n=('abandono', 'count'), ab=('abandono', 'sum'))
        .reset_index()
    )
    ev_uji['tasa'] = (ev_uji['ab'] / ev_uji['n'] * 100).round(1)

    fig = go.Figure()

    # Línea de referencia UJI (fondo, gris)
    fig.add_trace(go.Scatter(
        x=ev_uji['anio_cohorte'],
        y=ev_uji['tasa'],
        mode='lines',
        name='Media UJI',
        line=dict(color=COLORES['texto_suave'], width=1.5, dash='dot'),
        hovertemplate="<b>UJI · Cohorte %{x}</b><br>Tasa: %{y:.1f}%<extra></extra>",
    ))

    # Línea de la titulación (primer plano, color destacado)
    fig.add_trace(go.Scatter(
        x=ev_tit['anio_cohorte'],
        y=ev_tit['tasa'],
        mode='lines+markers',
        name=titulo,
        line=dict(color=COLORES['abandono'], width=2.5),
        marker=dict(size=8),
        hovertemplate=(
            f"<b>{titulo} · Cohorte %{{x}}</b><br>"
            "Abandono: %{y:.1f}%<br>"
            "N alumnos: %{customdata}<extra></extra>"
        ),
        customdata=ev_tit['n']
    ))

    fig.update_layout(
        xaxis_title="Año de cohorte",
        yaxis_title="Tasa de abandono (%)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORES['borde'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# BLOQUE 3: Importancia de variables para esta titulación
# =============================================================================

def _bloque_importancia_variables(df_tit: pd.DataFrame, shap_values,
                                   titulo: str):
    """
    Muestra qué variables influyen más en el abandono para esta titulación.

    Si hay valores SHAP disponibles (Fase 6 ejecutada), los usa filtrados
    por titulación. Si no, calcula una proxy simple con diferencia de medias
    entre alumnos que abandonan y los que no.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        🔍 Variables más influyentes
    </h4>
    """, unsafe_allow_html=True)

    # Columnas numéricas disponibles (features del modelo)
    cols_meta   = ['abandono', 'titulacion', 'rama', 'anio_cohorte',
                   'sexo', 'nivel_riesgo', 'prob_abandono']
    cols_num    = [
        c for c in df_tit.select_dtypes(include=[np.number]).columns
        if c not in cols_meta
    ]

    if len(cols_num) == 0:
        st.info("No hay variables numéricas disponibles para este análisis.")
        return

    if 'abandono' not in df_tit.columns or df_tit['abandono'].nunique() < 2:
        st.info("No hay suficiente variabilidad en la variable abandono para esta titulación.")
        return

    # --- Proxy de importancia: diferencia de medias estandarizada ---
    # Para cada variable numérica calculamos cuánto difiere la media
    # entre los que abandonan y los que no. Es una aproximación sencilla
    # cuando no hay SHAP disponible, y también útil como complemento.
    abandona   = df_tit[df_tit['abandono'] == 1][cols_num]
    no_abandona = df_tit[df_tit['abandono'] == 0][cols_num]

    importancia = []
    for col in cols_num:
        media_ab  = abandona[col].mean()
        media_no  = no_abandona[col].mean()
        std_total = df_tit[col].std()
        if std_total > 0:
            efecto = abs(media_ab - media_no) / std_total  # d de Cohen simplificado
        else:
            efecto = 0
        nombre_legible = NOMBRES_VARIABLES.get(col, col.replace('_', ' ').title())
        importancia.append({
            'variable':    col,
            'nombre':      nombre_legible,
            'efecto':      round(efecto, 3),
            'media_ab':    round(media_ab, 2),
            'media_no':    round(media_no, 2),
            'diferencia':  round(media_ab - media_no, 2),
        })

    df_imp = (
        pd.DataFrame(importancia)
        .sort_values('efecto', ascending=True)  # ascendente para barras horizontales
        .tail(10)                               # top 10
    )

    # Color según si abandonar implica valor más alto o más bajo
    # (diferencia positiva = el que abandona tiene más → puede ser negativo como beca)
    colores_barras = [
        COLORES['abandono'] if d > 0 else COLORES['primario']
        for d in df_imp['diferencia']
    ]

    fig = go.Figure(go.Bar(
        x=df_imp['efecto'],
        y=df_imp['nombre'],
        orientation='h',
        marker_color=colores_barras,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Efecto: %{x:.3f}<br>"
            "Media abandona: %{customdata[0]}<br>"
            "Media no abandona: %{customdata[1]}<extra></extra>"
        ),
        customdata=df_imp[['media_ab', 'media_no']].values,
    ))

    fig.update_layout(
        xaxis_title="Tamaño del efecto (diferencia de medias estandarizada)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=40),
        height=350,
        xaxis=dict(showgrid=True, gridcolor=COLORES['borde']),
        yaxis=dict(showgrid=False),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Nota metodológica
    metodo = "SHAP" if shap_values is not None else "diferencia de medias estandarizada"
    st.caption(
        f"💡 Importancia calculada mediante {metodo}. "
        "Rojo: el alumno que abandona tiene valores más altos. "
        "Azul: el alumno que abandona tiene valores más bajos."
    )


# =============================================================================
# BLOQUE 4: Tabla de alumnos ordenados por riesgo
# =============================================================================

def _bloque_alumnos_riesgo(df_tit: pd.DataFrame, titulo: str):
    """
    Tabla con los alumnos de la titulación del conjunto de test,
    ordenados de mayor a menor probabilidad de abandono predicha.
    Útil para que el coordinador identifique los casos más críticos.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        👥 Alumnos por nivel de riesgo
    </h4>
    """, unsafe_allow_html=True)

    # Filtro por nivel de riesgo encima de la tabla
    col_filtro, col_info = st.columns([1, 2])
    with col_filtro:
        nivel_filtro = st.selectbox(
            label="Mostrar alumnos en riesgo",
            options=["Todos", "Alto", "Medio", "Bajo"],
            index=0,
            key="filtro_nivel_p02",   # key única para evitar conflictos con otros widgets
            help="Filtra la tabla por nivel de riesgo predicho"
        )

    df_tabla = df_tit.copy()
    if nivel_filtro != "Todos":
        df_tabla = df_tabla[df_tabla['nivel_riesgo'] == nivel_filtro]

    with col_info:
        st.markdown(f"""
        <div style="font-size: 0.82rem; color: {COLORES['texto_suave']}; margin-top: 1.5rem;">
            Mostrando <strong>{len(df_tabla):,}</strong> alumnos
            {'en riesgo ' + nivel_filtro.lower() if nivel_filtro != 'Todos' else 'en total'}
        </div>
        """, unsafe_allow_html=True)

    # Seleccionamos columnas a mostrar (solo las que existen)
    cols_mostrar_candidatas = [
        'anio_cohorte', 'sexo', 'nivel_riesgo', 'prob_abandono', 'abandono'
    ]
    # Añadimos features numéricas clave si existen
    for col in ['nota_acceso', 'nota_1er_anio', 'n_anios_beca', 'tasa_rendimiento']:
        if col in df_tabla.columns:
            cols_mostrar_candidatas.append(col)

    cols_mostrar = [c for c in cols_mostrar_candidatas if c in df_tabla.columns]

    tabla = (
        df_tabla[cols_mostrar]
        .sort_values('prob_abandono', ascending=False)
        .rename(columns={
            'anio_cohorte':    'Cohorte',
            'sexo':            'Sexo',
            'nivel_riesgo':    'Riesgo',
            'prob_abandono':   'Prob. abandono',
            'abandono':        'Abandonó (real)',
            'nota_acceso':     'Nota acceso',
            'nota_1er_anio':   'Nota 1er año',
            'n_anios_beca':    'Años beca',
            'tasa_rendimiento': 'Tasa rendimiento',
        })
    )

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={
            "Prob. abandono": st.column_config.ProgressColumn(
                "Prob. abandono",
                min_value=0.0,
                max_value=1.0,
                format="%.1%",
                help="Probabilidad de abandono predicha por el modelo (0-1)"
            ),
            "Abandonó (real)": st.column_config.CheckboxColumn(
                "Abandonó (real)",
                help="Valor real observado en los datos históricos"
            ),
        }
    )

    st.caption(
        "💡 Haz clic en cualquier encabezado de columna para ordenar. "
        "La columna 'Abandonó (real)' muestra lo que ocurrió realmente, "
        "no la predicción del modelo."
    )


# =============================================================================
# BLOQUE 5: Perfil comparativo abandona vs no abandona
# =============================================================================

def _bloque_perfil_comparativo(df_tit: pd.DataFrame, titulo: str):
    """
    Compara el perfil medio del alumno que abandona vs el que no abandona
    en esta titulación, para las variables numéricas más relevantes.
    Gráfico de barras agrupadas.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        ⚖️ Perfil comparativo: abandona vs no abandona
    </h4>
    """, unsafe_allow_html=True)

    if 'abandono' not in df_tit.columns or df_tit['abandono'].nunique() < 2:
        st.info("No hay suficientes datos de ambos grupos para esta titulación.")
        return

    # Variables a comparar — solo las que existen en el DataFrame
    vars_comparar = [
        'nota_acceso', 'nota_1er_anio', 'n_anios_beca',
        'creditos_superados', 'tasa_rendimiento'
    ]
    vars_disponibles = [v for v in vars_comparar if v in df_tit.columns]

    if not vars_disponibles:
        st.info("No hay variables de perfil disponibles para esta comparativa.")
        return

    # Calculamos medias por grupo
    grupos = df_tit.groupby('abandono')[vars_disponibles].mean().round(2)

    # Normalizamos a escala 0-100 para que sean comparables en el mismo gráfico
    # (nota_acceso va de 0-14, n_anios_beca de 0-6, etc. — escalas muy distintas)
    grupos_norm = grupos.copy()
    for col in vars_disponibles:
        col_max = df_tit[col].max()
        col_min = df_tit[col].min()
        rango = col_max - col_min
        if rango > 0:
            grupos_norm[col] = (grupos[col] - col_min) / rango * 100
        else:
            grupos_norm[col] = 50  # si no hay variación, ponemos el centro

    # Nombres legibles para el eje X
    nombres_eje = [NOMBRES_VARIABLES.get(v, v.replace('_', ' ').title())
                   for v in vars_disponibles]

    fig = go.Figure()

    # Barra grupo "No abandona" (azul)
    fig.add_trace(go.Bar(
        name='No abandona',
        x=nombres_eje,
        y=grupos_norm.loc[0].values if 0 in grupos_norm.index else [],
        marker_color=COLORES['primario'],
        opacity=0.85,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "No abandona: %{customdata:.2f} (valor real)<extra></extra>"
        ),
        customdata=grupos.loc[0].values if 0 in grupos.index else [],
    ))

    # Barra grupo "Abandona" (rojo)
    fig.add_trace(go.Bar(
        name='Abandona',
        x=nombres_eje,
        y=grupos_norm.loc[1].values if 1 in grupos_norm.index else [],
        marker_color=COLORES['abandono'],
        opacity=0.85,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Abandona: %{customdata:.2f} (valor real)<extra></extra>"
        ),
        customdata=grupos.loc[1].values if 1 in grupos.index else [],
    ))

    fig.update_layout(
        barmode='group',          # barras agrupadas, no apiladas
        yaxis_title="Valor normalizado (0–100)",
        yaxis=dict(range=[0, 110]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=80),
        height=380,
    )
    fig.update_xaxes(showgrid=False, tickangle=-20)
    fig.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💡 Los valores están normalizados (0–100) para que sean comparables "
        "entre variables con escalas distintas. El tooltip muestra el valor real. "
        "Una barra azul más alta = ese factor protege contra el abandono."
    )


# =============================================================================
# HELPERS DE DATOS (iguales que en p01 — en el futuro se pueden mover a utils/)
# =============================================================================

def _añadir_probabilidades(df_raw: pd.DataFrame, modelo, pipeline) -> pd.DataFrame:
    """Añade columna 'prob_abandono' con la predicción del modelo."""
    df = df_raw.copy()
    cols_meta = ['abandono', 'titulacion', 'rama', 'anio_cohorte', 'sexo']
    cols_meta_presentes = [c for c in cols_meta if c in df.columns]
    cols_features = [c for c in df.columns if c not in cols_meta_presentes]
    X = df[cols_features]
    try:
        X_prep = pipeline.transform(X)
        df['prob_abandono'] = modelo.predict_proba(X_prep)[:, 1]
    except Exception as e:
        st.warning(f"⚠️ No se pudieron calcular las probabilidades: {e}")
        df['prob_abandono'] = np.nan
    return df


def _añadir_nivel_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columna 'nivel_riesgo' con valores Bajo / Medio / Alto."""
    df = df.copy()
    def _clasificar(prob):
        if pd.isna(prob):           return 'Desconocido'
        if prob < UMBRALES['riesgo_bajo']:  return 'Bajo'
        if prob < UMBRALES['riesgo_medio']: return 'Medio'
        return 'Alto'
    df['nivel_riesgo'] = df['prob_abandono'].apply(_clasificar)
    return df


# =============================================================================
# FIN DE p02_titulacion.py
# =============================================================================
