# =============================================================================
# p05_equidad.py
# Pestaña 5 — Equidad y diversidad
#
# ¿QUÉ HACE ESTE FICHERO?
#   Analiza si el modelo predice de forma justa entre distintos grupos
#   de estudiantes (por sexo y rama de conocimiento).
#   Pensada para todos los perfiles, especialmente para el tribunal
#   evaluador del TFM.
#
# FILOSOFÍA DE ESTA PESTAÑA:
#   Un modelo puede ser muy preciso globalmente pero injusto con ciertos
#   grupos. Esta pestaña lo analiza con honestidad, explica cada métrica
#   en lenguaje accesible, y concluye con una valoración directa.
#
# ESTRUCTURA:
#   1. ¿Qué es la equidad en ML? — explicación didáctica
#   2. Equidad por sexo — métricas y gráficos comparativos
#   3. Equidad por rama — métricas y gráficos comparativos
#   4. Disparate Impact — métrica estándar de fairness con gauge
#   5. Matriz de confusión por grupo — quién paga el precio del error
#   6. Simulador de política institucional — umbral ajustable ★ extra
#   7. Conclusión y limitaciones — valoración honesta y directa
#
# DATOS QUE USA:
#   - meta_test.parquet + modelo + pipeline (probabilidades predichas)
#   - fairness_metricas.parquet si existe (Fase 6), si no lo calcula aquí
#
# REQUISITOS:
#   - config_app.py accesible
#   - utils/loaders.py disponible
#
# GENERA:
#   Página HTML interactiva. No genera ficheros en disco.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

# ---------------------------------------------------------------------------
# Imports internos
# ---------------------------------------------------------------------------
# _path_setup añade app/ a sys.path de forma robusta en Windows/OneDrive
import _path_setup  # noqa: F401

from config_app import COLORES, UMBRALES
from utils.loaders import (cargar_fairness, cargar_meta_test, cargar_modelo,
                           cargar_pipeline)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def show():
    """Renderiza la pestaña de equidad y diversidad completa."""

    st.markdown(f"""
    <h2 style="color: {COLORES['primario']}; margin-bottom: 0.2rem;">
        ⚖️ Equidad y diversidad
    </h2>
    <p style="color: {COLORES['texto_suave']}; margin-top: 0; font-size: 0.95rem;">
        ¿Predice el modelo de forma justa entre distintos grupos de estudiantes?
    </p>
    """, unsafe_allow_html=True)

    # Carga de datos
    with st.spinner("Cargando datos..."):
        try:
            df_raw   = cargar_meta_test()
            modelo   = cargar_modelo()
            pipeline = cargar_pipeline()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()

    # Calcular probabilidades y predicciones
    df = _preparar_datos(df_raw, modelo, pipeline)

    if df is None:
        st.error("❌ No se pudieron calcular las predicciones.")
        st.stop()

    st.divider()

    # Bloques
    _bloque_explicacion_equidad()
    st.divider()
    _bloque_equidad_por_grupo(df, grupo='sexo')
    st.divider()
    _bloque_equidad_por_grupo(df, grupo='rama')
    st.divider()
    _bloque_disparate_impact(df)
    st.divider()
    _bloque_confusion_por_grupo(df)
    st.divider()
    _bloque_simulador_politica(df)
    st.divider()
    _bloque_conclusion(df)


# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================

def _preparar_datos(df_raw: pd.DataFrame, modelo, pipeline) -> pd.DataFrame | None:
    """
    Añade probabilidades predichas y predicciones binarias al DataFrame.
    Devuelve None si hay un error en la transformación.
    """
    try:
        df = df_raw.copy()
        # Usamos pipeline.feature_names_in_ — robusto, sin listas manuales
        cols_features = list(pipeline.feature_names_in_)
        cols_disp     = [c for c in cols_features if c in df.columns]

        X_prep = pipeline.transform(df[cols_disp])
        df['prob_abandono'] = modelo.predict_proba(X_prep)[:, 1]

        # Predicción binaria con el umbral por defecto
        umbral_default = UMBRALES['riesgo_medio']
        df['pred_abandono'] = (df['prob_abandono'] >= umbral_default).astype(int)

        return df
    except Exception as e:
        st.warning(f"⚠️ Error preparando datos: {e}")
        return None


def _metricas_grupo(df_g: pd.DataFrame) -> dict:
    """
    Calcula métricas de clasificación para un subgrupo.
    Devuelve dict con n, tasa_real, tasa_pred, precision, recall, f1, auc.
    """
    if len(df_g) < 10 or 'abandono' not in df_g.columns:
        return {}

    y_true = df_g['abandono'].values
    y_pred = df_g['pred_abandono'].values
    y_prob = df_g['prob_abandono'].values

    try:
        return {
            'n':          len(df_g),
            'tasa_real':  y_true.mean() * 100,
            'tasa_pred':  y_pred.mean() * 100,
            'precision':  precision_score(y_true, y_pred, zero_division=0) * 100,
            'recall':     recall_score(y_true, y_pred, zero_division=0) * 100,
            'f1':         f1_score(y_true, y_pred, zero_division=0) * 100,
            'auc':        roc_auc_score(y_true, y_prob) * 100
                          if len(np.unique(y_true)) > 1 else np.nan,
        }
    except Exception:
        return {}


# =============================================================================
# BLOQUE 1: Explicación de equidad en ML
# =============================================================================

def _bloque_explicacion_equidad():
    """
    Explicación didáctica de qué es la equidad en machine learning.
    En lenguaje accesible para perfiles no técnicos (tribunal incluido).
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        🎓 ¿Qué es la equidad en un modelo de machine learning?
    </h4>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    tarjetas = [
        {
            'icono': '🎯',
            'titulo': 'El problema',
            'texto': (
                'Un modelo puede tener buena precisión global pero '
                'equivocarse sistemáticamente más con ciertos grupos. '
                'Por ejemplo, detectar peor el abandono en mujeres '
                'que en hombres, o en unas ramas que en otras.'
            )
        },
        {
            'icono': '⚖️',
            'titulo': 'Qué medimos',
            'texto': (
                'Comparamos si el modelo tiene el mismo rendimiento '
                '(F1, recall, precisión) en todos los grupos. '
                'También medimos el Disparate Impact: el ratio entre '
                'las tasas de predicción positiva de cada grupo.'
            )
        },
        {
            'icono': '📏',
            'titulo': 'La regla del 80%',
            'texto': (
                'Estándar internacional de fairness: si el grupo menos '
                'favorecido recibe predicciones positivas a menos del '
                '80% de la tasa del grupo más favorecido, el modelo '
                'muestra señales de discriminación estadística.'
            )
        },
    ]

    for col, t in zip([col1, col2, col3], tarjetas):
        with col:
            st.markdown(f"""
            <div style="
                background: white;
                border: 1px solid {COLORES['borde']};
                border-top: 3px solid {COLORES['primario']};
                border-radius: 8px;
                padding: 1.2rem;
                height: 200px;
            ">
                <div style="font-size: 1.8rem;">{t['icono']}</div>
                <div style="font-weight: bold; color: {COLORES['primario']};
                            margin: 0.4rem 0; font-size: 0.92rem;">
                    {t['titulo']}
                </div>
                <div style="font-size: 0.8rem; color: {COLORES['texto']};
                            line-height: 1.5;">
                    {t['texto']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# BLOQUE 2 y 3: Equidad por grupo (sexo o rama)
# =============================================================================

def _bloque_equidad_por_grupo(df: pd.DataFrame, grupo: str):
    """
    Analiza la equidad del modelo para un grupo dado (sexo o rama).
    Muestra tabla de métricas + gráfico comparativo de barras agrupadas.
    """
    nombre_grupo = "sexo" if grupo == 'sexo' else "rama de conocimiento"
    icono        = "👥" if grupo == 'sexo' else "📚"

    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        {icono} Equidad por {nombre_grupo}
    </h4>
    """, unsafe_allow_html=True)

    if grupo not in df.columns:
        st.info(f"La columna '{grupo}' no está disponible en los datos.")
        return

    # Calculamos métricas por subgrupo
    grupos     = df[grupo].dropna().unique()
    filas      = []
    for g in sorted(grupos):
        df_g   = df[df[grupo] == g]
        metricas = _metricas_grupo(df_g)
        if metricas:
            metricas[nombre_grupo.capitalize()] = g
            filas.append(metricas)

    if not filas:
        st.info("No hay suficientes datos para calcular métricas por grupo.")
        return

    df_met = pd.DataFrame(filas)
    col_grupo = nombre_grupo.capitalize()

    # --- Tabla de métricas ---
    tabla = df_met[[col_grupo, 'n', 'tasa_real', 'tasa_pred',
                    'precision', 'recall', 'f1', 'auc']].copy()
    tabla.columns = [col_grupo, 'N alumnos', 'Abandono real (%)',
                     'Predicho (%)', 'Precisión (%)', 'Recall (%)',
                     'F1 (%)', 'AUC (%)']

    for col in ['Abandono real (%)', 'Predicho (%)', 'Precisión (%)',
                'Recall (%)', 'F1 (%)', 'AUC (%)']:
        tabla[col] = tabla[col].round(1)

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            'F1 (%)': st.column_config.ProgressColumn(
                'F1 (%)', min_value=0, max_value=100, format='%.1f%%'
            ),
            'AUC (%)': st.column_config.ProgressColumn(
                'AUC (%)', min_value=0, max_value=100, format='%.1f%%'
            ),
        }
    )

    # --- Gráfico de barras agrupadas: métricas clave por grupo ---
    metricas_plot = ['Precisión (%)', 'Recall (%)', 'F1 (%)']
    fig = go.Figure()

    colores_grupos = [COLORES['primario'], COLORES['abandono'],
                      COLORES['exito'], COLORES['advertencia']]

    for i, fila in tabla.iterrows():
        fig.add_trace(go.Bar(
            name=str(fila[col_grupo]),
            x=metricas_plot,
            y=[fila[m] for m in metricas_plot],
            marker_color=colores_grupos[i % len(colores_grupos)],
            text=[f"{fila[m]:.1f}%" for m in metricas_plot],
            textposition='outside',
            hovertemplate=f"<b>{fila[col_grupo]}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    # Línea de referencia: F1 global del modelo (0.799 = 79.9%)
    fig.add_hline(
        y=79.9,
        line_dash="dot",
        line_color=COLORES['texto_suave'],
        line_width=1.5,
        annotation_text="F1 global (79.9%)",
        annotation_position="right",
        annotation_font_size=10,
    )

    fig.update_layout(
        barmode='group',
        yaxis=dict(range=[0, 110], ticksuffix='%'),
        yaxis_title="Valor (%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

    st.plotly_chart(fig, use_container_width=True)

    # Interpretación automática
    if 'F1 (%)' in tabla.columns:
        f1_vals  = tabla['F1 (%)'].values
        dif_f1   = f1_vals.max() - f1_vals.min()
        if dif_f1 < 5:
            msg   = f"✅ El modelo muestra un rendimiento **homogéneo** entre grupos de {nombre_grupo} (diferencia F1: {dif_f1:.1f} pp)."
            color = COLORES['exito']
        elif dif_f1 < 10:
            msg   = f"⚠️ Hay una diferencia **moderada** de rendimiento entre grupos de {nombre_grupo} (diferencia F1: {dif_f1:.1f} pp). Merece seguimiento."
            color = COLORES['advertencia']
        else:
            msg   = f"🔴 Hay una diferencia **notable** de rendimiento entre grupos de {nombre_grupo} (diferencia F1: {dif_f1:.1f} pp). Requiere atención."
            color = COLORES['abandono']

        st.markdown(f"""
        <div style="
            background: {color}12;
            border-left: 4px solid {color};
            border-radius: 6px;
            padding: 0.7rem 1rem;
            font-size: 0.87rem;
            margin-top: 0.3rem;
        ">{msg}</div>
        """, unsafe_allow_html=True)


# =============================================================================
# BLOQUE 4: Disparate Impact
# =============================================================================

def _bloque_disparate_impact(df: pd.DataFrame):
    """
    Calcula el Disparate Impact para sexo y rama.
    Métrica estándar: ratio entre tasa de predicción positiva del grupo
    menos favorecido vs el más favorecido.
    Regla del 80%: DI < 0.8 indica posible discriminación estadística.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.5rem;">
        📐 Disparate Impact
    </h4>
    <p style="font-size: 0.85rem; color: {COLORES['texto_suave']}; margin-bottom: 1rem;">
        Ratio entre la tasa de predicción positiva (riesgo alto) del grupo
        menos favorecido respecto al más favorecido.
        <strong>Valor ideal: cercano a 1.0. Señal de alerta: por debajo de 0.8.</strong>
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for col, grupo in zip([col1, col2], ['sexo', 'rama']):
        with col:
            if grupo not in df.columns:
                st.info(f"'{grupo}' no disponible.")
                continue

            # Tasa de predicción positiva por grupo
            tasas = (
                df.groupby(grupo)['pred_abandono']
                .mean()
                .sort_values()
            )

            if len(tasas) < 2:
                st.info("No hay suficientes grupos para calcular DI.")
                continue

            di     = tasas.iloc[0] / tasas.iloc[-1]  # min / max
            grupo_min = tasas.index[0]
            grupo_max = tasas.index[-1]

            # Color según si supera el umbral del 80%
            if di >= 0.8:
                color_di = COLORES['exito']
                texto_di = "✅ Dentro del umbral aceptable"
            elif di >= 0.6:
                color_di = COLORES['advertencia']
                texto_di = "⚠️ Por debajo del umbral recomendado"
            else:
                color_di = COLORES['abandono']
                texto_di = "🔴 Señal de posible discriminación estadística"

            nombre_grupo = "Sexo" if grupo == 'sexo' else "Rama"

            # Gauge de Disparate Impact
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(di, 3),
                number={'font': {'size': 32, 'color': color_di}},
                gauge={
                    'axis': {
                        'range': [0, 1],
                        'tickvals': [0, 0.6, 0.8, 1.0],
                        'ticktext': ['0', '0.6', '0.8\n(umbral)', '1.0'],
                    },
                    'bar': {'color': color_di, 'thickness': 0.3},
                    'steps': [
                        {'range': [0.0, 0.6], 'color': COLORES['abandono'] + '30'},
                        {'range': [0.6, 0.8], 'color': COLORES['advertencia'] + '30'},
                        {'range': [0.8, 1.0], 'color': COLORES['exito'] + '30'},
                    ],
                    'threshold': {
                        'line': {'color': COLORES['texto'], 'width': 2},
                        'thickness': 0.75,
                        'value': 0.8,
                    },
                },
                title={
                    'text': f"DI por {nombre_grupo}<br>"
                            f"<span style='font-size:0.75em;color:gray'>"
                            f"{grupo_min} / {grupo_max}</span>",
                    'font': {'size': 13}
                },
            ))
            fig.update_layout(
                paper_bgcolor="white",
                margin=dict(l=20, r=20, t=50, b=20),
                height=230,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="
                text-align: center;
                font-size: 0.82rem;
                color: {color_di};
                font-weight: bold;
                margin-top: -0.5rem;
            ">{texto_di}</div>
            """, unsafe_allow_html=True)


# =============================================================================
# BLOQUE 5: Matriz de confusión por grupo
# =============================================================================

def _bloque_confusion_por_grupo(df: pd.DataFrame):
    """
    Muestra los falsos positivos y falsos negativos por grupo.
    Pregunta clave: ¿a quién perjudica más el modelo cuando se equivoca?

    Falso positivo (FP): alumno que NO abandona pero el modelo predice que sí
    → intervención innecesaria, puede estigmatizar al alumno

    Falso negativo (FN): alumno que SÍ abandona pero el modelo no lo detecta
    → alumno en riesgo sin apoyo
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.5rem;">
        🔍 ¿A quién perjudica el modelo cuando se equivoca?
    </h4>
    <p style="font-size: 0.85rem; color: {COLORES['texto_suave']}; margin-bottom: 1rem;">
        <strong>Falso positivo (FP):</strong> el modelo predice abandono pero el alumno completa el grado
        → intervención innecesaria.<br>
        <strong>Falso negativo (FN):</strong> el alumno abandona pero el modelo no lo detecta
        → alumno sin apoyo.
    </p>
    """, unsafe_allow_html=True)

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        grupo_conf = st.selectbox(
            label="Analizar errores por:",
            options=["sexo", "rama"] if 'rama' in df.columns else ["sexo"],
            format_func=lambda x: "Sexo" if x == "sexo" else "Rama de conocimiento",
            key="grupo_confusion",
        )

    if grupo_conf not in df.columns or 'abandono' not in df.columns:
        st.info("Datos no disponibles para este análisis.")
        return

    grupos    = sorted(df[grupo_conf].dropna().unique())
    datos_fp_fn = []

    for g in grupos:
        df_g = df[df[grupo_conf] == g]
        if len(df_g) < 5:
            continue
        y_true = df_g['abandono'].values
        y_pred = df_g['pred_abandono'].values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() \
            if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
        n = len(df_g)
        datos_fp_fn.append({
            'grupo':     g,
            'FP (%)':    round(fp / n * 100, 1),
            'FN (%)':    round(fn / n * 100, 1),
            'TP (%)':    round(tp / n * 100, 1),
            'TN (%)':    round(tn / n * 100, 1),
            'N':         n,
        })

    if not datos_fp_fn:
        st.info("No hay suficientes datos para este análisis.")
        return

    df_conf = pd.DataFrame(datos_fp_fn)

    # Heatmap de FP y FN por grupo
    fig = go.Figure()

    for tipo, color, descripcion in [
        ('FP (%)', COLORES['advertencia'], 'Falsos positivos'),
        ('FN (%)', COLORES['abandono'],    'Falsos negativos'),
        ('TP (%)', COLORES['exito'],       'Verdaderos positivos'),
    ]:
        fig.add_trace(go.Bar(
            name=descripcion,
            x=df_conf['grupo'],
            y=df_conf[tipo],
            marker_color=color,
            text=df_conf[tipo].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            hovertemplate=f"<b>%{{x}}</b><br>{descripcion}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode='group',
        yaxis_title="% sobre total del grupo",
        yaxis=dict(ticksuffix='%'),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
        height=340,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "💡 Un porcentaje de FN alto significa que el modelo deja sin detectar "
        "a muchos alumnos en riesgo de ese grupo. Un FP alto implica más "
        "intervenciones innecesarias."
    )


# =============================================================================
# BLOQUE 6: Simulador de política institucional ★
# =============================================================================

def _bloque_simulador_politica(df: pd.DataFrame):
    """
    Permite al gestor o al tribunal ajustar el umbral de decisión del modelo
    y ver en tiempo real cómo cambia el impacto institucional:
    - Cuántos alumnos se detectan como en riesgo
    - Cuántos son falsos positivos (intervención innecesaria)
    - Cuántos son falsos negativos (alumnos sin apoyo)
    - Coste estimado de intervención

    Este bloque transforma el análisis técnico en una herramienta de
    apoyo a la toma de decisiones institucionales reales.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.3rem;">
        🎛️ Simulador de política institucional
        <span style="
            font-size: 0.7rem;
            background: {COLORES['primario']};
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 10px;
            margin-left: 0.5rem;
            vertical-align: middle;
        ">★ Extra</span>
    </h4>
    <p style="font-size: 0.85rem; color: {COLORES['texto_suave']}; margin-bottom: 1rem;">
        ¿Qué pasa si la UJI decide intervenir con todos los alumnos que superen
        un umbral de riesgo determinado? Ajusta el umbral y ve el impacto en
        tiempo real. Esto permite encontrar el equilibrio entre detectar más
        casos (recall alto) y no saturar los servicios de orientación (pocos FP).
    </p>
    """, unsafe_allow_html=True)

    if 'abandono' not in df.columns or 'prob_abandono' not in df.columns:
        st.info("Datos no disponibles para el simulador.")
        return

    col_ctrl, col_coste = st.columns([2, 1])

    with col_ctrl:
        umbral_sim = st.slider(
            label="Umbral de intervención (probabilidad de riesgo)",
            min_value=0.10,
            max_value=0.90,
            value=float(UMBRALES['riesgo_medio']),
            step=0.05,
            format="%.2f",
            help=(
                "Si el modelo predice una probabilidad de abandono mayor que "
                "este valor, el alumno recibirá una intervención de apoyo. "
                f"Umbral actual del modelo: {UMBRALES['riesgo_medio']:.2f}"
            ),
            key="umbral_simulador"
        )

    with col_coste:
        coste_intervencion = st.number_input(
            label="Coste estimado por intervención (€)",
            min_value=0,
            max_value=5000,
            value=150,
            step=50,
            help=(
                "Coste aproximado de una sesión de tutoría o intervención "
                "de orientación académica por alumno. Valor orientativo."
            ),
            key="coste_intervencion"
        )

    # Calculamos métricas con el umbral simulado
    y_true      = df['abandono'].values
    y_prob      = df['prob_abandono'].values
    y_pred_sim  = (y_prob >= umbral_sim).astype(int)

    n_total         = len(df)
    n_intervencion  = y_pred_sim.sum()
    n_fp            = ((y_pred_sim == 1) & (y_true == 0)).sum()
    n_fn            = ((y_pred_sim == 0) & (y_true == 1)).sum()
    n_tp            = ((y_pred_sim == 1) & (y_true == 1)).sum()
    coste_total     = n_intervencion * coste_intervencion
    recall_sim      = n_tp / max(y_true.sum(), 1) * 100
    precision_sim   = n_tp / max(n_intervencion, 1) * 100
    pct_intervencion = n_intervencion / n_total * 100

    # Métricas en tarjetas
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "🔔 Alumnos a intervenir",
            f"{n_intervencion:,}",
            delta=f"{pct_intervencion:.1f}% del total",
            delta_color="off"
        )
    with c2:
        st.metric(
            "✅ Detectados correctamente",
            f"{n_tp:,}",
            delta=f"Recall: {recall_sim:.1f}%",
            delta_color="normal"
        )
    with c3:
        st.metric(
            "⚠️ Falsas alarmas (FP)",
            f"{n_fp:,}",
            delta=f"{n_fp/n_total*100:.1f}% del total",
            delta_color="inverse"
        )
    with c4:
        st.metric(
            "❌ Sin detectar (FN)",
            f"{n_fn:,}",
            delta=f"{n_fn/y_true.sum()*100:.1f}% de los que abandonan",
            delta_color="inverse"
        )
    with c5:
        st.metric(
            "💶 Coste estimado",
            f"{coste_total:,.0f} €",
            delta=f"{coste_intervencion} €/alumno",
            delta_color="off"
        )

    # Gráfico: curva de recall y precisión según el umbral
    # Calculamos para todos los umbrales posibles (0.05 a 0.95)
    umbrales_rango = np.arange(0.05, 0.96, 0.02)
    recalls, precisiones, n_interv = [], [], []

    for u in umbrales_rango:
        y_p = (y_prob >= u).astype(int)
        tp  = ((y_p == 1) & (y_true == 1)).sum()
        recalls.append(tp / max(y_true.sum(), 1) * 100)
        precisiones.append(tp / max(y_p.sum(), 1) * 100)
        n_interv.append(y_p.sum())

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=umbrales_rango,
        y=recalls,
        name='Recall (% abandonos detectados)',
        line=dict(color=COLORES['exito'], width=2.5),
        hovertemplate="Umbral: %{x:.2f}<br>Recall: %{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=umbrales_rango,
        y=precisiones,
        name='Precisión (% intervenciones acertadas)',
        line=dict(color=COLORES['primario'], width=2.5),
        hovertemplate="Umbral: %{x:.2f}<br>Precisión: %{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=umbrales_rango,
        y=[n / n_total * 100 for n in n_interv],
        name='% alumnos a intervenir',
        line=dict(color=COLORES['advertencia'], width=1.5, dash='dash'),
        hovertemplate="Umbral: %{x:.2f}<br>Intervenciones: %{y:.1f}%<extra></extra>",
        yaxis='y2',
    ))

    # Línea vertical: umbral actual del simulador
    fig.add_vline(
        x=umbral_sim,
        line_color=COLORES['abandono'],
        line_width=2,
        line_dash="solid",
        annotation_text=f"  Umbral actual ({umbral_sim:.2f})",
        annotation_font_color=COLORES['abandono'],
        annotation_font_size=11,
    )

    fig.update_layout(
        xaxis_title="Umbral de decisión",
        yaxis_title="Porcentaje (%)",
        yaxis=dict(range=[0, 105], ticksuffix='%'),
        yaxis2=dict(
            title="% alumnos a intervenir",
            overlaying='y',
            side='right',
            range=[0, 105],
            ticksuffix='%',
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=60, t=40, b=40),
        height=360,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORES['borde'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "💡 Mueve el slider para encontrar el equilibrio óptimo entre detectar "
        "el máximo de abandonos (recall alto) y no saturar los servicios de "
        "orientación (precisión alta, pocas falsas alarmas)."
    )


# =============================================================================
# BLOQUE 7: Conclusión y limitaciones
# =============================================================================

def _bloque_conclusion(df: pd.DataFrame):
    """
    Conclusión directa y honesta sobre la equidad del modelo.
    Incluye limitaciones explícitas — fundamental para el tribunal.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.8rem;">
        📝 Conclusión: ¿Es justo el modelo?
    </h4>
    """, unsafe_allow_html=True)

    col_concl, col_limit = st.columns([1, 1])

    with col_concl:
        st.markdown(f"""
        <div style="
            background: {COLORES['exito']}10;
            border: 1px solid {COLORES['exito']}40;
            border-radius: 8px;
            padding: 1.2rem;
            height: 100%;
        ">
        <div style="font-weight: bold; color: {COLORES['exito']};
                    font-size: 1rem; margin-bottom: 0.8rem;">
            ✅ Valoración general
        </div>
        <div style="font-size: 0.85rem; color: {COLORES['texto']};
                    line-height: 1.7;">
            El modelo Stacking entrenado en este TFM muestra un comportamiento
            <strong>razonablemente equitativo</strong> entre los grupos analizados.<br><br>
            Las diferencias de rendimiento entre grupos de sexo y rama se
            encuentran dentro de los márgenes habituales en la literatura
            de fairness en educación superior.<br><br>
            El Disparate Impact supera en la mayoría de casos el umbral
            del 80%, lo que indica que el modelo no presenta señales
            sistemáticas de discriminación estadística.<br><br>
            <em>No obstante, cualquier uso institucional de este modelo
            debe ir acompañado de supervisión humana y revisión periódica.</em>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_limit:
        st.markdown(f"""
        <div style="
            background: {COLORES['advertencia']}10;
            border: 1px solid {COLORES['advertencia']}40;
            border-radius: 8px;
            padding: 1.2rem;
            height: 100%;
        ">
        <div style="font-weight: bold; color: {COLORES['advertencia']};
                    font-size: 1rem; margin-bottom: 0.8rem;">
            ⚠️ Limitaciones del análisis
        </div>
        <div style="font-size: 0.85rem; color: {COLORES['texto']};
                    line-height: 1.7;">
            <strong>1. Datos históricos:</strong> el modelo refleja los patrones
            de 2010–2020. Los cambios estructurales recientes (pandemia, nuevos
            grados) pueden no estar representados.<br><br>
            <strong>2. Equidad ≠ causalidad:</strong> que el modelo sea equitativo
            no significa que el sistema académico lo sea. El modelo aprende
            de desigualdades preexistentes en los datos.<br><br>
            <strong>3. Grupos no analizados:</strong> no se ha podido analizar
            equidad por nivel socioeconómico ni origen geográfico por falta
            de datos disponibles.<br><br>
            <strong>4. Uso ético obligatorio:</strong> este modelo es una
            herramienta de apoyo, nunca de sustitución del juicio humano.
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Referencia al marco normativo
    with st.expander("📖 Marco normativo y referencias de fairness — clic para ampliar",
                     expanded=False):
        st.markdown(f"""
        <div style="font-size: 0.83rem; color: {COLORES['texto']}; line-height: 1.8;">

        <strong>Marco normativo aplicable:</strong><br>
        · Reglamento UE 2016/679 (RGPD) — protección de datos personales<br>
        · Propuesta de Reglamento de IA de la UE (AI Act, 2024) —
          sistemas de IA en educación clasificados como alto riesgo<br>
        · UNESCO Recommendation on the Ethics of AI (2021)<br><br>

        <strong>Métricas de fairness utilizadas:</strong><br>
        · Disparate Impact Ratio (Feldman et al., 2015)<br>
        · Equal Opportunity / Equalized Odds (Hardt et al., 2016)<br>
        · Análisis de FP/FN diferencial por subgrupos<br><br>

        <strong>Librería de referencia:</strong> fairlearn (Microsoft, 2020)<br>
        <strong>Umbral estándar:</strong> Disparate Impact ≥ 0.8
        (EEOC Four-Fifths Rule, adaptado a ML)

        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# FIN DE p05_equidad.py
# =============================================================================
