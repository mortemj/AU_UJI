# =============================================================================
# p00_inicio.py
# Pantalla de bienvenida — lo primero que ve cualquier usuario
#
# ¿QUÉ HACE ESTE FICHERO?
#   Muestra la pantalla de entrada de la app. No hace cálculos ni carga
#   modelos. Su único objetivo es orientar al usuario: qué es esto,
#   para quién es, cómo se usa, y qué puede encontrar en cada sección.
#
# ¿CUÁNDO SE MUESTRA?
#   Siempre que la app arranca por primera vez, o cuando el usuario
#   selecciona "Inicio" en la barra lateral.
#
# ESTRUCTURA DE LA PÁGINA:
#   1. Banner superior con título y descripción
#   2. Métricas clave del modelo (AUC, F1, nº alumnos...)
#   3. Tarjetas de navegación — una por pestaña
#   4. Nota metodológica breve
#   5. Pie de página
#
# REQUISITOS:
#   - config_app.py accesible (un nivel arriba)
#
# GENERA:
#   Página HTML renderizada por Streamlit. No genera ficheros en disco.
#
# SIGUIENTE:
#   pages/p01_institucional.py — primera pestaña de análisis
# =============================================================================

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Imports internos
# ---------------------------------------------------------------------------
# _path_setup añade app/ a sys.path de forma robusta en Windows/OneDrive
import _path_setup  # noqa: F401

from config_app import APP_CONFIG, COLORES, PESTANAS


# =============================================================================
# FUNCIÓN PRINCIPAL — llamada desde main.py
# =============================================================================
# Todas las páginas exponen una función show() sin argumentos.
# main.py la llama cuando el usuario selecciona esta sección.

def show():
    """Renderiza la pantalla de bienvenida completa."""

    _banner_principal()
    st.divider()
    _metricas_modelo()
    st.divider()
    _tarjetas_navegacion()
    st.divider()
    _nota_metodologica()
    _pie_pagina()


# =============================================================================
# SECCIÓN 1: Banner principal
# =============================================================================

def _banner_principal():
    """Título, subtítulo y descripción general de la app."""

    # st.columns() divide la fila en columnas.
    # [2, 1] significa: primera columna el doble de ancha que la segunda.
    col_texto, col_logo = st.columns([2, 1])

    with col_texto:
        st.markdown(f"""
        <h1 style="color: {COLORES["primario"]}; margin-bottom: 0.2rem;">
            {APP_CONFIG["icono"]} Predicción de Abandono Universitario
        </h1>
        <p style="font-size: 1.15rem; color: {COLORES["texto_suave"]}; margin-top: 0;">
            Universitat Jaume I &nbsp;·&nbsp; Trabajo Final de Máster
        </p>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <p style="font-size: 1rem; color: {COLORES["texto"]}; max-width: 680px;">
            Esta herramienta analiza y predice el riesgo de abandono en los grados
            de la UJI mediante un modelo de <em>machine learning</em> entrenado con
            datos reales de <strong>30.872 estudiantes</strong> entre 2010 y 2020.
            Permite explorar patrones globales, profundizar por titulación,
            y obtener pronósticos individualizados.
        </p>
        """, unsafe_allow_html=True)

    with col_logo:
        # Bloque visual decorativo con el escudo/icono de la UJI
        # En una versión futura se puede sustituir por st.image() con el logo real
        st.markdown(f"""
        <div style="
            text-align: center;
            padding: 2rem 1rem;
            background-color: {COLORES["primario"]}15;
            border-radius: 12px;
            border: 1px solid {COLORES["primario"]}30;
            margin-top: 0.5rem;
        ">
            <div style="font-size: 4rem;">🎓</div>
            <div style="font-size: 0.85rem; color: {COLORES["primario"]}; font-weight: bold;">
                UJI · 2010–2020
            </div>
            <div style="font-size: 0.75rem; color: {COLORES["texto_suave"]};">
                Castellón de la Plana
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SECCIÓN 2: Métricas clave del modelo
# =============================================================================

def _metricas_modelo():
    """Banner con los indicadores clave del modelo entrenado."""

    st.markdown(f"""
    <h3 style="color: {COLORES["texto"]}; margin-bottom: 1rem;">
        📊 Indicadores del modelo
    </h3>
    """, unsafe_allow_html=True)

    # st.columns(N) con N igual crea N columnas del mismo ancho
    c1, c2, c3, c4, c5 = st.columns(5)

    # st.metric() es el widget nativo de Streamlit para mostrar KPIs.
    # Parámetros: label (título), value (valor principal), delta (cambio respecto baseline)
    # delta_color="normal" → verde si positivo, rojo si negativo
    # delta_color="off"    → gris siempre (para valores sin dirección)

    with c1:
        st.metric(
            label="🎯 AUC-ROC",
            value="0.931",
            delta="+0.004 vs baseline",
            delta_color="normal",
            help="Área bajo la curva ROC. Mide la capacidad discriminativa del modelo. "
                 "Baseline AutoML (CatBoost_BAG_L2): 0.927"
        )

    with c2:
        st.metric(
            label="⚖️ F1-Score test",
            value="0.799",
            delta="+0.002 vs baseline",
            delta_color="normal",
            help="Media armónica de precisión y recall sobre el conjunto de test. "
                 "Baseline: 0.797"
        )

    with c3:
        st.metric(
            label="👥 Alumnos únicos",
            value="30.872",
            delta=None,
            delta_color="off",
            help="Total de estudiantes en el dataset original (cursos 2010–2020)"
        )

    with c4:
        st.metric(
            label="📉 Tasa abandono",
            value="29,2 %",
            delta=None,
            delta_color="off",
            help="Porcentaje de abandono en el conjunto de test (meta_test.parquet)"
        )

    with c5:
        st.metric(
            label="🏆 Mejor modelo",
            value="Stacking",
            delta=None,
            delta_color="off",
            help="Ensamble de CatBoost + XGBoost + LightGBM + RF con meta-learner logístico"
        )


# =============================================================================
# SECCIÓN 3: Tarjetas de navegación
# =============================================================================

def _tarjetas_navegacion():
    """Una tarjeta visual por cada sección de la app."""

    st.markdown(f"""
    <h3 style="color: {COLORES["texto"]}; margin-bottom: 1rem;">
        🗺️ Secciones disponibles
    </h3>
    """, unsafe_allow_html=True)

    # Creamos una fila de 5 columnas, una por pestaña
    columnas = st.columns(len(PESTANAS))

    for col, pestana in zip(columnas, PESTANAS):
        with col:
            st.markdown(f"""
            <div style="
                background-color: white;
                border: 1px solid {COLORES["borde"]};
                border-top: 3px solid {COLORES["primario"]};
                border-radius: 8px;
                padding: 1.2rem 1rem;
                text-align: center;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 0.4rem;
            ">
                <div style="font-size: 2.2rem;">{pestana["icono"]}</div>
                <div style="font-weight: bold; font-size: 0.9rem;
                            color: {COLORES["primario"]};">{pestana["titulo"]}</div>
                <div style="font-size: 0.75rem; color: {COLORES["texto_suave"]};
                            line-height: 1.3;">{pestana["descripcion"]}</div>
            </div>
            <div style="font-size:0.72rem; color:{COLORES["texto_suave"]};
                        text-align:center; margin-top:0.3rem;">
                👤 {pestana["perfil"]}
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# SECCIÓN 4: Nota metodológica
# =============================================================================

def _nota_metodologica():
    """Resumen metodológico breve para contextualizar al usuario."""

    # st.expander() crea un bloque plegable/desplegable.
    # expanded=False → empieza cerrado (no ocupa espacio visual)
    with st.expander("📋 Nota metodológica — haz clic para ampliar", expanded=False):
        st.markdown(f"""
        <div style="color:{COLORES["texto"]}; font-size:0.9rem; line-height:1.6;">
        <strong>Dataset:</strong> 109.568 registros · 30.872 alumnos únicos · Universitat Jaume I · Cursos académicos 2010–2020.<br>
        <strong>Variable objetivo:</strong> abandono definitivo del grado (definición estricta, sin incluir traslados ni cambios de titulación). Tasa de abandono en test: <strong>29,2 %</strong>.<br>
        <strong>Proceso:</strong> ingestión → EDA → ingeniería de características (auditoría de leakage) → modelado (21 algoritmos, validación cruzada 5-fold estratificada) → interpretabilidad (SHAP, LIME) → esta aplicación.<br>
        <strong>Modelo final:</strong> Stacking con CatBoost, XGBoost, LightGBM y Random Forest como modelos base, y regresión logística como meta-learner. AUC = 0.931 · F1 = 0.799.<br>
        <strong>Limitaciones:</strong> el modelo está entrenado con datos hasta 2020. Las predicciones para cohortes posteriores deben interpretarse con cautela. Los resultados son orientativos y no deben usarse como único criterio de decisión sobre ningún estudiante.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SECCIÓN 5: Pie de página
# =============================================================================

def _pie_pagina():
    """Información de autoría y créditos."""

    st.markdown("<br>", unsafe_allow_html=True)  # espaciado

    st.markdown(f"""
    <div style="
        text-align: center;
        font-size: 0.78rem;
        color: {COLORES["texto_suave"]};
        padding: 1rem;
        border-top: 1px solid {COLORES["borde"]};
    ">
        María José Morte Ruiz &nbsp;·&nbsp;
        TFM Ciencia de Datos &nbsp;·&nbsp;
        UOC + Universitat Jaume I &nbsp;·&nbsp; 2025<br>
        <span style="font-size: 0.72rem;">
            mjmorteruiz@uoc.edu &nbsp;·&nbsp; morte@uji.es
        </span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FIN DE p00_inicio.py
# =============================================================================
