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

from config_app import APP_CONFIG, COLORES, PESTANAS, RUTAS
import json as _json

def _cargar_metricas() -> dict:
    """Lee metricas_modelo.json generado por f6_m00_preparacion.ipynb."""
    ruta = RUTAS.get("metricas_modelo")
    if ruta and ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            return _json.load(f)
    # Fallback si el fichero no existe todavía
    return {
        "auc": 0.9308, "f1": 0.7988, "baseline_auc": 0.9270, "baseline_f1": 0.7970,
        "n_alumnos_unicos": 30872, "n_registros": 33621,
        "tasa_abandono": 0.292, "periodo_inicio": 2010, "periodo_fin": 2021,
        "modelo_nombre": "Stacking (CatBoost + RF + LogReg)",
        "baseline_nombre": "CatBoost AutoML",
    }


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
    _semaforo_estado()
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
    m = _cargar_metricas()
    n_registros     = f"{m['n_registros']:,}".replace(",", ".")
    n_alumnos       = f"{m['n_alumnos_unicos']:,}".replace(",", ".")
    periodo_inicio  = m['periodo_inicio']
    periodo_fin     = m['periodo_fin']

    col_texto, col_logo = st.columns([2, 1])

    with col_texto:
        st.markdown(f"""
        <h1 style="color: {COLORES["primario"]}; margin-bottom: 0.2rem;">
            {APP_CONFIG["icono"]} {APP_CONFIG["titulo"].split("—")[0].strip()}
        </h1>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <p style="font-size: 1rem; color: {COLORES["texto"]}; max-width: 680px;">
            Esta herramienta analiza y predice el riesgo de abandono en los grados
            de la {APP_CONFIG['universidad_datos']} mediante un modelo de <em>machine learning</em> entrenado con
            datos reales de <strong>{n_registros} registros</strong> de
            <strong>{n_alumnos} estudiantes únicos</strong>
            entre {periodo_inicio} y {periodo_fin}.
            Permite explorar patrones globales, profundizar por titulación,
            y obtener pronósticos individualizados.
        </p>
        """, unsafe_allow_html=True)

    with col_logo:
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
                {APP_CONFIG['universidad_datos']} · <strong>{periodo_inicio}–{periodo_fin}</strong>
            </div>
            <div style="font-size: 0.75rem; color: {COLORES["texto_suave"]};">
                <strong>{APP_CONFIG['ciudad']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SECCIÓN 2: Métricas clave del modelo
# =============================================================================

def _semaforo_estado():
    """Franja de estado del sistema — modelo, datos y última ejecución."""
    import datetime as _dt

    def _check(ruta):
        return ruta is not None and ruta.exists()

    ruta_modelo  = RUTAS.get("modelo")
    ruta_datos   = RUTAS.get("meta_test_app")
    ruta_metricas = RUTAS.get("metricas_modelo")

    ok_modelo = _check(ruta_modelo)
    ok_datos  = _check(ruta_datos)

    # Fecha última ejecución desde metricas_modelo.json
    ultima_eje = "—"
    if _check(ruta_metricas):
        ts = ruta_metricas.stat().st_mtime
        ultima_eje = _dt.datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M")

    # Contar registros si los datos existen
    n_registros = ""
    if ok_datos:
        try:
            import pandas as _pd_sem
            n = len(_pd_sem.read_parquet(ruta_datos))
            n_registros = f" · {n:,} registros".replace(",", ".")
        except Exception:
            pass

    icono_modelo = "✅" if ok_modelo else "❌"
    icono_datos  = "✅" if ok_datos  else "❌"
    color_modelo = COLORES["exito"] if ok_modelo else COLORES["abandono"]
    color_datos  = COLORES["exito"] if ok_datos  else COLORES["abandono"]

    nombre_modelo = ruta_modelo.name if ok_modelo else "No encontrado"

    st.markdown(f"""
    <div style="
        background: {COLORES['fondo']};
        border: 1px solid {COLORES['borde']};
        border-left: 4px solid {COLORES['primario']};
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin-top: 0.8rem;
        font-size: 0.82rem;
        color: {COLORES['texto_suave']};
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        align-items: center;
    ">
        <span><span style="color:{color_modelo}; font-weight:600;">{icono_modelo} Modelo:</span> {nombre_modelo}</span>
        <span>·</span>
        <span><span style="color:{color_datos}; font-weight:600;">{icono_datos} Datos:</span> {ruta_datos.name if ok_datos else 'No encontrado'}{n_registros}</span>
        <span>·</span>
        <span>🕐 <strong>Última ejecución:</strong> {ultima_eje}</span>
    </div>
    """, unsafe_allow_html=True)


def _sparkline(val_base: float, val_modelo: float, nombre_base: str, nombre_modelo: str):
    """Mini barra comparativa con hover — barra, texto y tooltip al pasar el ratón."""
    pct_base   = round(val_base * 100, 1)
    pct_modelo = round(val_modelo * 100, 1)
    pct_mejora = round((val_modelo - val_base) * 100, 1)
    pct_mejora_rel = round((val_modelo - val_base) / val_base * 100, 1)

    uid = nombre_base.replace(" ", "_") + nombre_modelo.replace(" ", "_")

    st.markdown(f"""
    <style>
    .spark_{uid}:hover .spark-bar-base_{uid} {{ height: 10px !important; }}
    .spark_{uid}:hover .spark-bar-mejora_{uid} {{ height: 10px !important; }}
    .spark_{uid}:hover .spark-texto_{uid} {{ font-size: 0.85rem !important; }}
    .spark_{uid}:hover .spark-tooltip_{uid} {{ display: block !important; }}
    </style>
    <div class="spark_{uid}" style="margin-top:0.3rem; cursor:default; position:relative;">
        <div style="display:flex; align-items:center; gap:3px; margin-bottom:3px; transition: all 0.2s;">
            <div class="spark-bar-base_{uid}" style="
                width:{pct_base}%;
                height:6px;
                background:{COLORES['borde']};
                border-radius:3px 0 0 3px;
                transition: height 0.2s;
            "></div>
            <div class="spark-bar-mejora_{uid}" style="
                width:{pct_mejora}%;
                height:6px;
                background:{COLORES['primario']};
                border-radius:0 3px 3px 0;
                transition: height 0.2s;
            "></div>
        </div>
        <div class="spark-texto_{uid}" style="font-size:0.7rem; color:{COLORES['texto_suave']}; transition: font-size 0.2s;">
            <span>{nombre_base}: {str(pct_base).replace('.', ',')}%</span>
            &nbsp;→&nbsp;
            <span style="color:{COLORES['exito']}; font-weight:600;">
                {nombre_modelo}: {str(pct_modelo).replace('.', ',')}%
            </span>
        </div>
        <div class="spark-tooltip_{uid}" style="
            display:none;
            position:absolute;
            top:-4.5rem;
            left:0;
            background:{COLORES['blanco']};
            border:1px solid {COLORES['borde']};
            border-radius:6px;
            padding:0.4rem 0.7rem;
            font-size:0.82rem;
            color:{COLORES['texto']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.10);
            white-space:nowrap;
            z-index:100;
        ">
            <strong>{nombre_base}:</strong> {str(pct_base).replace('.', ',')}%<br>
            <strong style="color:{COLORES['exito']}">{nombre_modelo}:</strong> {str(pct_modelo).replace('.', ',')}%<br>
            <span style="color:{COLORES['exito']}; font-weight:600;">
                ▲ +{str(pct_mejora).replace('.', ',')} pp &nbsp;·&nbsp; +{str(pct_mejora_rel).replace('.', ',')}% relativo
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _metricas_modelo():
    """Banner con los indicadores clave del modelo — valores del JSON dinámico."""
    m = _cargar_metricas()

    def _fmt(v):
        return f"{v:.3f}".replace(".", ",")
    def _fmt_pct(v):
        return f"{v*100:.1f} %".replace(".", ",")
    def _fmt_n(v):
        return f"{v:,}".replace(",", ".")

    delta_auc = m['auc'] - m['baseline_auc']
    delta_f1  = m['f1']  - m['baseline_f1']

    st.markdown(f"""
    <h3 style="color: {COLORES["texto"]}; margin-bottom: 1rem;">
        📊 Indicadores del modelo
    </h3>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            label="🎯 AUC-ROC",
            value=_fmt(m['auc']),
            delta=f"{delta_auc:+.3f} vs {m['baseline_nombre']}".replace(".", ","),
            delta_color="normal",
            help=f"Área bajo la curva ROC. {m['baseline_nombre']}: {_fmt(m['baseline_auc'])}"
        )
        _sparkline(m['baseline_auc'], m['auc'], m['baseline_nombre'], m['modelo_nombre'].split('(')[0].strip())
    with c2:
        st.metric(
            label="⚖️ F1-Score test",
            value=_fmt(m['f1']),
            delta=f"{delta_f1:+.3f} vs {m['baseline_nombre']}".replace(".", ","),
            delta_color="normal",
            help=f"Media armónica de precisión y recall. {m['baseline_nombre']}: {_fmt(m['baseline_f1'])}"
        )
        _sparkline(m['baseline_f1'], m['f1'], m['baseline_nombre'], m['modelo_nombre'].split('(')[0].strip())
    with c3:
        st.metric(
            label="👥 Alumnos únicos",
            value=_fmt_n(m['n_alumnos_unicos']),
            help=f"Alumnos únicos en el dataset original. Dataset de modelado: {_fmt_n(m['n_registros'])} registros."
        )
    with c4:
        st.metric(
            label="📉 Tasa abandono",
            value=_fmt_pct(m['tasa_abandono']),
            help=f"Porcentaje de abandono en el dataset de modelado ({RUTAS.get('metricas_modelo').name})"
        )
    with c5:
        st.metric(
            label="🏆 Mejor modelo",
            value=m['modelo_nombre'].split('(')[0].strip(),
            help=f"{m['modelo_nombre']}"
        )


# =============================================================================
# SECCIÓN 3: Tarjetas de navegación
# =============================================================================

def _tarjetas_navegacion():
    """Una tarjeta visual por cada sección de la app."""

    m = _cargar_metricas()

    def m_fmt_n(v):
        return f"{v:,}".replace(",", ".")

    st.markdown(f"""
    <h3 style="color: {COLORES["texto"]}; margin-bottom: 1rem;">
        🗺️ Secciones disponibles
    </h3>
    """, unsafe_allow_html=True)

    columnas = st.columns(len(PESTANAS))

    # Pasada 1 — tarjetas HTML
    for col, pestana in zip(columnas, PESTANAS):
        with col:
            if pestana["id"] == "institucional":
                stat = (f'<span style="font-size:0.78rem;font-weight:600;color:{COLORES["primario"]};"><strong>' +
                        m_fmt_n(m["n_titulaciones"]) + '</strong></span>' +
                        f'<span style="font-size:0.7rem;color:{COLORES["texto_suave"]};"> titulaciones · </span>' +
                        f'<span style="font-size:0.78rem;font-weight:600;color:{COLORES["primario"]};"><strong>' +
                        m_fmt_n(m["n_alumnos_unicos"]) + '</strong></span>' +
                        f'<span style="font-size:0.7rem;color:{COLORES["texto_suave"]};"> alumnos</span>')
            elif pestana["id"] == "titulacion":
                stat = (f'<span style="font-size:0.78rem;font-weight:600;color:{COLORES["primario"]};"><strong>' +
                        str(APP_CONFIG["n_ramas"]) + '</strong></span>' +
                        f'<span style="font-size:0.7rem;color:{COLORES["texto_suave"]};"> ramas · </span>' +
                        f'<span style="font-size:0.78rem;font-weight:600;color:{COLORES["primario"]};"><strong>' +
                        m_fmt_n(m["n_titulaciones"]) + '</strong></span>' +
                        f'<span style="font-size:0.7rem;color:{COLORES["texto_suave"]};"> grados</span>')
            elif pestana["id"] == "prospecto":
                stat = (f'<span style="font-size:0.78rem;font-weight:600;color:{COLORES["primario"]};"><strong>' +
                        str(APP_CONFIG["n_variables"]) + '</strong></span>' +
                        f'<span style="font-size:0.7rem;color:{COLORES["texto_suave"]};"> variables · AUC </span>' +
                        f'<span style="font-size:0.78rem;font-weight:600;color:{COLORES["primario"]};"><strong>' +
                        str(round(m["auc"], 3)).replace(".", ",") + '</strong></span>')
            elif pestana["id"] == "en_curso":
                stat = (f'<span style="font-size:0.78rem;font-weight:600;color:{COLORES["primario"]};"><strong>' +
                        m_fmt_n(m["n_alumnos_unicos"]) + '</strong></span>' +
                        f'<span style="font-size:0.7rem;color:{COLORES["texto_suave"]};"> alumnos · F1 </span>' +
                        f'<span style="font-size:0.78rem;font-weight:600;color:{COLORES["primario"]};"><strong>' +
                        str(round(m["f1"], 3)).replace(".", ",") + '</strong></span>')
            else:
                stat = (f'<span style="font-size:0.78rem;font-weight:600;color:{COLORES["primario"]};"><strong>' +
                        str(APP_CONFIG["n_ramas"]) + '</strong></span>' +
                        f'<span style="font-size:0.7rem;color:{COLORES["texto_suave"]};"> grupos analizados</span>')

            stat_html = (f'<div style="border-top:0.5px solid {COLORES["borde"]};padding-top:5px;margin-top:4px;">' +
                         stat + '</div>')

            st.markdown(
                '<div style="' +
                f'background-color:{COLORES["blanco"]};border:0.5px solid {COLORES["borde"]};' +
                f'border-top:3px solid {COLORES["primario"]};border-radius:8px;' +
                'padding:1rem 0.8rem 0.8rem 0.8rem;text-align:center;height:230px;' +
                'display:flex;flex-direction:column;justify-content:center;align-items:center;gap:0.25rem;">' +
                f'<div style="font-size:1.8rem;">{pestana["icono"]}</div>' +
                f'<div style="font-weight:bold;font-size:0.88rem;color:{COLORES["primario"]};">{pestana["titulo"]}</div>' +
                f'<div style="font-size:0.72rem;color:{COLORES["texto_suave"]};line-height:1.3;">{pestana["descripcion"]}</div>' +
                f'<hr style="border:none;border-top:0.5px solid {COLORES["borde"]};margin:6px 0 4px 0;width:80%;">' +
                f'<div style="font-size:0.68rem;color:{COLORES["texto_suave"]};font-style:italic;line-height:1.3;">{pestana["detalle"]}</div>' +
                stat_html +
                '</div>' +
                f'<div style="font-size:0.68rem;color:{COLORES["texto_suave"]};text-align:center;height:2.5rem;' +
                'display:flex;align-items:center;justify-content:center;margin-bottom:0.3rem;">' +
                f'👤 {pestana["perfil"]}</div>',
                unsafe_allow_html=True
            )

    # Pasada 2 — botones
    for i, (col, pestana) in enumerate(zip(columnas, PESTANAS)):
        with col:
            if st.button(
                f"→ {pestana['titulo']}",
                key=f"btn_nav_{pestana['id']}",
                width='stretch',
            ):
                st.session_state["nav_idx"] = i + 1
                st.rerun()
# =============================================================================
# SECCIÓN 4: Nota metodológica
# =============================================================================

def _nota_metodologica():
    """Resumen ejecutivo dinámico + nota metodológica detallada."""

    with st.expander("📋 Resumen ejecutivo y nota metodológica — haz clic para ampliar", expanded=False):
        m = _cargar_metricas()
        def _fmt(v): return f"{v:.3f}".replace(".", ",")
        def _fmt_pct(v): return f"{v*100:.1f}%".replace(".", ",")
        def _fmt_n(v): return f"{v:,}".replace(",", ".")

        modelo_corto = m["modelo_nombre"].split("(")[0].strip()
        baseline     = m.get("baseline_nombre", "baseline")

        # --- Resumen ejecutivo dinámico ---
        st.markdown(f"""
        <div style="
            background:{COLORES['fondo']};
            border-left:4px solid {COLORES['primario']};
            border-radius:6px;
            padding:0.8rem 1rem;
            margin-bottom:1rem;
            font-size:0.92rem;
            color:{COLORES['texto']};
            line-height:1.7;
        ">
            En la cohorte <strong>{m["periodo_inicio"]}–{m["periodo_fin"]}</strong>
            de la <strong>{APP_CONFIG["universidad_datos"]}</strong>, el modelo
            <strong><em>{modelo_corto}</em></strong> analiza
            <strong>{_fmt_n(m["n_alumnos_unicos"])}</strong> estudiantes únicos
            (<strong>{_fmt_n(m["n_registros"])}</strong> registros) y detecta una tasa de abandono del
            <strong>{_fmt_pct(m["tasa_abandono"])}</strong>.
            El modelo supera al {baseline} en
            <strong>+{str(round(m["auc"]-m["baseline_auc"],3)).replace(".",",")} puntos de AUC</strong>
            (<strong>{_fmt(m["auc"])}</strong> vs {_fmt(m["baseline_auc"])})
            y en <strong>+{str(round(m["f1"]-m["baseline_f1"],3)).replace(".",",")} puntos de F1</strong>
            (<strong>{_fmt(m["f1"])}</strong> vs {_fmt(m["baseline_f1"])}).
        </div>
        """, unsafe_allow_html=True)

        # --- Detalles técnicos ---
        st.markdown(f"""
        <div style="color:{COLORES["texto"]}; font-size:0.88rem; line-height:1.6;">
        <strong>Dataset:</strong> <strong>{_fmt_n(m["n_registros"])}</strong> registros · <strong>{_fmt_n(m["n_alumnos_unicos"])}</strong> alumnos únicos · <strong>{APP_CONFIG['universidad_datos']}</strong> · Cursos <strong>{m["periodo_inicio"]}–{m["periodo_fin"]}</strong>.<br>
        <strong>Variable objetivo:</strong> abandono definitivo del grado (definición estricta, sin traslados ni cambios de titulación). Tasa: <strong>{_fmt_pct(m["tasa_abandono"])}</strong>.<br>
        <strong>Proceso:</strong> ingestión → EDA → ingeniería de características → modelado (validación cruzada 5-fold) → interpretabilidad (SHAP, LIME) → esta aplicación.<br>
        <strong>Modelo final:</strong> <strong><em>{m["modelo_nombre"]}</em></strong>. AUC = <strong>{_fmt(m["auc"])}</strong> · F1 = <strong>{_fmt(m["f1"])}</strong>.<br>
        <strong>Limitaciones:</strong> modelo entrenado hasta <strong>{m["periodo_fin"]}</strong>. Resultados orientativos — no usar como único criterio de decisión sobre ningún estudiante.
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
        {APP_CONFIG['autora']} &nbsp;·&nbsp;
        {APP_CONFIG['tipo_trabajo']} &nbsp;·&nbsp;
        {APP_CONFIG['universidad_master']} + {APP_CONFIG['universidad_datos']} &nbsp;·&nbsp; {APP_CONFIG['año']}<br>
        <span style="font-size: 0.72rem;">
            {APP_CONFIG['email_master']} &nbsp;·&nbsp; {APP_CONFIG['email_datos']}
        </span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FIN DE p00_inicio.py
# =============================================================================
