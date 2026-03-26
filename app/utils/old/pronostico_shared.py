# =============================================================================
# pronostico_shared.py
# Módulo compartido de pronóstico personalizado
#
# ¿QUÉ HACE ESTE FICHERO?
#   Contiene toda la lógica y los gráficos del pronóstico individual.
#   Es usado tanto por p03_prospecto.py como por p04_en_curso.py.
#   La diferencia entre ambos modos es mínima:
#     - modo="prospecto" → sin nota_1er_anio ni créditos (aún no cursó nada)
#     - modo="en_curso"  → con nota_1er_anio y créditos superados
#
# ¿POR QUÉ EXISTE ESTE FICHERO?
#   Para no duplicar código entre p03 y p04. Toda la lógica vive aquí.
#   p03 y p04 son wrappers finos que solo llaman a show_pronostico().
#
# ESTRUCTURA:
#   1. Selector de contexto (titulación / rama / sin contexto)
#   2. Formulario de perfil:
#        - Features básicas obligatorias (siempre visibles)
#        - Features avanzadas opcionales (en expander)
#   3. Cálculo de probabilidad con el modelo
#   4. Gráfico 1 — Indicador de riesgo (velocímetro)
#   5. Gráfico 2 — Radar: tu perfil vs éxito Y vs abandono (2 líneas)
#   6. Gráfico 3 — Cascada de contribuciones con selector de método
#        - Método rápido: proxy de diferencia de medias
#        - Método preciso: SHAP TreeExplainer en tiempo real
#   7. Gráfico 4 — Percentil con selector de 3 grupos de referencia
#   8. Recomendaciones personalizadas
#
# FIXES APLICADOS:
#   - cargar_meta_test_app() en vez de cargar_meta_test() (titulaciones fusionadas)
#   - Selector de contexto usa rama_meta (nombre legible) no rama numérica
#   - Radar con 2 trazas: perfil éxito + perfil abandono
#   - Formulario con expander para features avanzadas
#   - Botón de selección de método para la cascada (rápido vs preciso)
#
# PARÁMETROS DE show_pronostico():
#   modo : str — "prospecto" o "en_curso"
#
# REQUISITOS:
#   - config_app.py accesible vía _path_setup
#   - utils/loaders.py disponible
#   - Modelo y pipeline de Fase 5 cargados
#   - shap instalado (para método preciso de cascada)
#
# GENERA:
#   Página Streamlit interactiva. No genera ficheros en disco.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Imports internos — _path_setup añade app/ a sys.path en Windows/OneDrive
# ---------------------------------------------------------------------------
import _path_setup  # noqa: F401

from config_app import (
    COLORES, COLORES_RAMAS, COLORES_RIESGO, NOMBRES_VARIABLES,
    RAMAS_NOMBRES, UMBRALES,
    # Mapas de codificación numérica (texto del formulario → código del modelo)
    VIA_ACCESO_MAP, UNIVERSIDAD_ORIGEN_MAP, RAMA_MAP,
    SEXO_MAP, PROVINCIA_MAP, PAIS_NOMBRE_MAP, SITUACION_LABORAL_MAP,
)
from utils.loaders import cargar_meta_test_app, cargar_modelo, cargar_pipeline


# =============================================================================
# CONSTANTES INTERNAS
# =============================================================================

# Columnas que NO son features del modelo (metadatos del test)
_COLS_META = {
    'abandono', 'titulacion', 'rama', 'rama_meta', 'anio_cohorte',
    'sexo', 'sexo_meta', 'nivel_riesgo', 'prob_abandono',
    'pais_nombre', 'pais_nombre_meta', 'provincia', 'provincia_meta',
    'via_acceso', 'via_acceso_meta', 'per_id_ficticio',
}

# Features básicas — siempre visibles en el formulario
# Ordenadas de mayor a menor importancia predictiva
_FEATURES_BASICAS = [
    'nota_acceso',        # numérica — muy predictiva
    'situacion_laboral',  # categórica — predictor más fuerte (código numérico)
    'n_anios_beca',       # numérica — factor protector clave
    'edad_entrada',       # numérica — nombre real en el modelo (no edad_acceso)
    'via_acceso',         # categórica — nombre real en el modelo (no tipo_acceso)
]

# Features avanzadas — en expander, opcionales
# Solo disponibles en prospecto las pre-matrícula
_FEATURES_AVANZADAS_PROSPECTO = [
    'nota_selectividad',    # numérica
    'orden_preferencia',    # numérica
    'anios_gap',            # numérica
    'universidad_origen',   # categórica
    'sexo',                 # categórica
]

# Features adicionales solo en modo en_curso
_FEATURES_EN_CURSO = [
    'nota_1er_anio',           # numérica — muy predictiva
    'cred_superados_anio_1er', # numérica
    'creditos_superados',      # numérica
    'tasa_rendimiento',        # numérica (calculada)
]

# Opciones para variables categóricas — extraídas del dataset real
_OPCIONES_VIA_ACCESO = list(VIA_ACCESO_MAP.keys())
_OPCIONES_LABORAL = list(SITUACION_LABORAL_MAP.keys())
_OPCIONES_SEXO = list(SEXO_MAP.keys())
_OPCIONES_UNIVERSIDAD = list(UNIVERSIDAD_ORIGEN_MAP.keys())


# =============================================================================
# FUNCIÓN PRINCIPAL — llamada desde p03 y p04
# =============================================================================


# =============================================================================
# CORRECCIÓN HEURÍSTICA: prob × (tasa_tit / tasa_rama)
# =============================================================================
# El modelo predice a nivel de RAMA (5 categorías). Para titulaciones de la
# misma rama el modelo daría el mismo %, aunque su abandono histórico difiera.
# Esta función aplica un factor de escala post-hoc basado en datos históricos.
#
# Limitación documentada: es univariante y lineal (no aprende interacciones).
# Referencia: README_titulacion_vs_rama.md · Fase 3 M08 (auditoría leakage)
# =============================================================================

def _ajustar_prob_por_titulacion(prob: float,
                                  contexto: dict,
                                  df_ref: "pd.DataFrame") -> tuple:
    """
    Aplica corrección heurística cuando el contexto es una titulación concreta.

    prob_ajustada = prob_modelo × (tasa_hist_titulacion / tasa_hist_rama)

    El factor se limita a [0.3, 3.0] para evitar extrapolaciones extremas.

    Returns
    -------
    prob_ajustada : float   (igual a prob si contexto no es titulación)
    ajustada      : bool    (True si se aplicó corrección)
    factor        : float   (el factor aplicado)
    aviso         : str     (texto para mostrar al usuario, "" si no aplica)
    """
    if contexto.get("tipo") != "titulacion":
        return prob, False, 1.0, ""

    titulacion = contexto.get("valor", "")
    col_rama   = "rama_meta" if "rama_meta" in df_ref.columns else "rama"

    if "titulacion" not in df_ref.columns or "abandono" not in df_ref.columns:
        return prob, False, 1.0, ""

    df_tit  = df_ref[df_ref["titulacion"] == titulacion]
    if len(df_tit) < 10:
        return prob, False, 1.0, ""

    tasa_tit  = float(df_tit["abandono"].mean())
    rama_val  = df_tit[col_rama].mode()[0] if col_rama in df_tit.columns else None

    if rama_val is not None:
        df_rama  = df_ref[df_ref[col_rama] == rama_val]
        tasa_rama = float(df_rama["abandono"].mean()) if len(df_rama) > 0 else tasa_tit
    else:
        tasa_rama = tasa_tit  # sin rama → no ajustamos

    if tasa_rama < 0.01:
        return prob, False, 1.0, ""

    factor = tasa_tit / tasa_rama
    factor = max(0.3, min(3.0, factor))   # límites de seguridad

    prob_ajustada = float(max(0.01, min(0.99, prob * factor)))

    aviso = (
        f"⚠️ La probabilidad base del modelo ({prob*100:.1f}%) opera a nivel de "
        f"**rama de conocimiento**. Se ha aplicado un ajuste por la tasa "
        f"histórica de abandono de **{titulacion.replace('Grado en ', '')}** "
        f"({tasa_tit*100:.1f}%) respecto a la media de su rama ({tasa_rama*100:.1f}%). "
        f"Este ajuste no forma parte del modelo entrenado."
    )

    return prob_ajustada, True, factor, aviso

def show_pronostico(modo: str = "prospecto"):
    """
    Renderiza la página completa de pronóstico personalizado.

    Parameters
    ----------
    modo : str
        "prospecto" → alumno antes de matricularse (sin nota_1er_anio)
        "en_curso"  → alumno ya matriculado (con nota_1er_anio y créditos)
    """
    assert modo in ("prospecto", "en_curso"), \
        "modo debe ser 'prospecto' o 'en_curso'"

    # Textos que cambian según el modo
    textos = {
        "prospecto": {
            "titulo":      "🔍 Pronóstico para futuro estudiante",
            "subtitulo":   "Estima tu riesgo de abandono antes de matricularte",
            "descripcion": (
                "Rellena tu perfil de entrada y obtendrás una estimación "
                "de tu riesgo de abandono basada en patrones de 30.872 "
                "estudiantes de la UJI entre 2010 y 2020."
            ),
        },
        "en_curso": {
            "titulo":      "📊 Pronóstico para alumno en curso",
            "subtitulo":   "Estima tu riesgo de abandono con tus datos actuales",
            "descripcion": (
                "Rellena tu perfil actual. Cuantos más datos introduzcas, "
                "más precisa será la estimación. Los resultados son "
                "orientativos y no sustituyen la orientación académica."
            ),
        },
    }[modo]

    # Cabecera
    st.markdown(f"""
    <h2 style="color:{COLORES['primario']}; margin-bottom:0.2rem;">
        {textos['titulo']}
    </h2>
    <p style="color:{COLORES['texto_suave']}; margin-top:0; font-size:0.95rem;">
        {textos['subtitulo']}
    </p>
    <p style="color:{COLORES['texto']}; font-size:0.88rem; max-width:700px;">
        {textos['descripcion']}
    </p>
    """, unsafe_allow_html=True)

    # Carga de datos y modelos — cacheados, solo se ejecuta la primera vez
    with st.spinner("Cargando modelo..."):
        try:
            df_ref   = cargar_meta_test_app()   # titulaciones fusionadas (para contexto/filtros)
            modelo   = cargar_modelo()
            pipeline = cargar_pipeline()
            # X_test_prep: features numéricas reales — para imputar medias correctas
            try:
                import pandas as _pd
                from config_app import RUTAS as _RUTAS
                _ruta_xtest = _RUTAS.get('X_test_prep')
                df_features = _pd.read_parquet(_ruta_xtest) if _ruta_xtest and _ruta_xtest.exists() else None
            except Exception:
                df_features = None
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()

    st.divider()

    # --- Paso 1: selector de contexto ---
    contexto = _selector_contexto(df_ref, modo)

    st.divider()

    # --- Bifurcación: comparativa vs pronóstico individual ---
    # Si el usuario eligió "Comparar varias titulaciones", mostramos
    # la comparativa directamente y salimos del flujo normal.
    if contexto["tipo"] == "comparativa":
        perfil, calcular = _formulario_perfil(modo, df_ref, contexto)
        if not calcular:
            _mostrar_instrucciones()
            return
        with st.spinner("Calculando comparativa..."):
            _mostrar_comparativa(perfil, contexto["titulaciones"],
                                 df_ref, modelo, pipeline,
                                 df_features=df_features if df_features is not None else df_ref)
        st.markdown(f"""
        <div style="font-size:0.75rem; color:{COLORES['texto_suave']};
                    border-top:1px solid {COLORES['borde']};
                    margin-top:1.5rem; padding-top:0.8rem;">
            ⚠️ <em>Este pronóstico es una estimación estadística basada en datos
            históricos. No determina ni condiciona ninguna decisión académica.</em>
        </div>
        """, unsafe_allow_html=True)
        return

    # --- Paso 2: formulario de perfil (básico + avanzado) ---
    perfil, calcular = _formulario_perfil(modo, df_ref, contexto)

    # Solo calculamos si el usuario pulsa el botón
    if not calcular:
        _mostrar_instrucciones()
        return

    # df_ctx: subconjunto filtrado por titulación/rama según el contexto elegido.
    # Se usa para imputar valores por defecto y calcular contribuciones
    # con las medias del grupo de referencia (no las globales de toda la UJI).
    # Si el contexto es "todas", df_ctx == df_ref completo.
    df_ctx = contexto.get("df_contexto", df_ref)
    if df_ctx is None or len(df_ctx) == 0:
        df_ctx = df_ref  # fallback seguro si el filtro dejó vacío

    # --- Paso 3: calcular probabilidad + corrección heurística por titulación ---
    with st.spinner("Calculando pronóstico..."):
        prob, error = _calcular_probabilidad(perfil, modelo, pipeline, df_ctx)

    if error:
        st.error(f"❌ Error al calcular: {error}")
        return

    # Corrección heurística: ajusta prob si el contexto es una titulación concreta.
    # El modelo opera a nivel de rama — dos titulaciones de la misma rama darían
    # el mismo %. El factor tasa_tit/tasa_rama diferencia entre ellas.
    prob, ajustada, factor, aviso_ajuste = _ajustar_prob_por_titulacion(
        prob, contexto, df_ref
    )

    st.divider()

    # --- Pasos 4-8: resultados ---
    _mostrar_resultado_principal(prob, modo)

    # Aviso de corrección heurística (solo si se aplicó)
    if ajustada and aviso_ajuste:
        st.markdown(
            f"<div style='font-size:0.78rem; color:#718096; "
            f"border-left:3px solid #d69e2e; padding:0.4rem 0.8rem; "
            f"margin:0.3rem 0; border-radius:4px; background:#fffbeb;'>"
            f"⚠️ {aviso_ajuste}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    _grafico_indicador_riesgo(prob)
    st.divider()

    col_radar, col_cascada = st.columns([1, 1])
    with col_radar:
        _grafico_radar(perfil, df_ref, contexto, prob)
    with col_cascada:
        _grafico_cascada(perfil, df_ctx, prob, modelo, pipeline)

    st.divider()
    _grafico_percentil(prob, df_ref, contexto, modelo, pipeline)
    st.divider()
    _recomendaciones(perfil, prob, modo)

    # Aviso legal / limitaciones
    st.markdown(f"""
    <div style="
        font-size:0.75rem;
        color:{COLORES['texto_suave']};
        border-top:1px solid {COLORES['borde']};
        margin-top:1.5rem;
        padding-top:0.8rem;
    ">
        ⚠️ <em>Este pronóstico es una estimación estadística basada en datos
        históricos. No determina ni condiciona ninguna decisión académica.
        Si tienes dudas sobre tu situación, consulta con el servicio de
        orientación universitaria de la UJI.</em>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PASO 1: Selector de contexto
# =============================================================================

def _selector_contexto(df_ref: pd.DataFrame, modo: str) -> dict:
    """
    Permite al usuario elegir si quiere compararse contra una titulación
    concreta, una rama, o toda la UJI.

    Devuelve un dict con:
        tipo        : "titulacion" | "rama" | "todas"
        valor       : nombre de la titulación/rama, o None
        df_contexto : subconjunto de df_ref según el contexto elegido
    """
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin-bottom:0.5rem;">
        1️⃣ ¿Tienes una titulación en mente?
    </h4>
    <p style="font-size:0.85rem; color:{COLORES['texto_suave']};">
        Opcional. Si eliges una titulación o rama, las comparativas
        se calcularán respecto a alumnos de ese mismo contexto.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        tipo_contexto = st.radio(
            label="Comparar contra:",
            options=[
                "Todas las titulaciones",
                "Una rama concreta",
                "Una titulación concreta",
                "🔀 Comparar varias titulaciones",
            ],
            index=0,
            key=f"tipo_contexto_{modo}",
            help=(
                "Elige el grupo con el que quieres compararte. "
                "Con 'Comparar varias titulaciones' puedes ver tu riesgo "
                "en 2-5 carreras a la vez."
            ),
        )

    with col2:
        valor_contexto = None

        # Usamos rama_meta (nombre legible) si está disponible, si no rama
        col_rama = 'rama_meta' if 'rama_meta' in df_ref.columns else 'rama'

        if tipo_contexto == "Una rama concreta" and col_rama in df_ref.columns:
            ramas = sorted(df_ref[col_rama].dropna().unique().tolist())
            valor_contexto = st.selectbox(
                label="Selecciona la rama",
                options=ramas,
                key=f"sel_rama_{modo}",
            )

        elif tipo_contexto == "Una titulación concreta" and 'titulacion' in df_ref.columns:
            # Primero filtramos por rama si el usuario la eligió
            # Así la lista de titulaciones es manejable (~8-10 por rama)
            ramas_disponibles = sorted(df_ref[col_rama].dropna().unique().tolist()) \
                if col_rama in df_ref.columns else []

            if ramas_disponibles:
                rama_filtro = st.selectbox(
                    label="Filtrar por rama (opcional)",
                    options=["Todas las ramas"] + ramas_disponibles,
                    key=f"sel_rama_filtro_{modo}",
                    help="Filtra la lista de titulaciones por rama para encontrarla más fácil",
                )
                if rama_filtro != "Todas las ramas":
                    df_filtrado = df_ref[df_ref[col_rama] == rama_filtro]
                else:
                    df_filtrado = df_ref
            else:
                df_filtrado = df_ref

            titulaciones = sorted(df_filtrado['titulacion'].dropna().unique().tolist())
            valor_contexto = st.selectbox(
                label="Selecciona la titulación",
                options=titulaciones,
                key=f"sel_tit_{modo}",
                help="Puedes escribir para buscar dentro de la lista",
            )

        # --- Opción comparativa: multiselect de titulaciones ---
        titulaciones_comparar = []
        if "Comparar" in tipo_contexto and 'titulacion' in df_ref.columns:
            col_rama_c = 'rama_meta' if 'rama_meta' in df_ref.columns else 'rama'

            # Filtro opcional por rama para acotar la lista
            ramas_disp = sorted(df_ref[col_rama_c].dropna().unique().tolist())                 if col_rama_c in df_ref.columns else []
            if ramas_disp:
                rama_filtro_c = st.selectbox(
                    label="Filtrar por rama (opcional)",
                    options=["Todas las ramas"] + ramas_disp,
                    key=f"rama_comp_{modo}",
                    help="Reduce la lista de titulaciones disponibles",
                )
                df_comp_filtrado = df_ref[df_ref[col_rama_c] == rama_filtro_c]                     if rama_filtro_c != "Todas las ramas" else df_ref
            else:
                df_comp_filtrado = df_ref

            todas_tit = sorted(df_comp_filtrado['titulacion'].dropna().unique().tolist())
            titulaciones_comparar = st.multiselect(
                label="Selecciona 2-5 titulaciones para comparar",
                options=todas_tit,
                default=[],
                max_selections=5,
                key=f"multisel_tit_{modo}",
                help="Mínimo 2, máximo 5 titulaciones.",
            )

            # Aviso si selección fuera de rango
            n = len(titulaciones_comparar)
            if n == 1:
                st.warning("⚠️ Selecciona al menos 2 titulaciones para comparar.")
            elif n >= 2:
                st.success(f"✅ {n} titulaciones seleccionadas. Rellena tu perfil y pulsa calcular.")

        # Tarjeta informativa del contexto elegido
        if valor_contexto and "Comparar" not in tipo_contexto:
            if tipo_contexto == "Una rama concreta":
                df_ctx = df_ref[df_ref[col_rama] == valor_contexto]
            else:
                df_ctx = df_ref[df_ref['titulacion'] == valor_contexto]

            tasa_ctx = (df_ctx['abandono'].sum() / len(df_ctx) * 100) \
                if 'abandono' in df_ctx.columns and len(df_ctx) > 0 else 0

            st.markdown(f"""
            <div style="
                background:{COLORES['fondo']};
                border-left:3px solid {COLORES['primario']};
                border-radius:4px;
                padding:0.6rem 1rem;
                font-size:0.83rem;
                margin-top:0.3rem;
            ">
                <strong>Contexto:</strong> {valor_contexto}<br>
                <strong>Alumnos en histórico:</strong> {len(df_ctx):,}<br>
                <strong>Tasa abandono histórica:</strong> {tasa_ctx:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            tasa_uji = (df_ref['abandono'].sum() / len(df_ref) * 100) \
                if 'abandono' in df_ref.columns else 0
            st.markdown(f"""
            <div style="
                background:{COLORES['fondo']};
                border-left:3px solid {COLORES['texto_suave']};
                border-radius:4px;
                padding:0.6rem 1rem;
                font-size:0.83rem;
                margin-top:0.3rem;
            ">
                <strong>Contexto:</strong> Toda la UJI<br>
                <strong>Alumnos en histórico:</strong> {len(df_ref):,}<br>
                <strong>Tasa abandono histórica:</strong> {tasa_uji:.1f}%
            </div>
            """, unsafe_allow_html=True)

    # Construimos el dict de contexto
    col_rama = 'rama_meta' if 'rama_meta' in df_ref.columns else 'rama'
    if "Comparar" in tipo_contexto:
        # Modo comparativa — la lista de titulaciones ya está en titulaciones_comparar
        return {
            "tipo":         "comparativa",
            "valor":        None,
            "df_contexto":  df_ref.copy(),
            "titulaciones": titulaciones_comparar,
        }
    elif tipo_contexto == "Una rama concreta" and valor_contexto:
        tipo_key = "rama"
        df_ctx   = df_ref[df_ref[col_rama] == valor_contexto]
    elif tipo_contexto == "Una titulación concreta" and valor_contexto:
        tipo_key = "titulacion"
        df_ctx   = df_ref[df_ref['titulacion'] == valor_contexto]
    else:
        tipo_key = "todas"
        df_ctx   = df_ref.copy()

    return {"tipo": tipo_key, "valor": valor_contexto, "df_contexto": df_ctx}


# =============================================================================
# PASO 2: Formulario de perfil
# =============================================================================

def _formulario_perfil(modo: str, df_ref: pd.DataFrame,
                        contexto: dict) -> tuple[dict, bool]:
    """
    Formulario con los datos del alumno, dividido en dos bloques:
      - Básicas obligatorias: siempre visibles
      - Avanzadas opcionales: dentro de un expander

    Devuelve (perfil_dict, calcular_bool).
    calcular_bool es True cuando el usuario pulsa el botón.
    """
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin-bottom:0.5rem;">
        2️⃣ Rellena tu perfil
    </h4>
    <p style="font-size:0.83rem; color:{COLORES['texto_suave']}; margin-top:0;">
        Los campos marcados con <strong>*</strong> son los más influyentes en el modelo.
        Los opcionales mejoran la precisión de la estimación.
    </p>
    """, unsafe_allow_html=True)

    df_ctx = contexto['df_contexto']

    # Helper: media del contexto como valor por defecto
    def _med(col, default):
        return float(df_ctx[col].mean()) \
            if col in df_ctx.columns and df_ctx[col].notna().any() else default

    perfil = {}

    # -------------------------------------------------------------------------
    # BLOQUE BÁSICO — siempre visible
    # -------------------------------------------------------------------------
    st.markdown(f"<p style='font-size:0.85rem; font-weight:600; color:{COLORES['texto']}; margin-bottom:0.3rem;'>📋 Datos principales</p>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # BLOQUE PRINCIPAL — distinto según modo
    # en_curso:  rendimiento académico primero (los más predictivos)
    # prospecto: datos de acceso primero (lo único disponible antes de entrar)
    # -------------------------------------------------------------------------

    if modo == "en_curso":
        # Encabezado diferenciador visual
        st.markdown(f"""
        <div style="
            background:#ebf8ff;
            border-left:4px solid {COLORES['primario']};
            border-radius:6px;
            padding:0.6rem 1rem;
            font-size:0.83rem;
            color:{COLORES['primario']};
            margin-bottom:0.8rem;
        ">
            📊 Introduce tus datos de rendimiento académico — son los predictores
            más importantes para un alumno que ya está cursando.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        # Columna 1: rendimiento académico (protagonista en p04)
        with col1:
            perfil['nota_1er_anio'] = st.slider(
                label="Nota media del primer año *",
                min_value=0.0, max_value=10.0,
                value=round(_med('nota_1er_anio', 6.0), 1),
                step=0.1,
                help="Uno de los predictores más fuertes del modelo para alumnos en curso.",
                key=f"nota_1er_{modo}",
            )
            perfil['cred_superados_anio_1er'] = st.number_input(
                label="Créditos superados en 1.º *",
                min_value=0, max_value=80,
                value=int(_med('cred_superados_anio_1er', 40)),
                step=1,
                help="Créditos aprobados durante el primer año académico.",
                key=f"cred_1er_{modo}",
            )

        # Columna 2: situación personal (también muy predictiva)
        with col2:
            perfil['situacion_laboral'] = st.selectbox(
                label="Situación laboral *",
                options=_OPCIONES_LABORAL,
                index=0,
                help="La situación laboral es el predictor categórico más fuerte del modelo (Cramér V=0.26).",
                key=f"laboral_{modo}",
            )
            perfil['n_anios_beca'] = st.slider(
                label="Años con beca *",
                min_value=0, max_value=6,
                value=int(round(_med('n_anios_beca', 2))),
                step=1,
                help="Factor protector muy importante. Los becarios tienen tasas de abandono significativamente más bajas.",
                key=f"beca_{modo}",
            )

        # Columna 3: datos de acceso (secundarios para un alumno en curso)
        with col3:
            _val_nota = float(np.clip(round(_med('nota_acceso', 8.0), 1), 5.0, 14.0))
            perfil['nota_acceso'] = st.slider(
                label="Nota de acceso (PAU / FP)",
                min_value=5.0, max_value=14.0,
                value=_val_nota,
                step=0.1,
                help="Nota de acceso a la universidad (escala 0–14).",
                key=f"nota_acceso_{modo}",
            )
            perfil['edad_entrada'] = st.number_input(
                label="Edad al acceder",
                min_value=17, max_value=65,
                value=int(np.clip(_med('edad_entrada', 19), 17, 65)),
                step=1,
                key=f"edad_{modo}",
            )

    else:
        # Modo prospecto: orden original (acceso primero)
        col1, col2, col3 = st.columns(3)

        with col1:
            _val_nota = float(np.clip(round(_med('nota_acceso', 8.0), 1), 5.0, 14.0))
            perfil['nota_acceso'] = st.slider(
                label="Nota de acceso (PAU / FP) *",
                min_value=5.0, max_value=14.0,
                value=_val_nota,
                step=0.1,
                help="Nota de acceso a la universidad (escala 0–14). Es uno de los predictores más importantes.",
                key=f"nota_acceso_{modo}",
            )
            perfil['via_acceso'] = st.selectbox(
                label="Vía de acceso *",
                options=_OPCIONES_VIA_ACCESO,
                index=0,
                key=f"tipo_acceso_{modo}",
            )

        with col2:
            perfil['situacion_laboral'] = st.selectbox(
                label="Situación laboral *",
                options=_OPCIONES_LABORAL,
                index=0,
                help="La situación laboral es el predictor categórico más fuerte del modelo (Cramér V=0.26).",
                key=f"laboral_{modo}",
            )
            perfil['n_anios_beca'] = st.slider(
                label="Años con beca previstos *",
                min_value=0, max_value=6,
                value=int(round(_med('n_anios_beca', 2))),
                step=1,
                help="Factor protector muy importante. Los becarios tienen tasas de abandono significativamente más bajas.",
                key=f"beca_{modo}",
            )

        with col3:
            perfil['edad_entrada'] = st.number_input(
                label="Edad al acceder",
                min_value=17, max_value=65,
                value=int(np.clip(_med('edad_entrada', 19), 17, 65)),
                step=1,
                key=f"edad_{modo}",
            )
            # Prospecto: aviso claro de que no se incluye rendimiento
            st.markdown(f"""
            <div style="
                background:{COLORES['fondo']};
                border-radius:6px;
                padding:0.7rem 0.9rem;
                font-size:0.80rem;
                color:{COLORES['texto_suave']};
                margin-top:0.3rem;
            ">
                📌 Como todavía no estás matriculado/a, los datos de
                rendimiento académico no están disponibles. El modelo
                los imputará con la media histórica de tu contexto.
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # BLOQUE AVANZADO — en expander
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)

    label_exp = "⚙️ Datos avanzados (opcionales — mejoran la precisión)"
    with st.expander(label_exp, expanded=False):
        st.markdown(f"""
        <p style="font-size:0.82rem; color:{COLORES['texto_suave']}; margin-bottom:0.8rem;">
            Estos campos son opcionales. Si no los rellenas, el modelo
            utilizará la media histórica de tu contexto como aproximación.
        </p>
        """, unsafe_allow_html=True)

        adv1, adv2 = st.columns(2)

        with adv1:
            perfil['nota_selectividad'] = st.slider(
                label="Nota de selectividad",
                min_value=0.0, max_value=10.0,
                value=round(_med('nota_selectividad', 6.5), 1),
                step=0.1,
                key=f"selectividad_{modo}",
            )

            perfil['orden_preferencia'] = st.number_input(
                label="Orden de preferencia de la titulación",
                min_value=1, max_value=20,
                value=max(1, int(_med('orden_preferencia', 1))),
                step=1,
                help="Posición en la que pediste esta titulación en la preinscripción (1 = primera opción).",
                key=f"orden_pref_{modo}",
            )

            perfil['anios_gap'] = st.number_input(
                label="Años de pausa antes de matricularte",
                min_value=0, max_value=30,
                value=int(_med('anios_gap', 0)),
                step=1,
                help="Años transcurridos entre acabar bachillerato/FP y matricularte. 0 = acceso directo.",
                key=f"gap_{modo}",
            )

        with adv2:
            perfil['universidad_origen'] = st.selectbox(
                label="Universidad / centro de procedencia",
                options=_OPCIONES_UNIVERSIDAD,
                index=0,
                key=f"univ_origen_{modo}",
            )

            perfil['sexo'] = st.selectbox(
                label="Sexo",
                options=_OPCIONES_SEXO,
                index=0,
                key=f"sexo_{modo}",
            )

            if modo == "en_curso":
                # via_acceso está en el bloque básico de prospecto pero no en en_curso
                # → la recogemos aquí en el expander
                perfil['via_acceso'] = st.selectbox(
                    label="Vía de acceso",
                    options=_OPCIONES_VIA_ACCESO,
                    index=0,
                    key=f"tipo_acceso_{modo}",
                )
                perfil['creditos_superados'] = st.number_input(
                    label="Créditos superados (total acumulado)",
                    min_value=0, max_value=300,
                    value=int(_med('creditos_superados', 40)),
                    step=1,
                    help="Total de créditos aprobados hasta la fecha, incluyendo el primer año.",
                    key=f"creditos_{modo}",
                )

    # -------------------------------------------------------------------------
    # Rellenar valores por defecto para features no mostradas
    # (el pipeline necesita todas las features, incluso las no introducidas)
    # -------------------------------------------------------------------------
    defaults_numericos = {
        # Variables NO disponibles para el prospecto — pasar NaN
        # para que el SimpleImputer del pipeline use su mediana del training set
        # (nota_1er_anio=6.79, cred_superados=54, max_pagos median)
        'nota_1er_anio':           np.nan,
        'cred_superados_anio_1er': np.nan,
        'creditos_superados':      np.nan,
        'tasa_rendimiento':        np.nan,
        'max_pagos':               np.nan,
        'creditos_matriculados':   np.nan,
        'indicador_interrupcion':  0,      # sí se conoce: no ha interrumpido
        'anio_cohorte':            2020,   # año aproximado de entrada
    }
    for col, val in defaults_numericos.items():
        if col not in perfil:
            perfil[col] = val

    # tasa_rendimiento calculada si se tienen los datos
    if modo == "en_curso" and 'creditos_superados' in perfil:
        matriculados = perfil.get('creditos_matriculados',
                                   _med('creditos_matriculados', 60.0))
        perfil['tasa_rendimiento'] = (
            perfil['creditos_superados'] / max(float(matriculados), 1)
        )

    # Botón centrado
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        calcular = st.button(
            label="🔮 Calcular mi pronóstico",
            type="primary",
            use_container_width=True,
            key=f"btn_calcular_{modo}",
        )

    return perfil, calcular


# =============================================================================
# PASO 3: Calcular probabilidad
# =============================================================================


def _traducir_perfil_a_codigos(perfil: dict) -> dict:
    """
    Convierte los valores de texto del formulario a los códigos numéricos
    que espera el pipeline del modelo.

    El modelo fue entrenado con enteros (via_acceso=10, sexo=0, etc.).
    El formulario recoge strings legibles ("Bachiller", "Mujer").
    Esta función hace la traducción usando los mapas de config_datos.py.

    Las variables numéricas (nota_acceso, n_anios_beca, etc.) se copian tal cual.
    """
    p = dict(perfil)  # copia para no mutar el original

    # Categóricas → código numérico
    if 'situacion_laboral' in p and isinstance(p['situacion_laboral'], str):
        p['situacion_laboral'] = SITUACION_LABORAL_MAP.get(
            p['situacion_laboral'],
            11  # default: No trabaja — código más frecuente
        )

    if 'via_acceso' in p and isinstance(p['via_acceso'], str):
        p['via_acceso'] = VIA_ACCESO_MAP.get(
            p['via_acceso'],
            10  # default: Bachillerato/PAU — código más frecuente
        )

    if 'sexo' in p and isinstance(p['sexo'], str):
        p['sexo'] = SEXO_MAP.get(p['sexo'], 0)

    if 'universidad_origen' in p and isinstance(p['universidad_origen'], str):
        p['universidad_origen'] = UNIVERSIDAD_ORIGEN_MAP.get(
            p['universidad_origen'],
            0  # default: Otra / sin datos
        )

    if 'rama' in p and isinstance(p['rama'], str):
        p['rama'] = RAMA_MAP.get(p['rama'], 3)  # default SO

    if 'provincia' in p and isinstance(p['provincia'], str):
        p['provincia'] = PROVINCIA_MAP.get(
            p['provincia'],
            0  # default: Otra / sin datos
        )

    if 'pais_nombre' in p and isinstance(p['pais_nombre'], str):
        p['pais_nombre'] = PAIS_NOMBRE_MAP.get(
            p['pais_nombre'],
            1  # default: España
        )

    # tuvo_beca: derivado de n_anios_beca
    if 'tuvo_beca' not in p:
        p['tuvo_beca'] = 1 if p.get('n_anios_beca', 0) > 0 else 0

    # anios_sin_beca: derivado si no está
    if 'anios_sin_beca' not in p:
        carrera_anios = 4
        p['anios_sin_beca'] = max(0, carrera_anios - int(p.get('n_anios_beca', 0)))

    return p


def _calcular_probabilidad(perfil: dict, modelo, pipeline,
                            df_ref: pd.DataFrame) -> tuple[float | None, str | None]:
    """
    Construye un DataFrame con el perfil del usuario en el formato
    que espera el pipeline, lo transforma y predice la probabilidad.

    Estrategia de imputación (por orden de prioridad):
      1. El usuario lo rellenó → usar directamente
      2. Está en df_ref → usar media/moda del contexto
      3. No está en df_ref pero el scaler tiene su media → usar media del training set
      4. Sin información → 0

    Devuelve (probabilidad, mensaje_error).
    """
    try:
        # Traducir strings del formulario a códigos numéricos del modelo
        perfil = _traducir_perfil_a_codigos(perfil)

        # Columnas que el pipeline espera — fuente de verdad
        if hasattr(pipeline, 'feature_names_in_'):
            cols_pipeline = list(pipeline.feature_names_in_)
        else:
            cols_pipeline = [c for c in df_ref.columns if c not in _COLS_META]

        # Extraer medias del scaler para imputación correcta
        # El scaler fue entrenado con el training set — sus medias son los valores
        # más representativos para imputar cuando no tenemos otra información
        medias_scaler = {}
        try:
            num_pipeline = pipeline.named_transformers_.get('num')
            if num_pipeline and hasattr(num_pipeline, 'steps'):
                scaler = dict(num_pipeline.steps).get('scaler')
                if scaler and hasattr(scaler, 'mean_'):
                    # Las columnas del scaler son las mismas que feature_names_in_
                    for col, mean in zip(cols_pipeline, scaler.mean_):
                        medias_scaler[col] = float(mean)
        except Exception:
            pass  # Si falla, seguimos sin medias del scaler

        # Construimos la fila con EXACTAMENTE las columnas del pipeline
        fila = {}
        for col in cols_pipeline:
            if col in perfil:
                # Prioridad 1: el usuario lo rellenó
                fila[col] = perfil[col]
            elif col in df_ref.columns and df_ref[col].notna().any():
                # Prioridad 2: media/moda del contexto (df_ref)
                if df_ref[col].dtype.kind in ('f', 'i'):
                    fila[col] = float(df_ref[col].mean())
                else:
                    fila[col] = df_ref[col].mode()[0]
            elif col in medias_scaler:
                # Prioridad 3: media del training set (del scaler)
                # Mejor que 0 — evita predicciones absurdas
                fila[col] = medias_scaler[col]
            else:
                # Prioridad 4: sin información — valor neutro
                fila[col] = 0

        X_usuario = pd.DataFrame([fila], columns=cols_pipeline)
        X_prep    = pipeline.transform(X_usuario)
        prob      = modelo.predict_proba(X_prep)[0, 1]

        return float(prob), None

    except Exception as e:
        return None, str(e)


# =============================================================================
# RESULTADO PRINCIPAL
# =============================================================================

def _mostrar_resultado_principal(prob: float, modo: str):
    """Banner con el resultado principal antes de los gráficos."""
    pct = prob * 100
    nivel, color, emoji, mensaje = _clasificar_riesgo(prob)

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg, {color}18, {color}08);
        border:2px solid {color}60;
        border-radius:12px;
        padding:1.5rem 2rem;
        text-align:center;
        margin:0.5rem 0;
    ">
        <div style="font-size:2.5rem;">{emoji}</div>
        <div style="font-size:2rem; font-weight:bold; color:{color};">
            {pct:.1f}%
        </div>
        <div style="font-size:1.1rem; font-weight:bold; color:{COLORES['texto']};">
            Riesgo de abandono: <span style="color:{color};">{nivel}</span>
        </div>
        <div style="font-size:0.88rem; color:{COLORES['texto_suave']}; margin-top:0.4rem;">
            {mensaje}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# GRÁFICO 1: Indicador de riesgo (velocímetro)
# =============================================================================


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convierte color hex a rgba para Plotly (que no acepta hex de 8 chars)."""
    h = hex_color.lstrip('#')
    if len(h) == 6:
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color  # fallback si ya es rgba u otro formato



def _grafico_velocimetro_comparativa(resultados: list):
    """
    Velocímetro único con un marcador de color distinto para cada titulación.
    El fondo muestra las zonas verde/amarillo/rojo como siempre.
    Cada titulación tiene su aguja/anotación en su color de la paleta comparativa.
    """
    fig = go.Figure()

    # Zonas de fondo del velocímetro
    fig.add_trace(go.Indicator(
        mode="gauge",
        value=0,  # valor dummy — no mostramos número central
        gauge={
            'axis': {
                'range': [0, 100],
                'tickvals': [0, 30, 60, 100],
                'ticktext': ['0%', '30%', '60%', '100%'],
                'tickwidth': 1,
                'tickcolor': COLORES['texto_suave'],
            },
            'bar': {'color': 'rgba(0,0,0,0)', 'thickness': 0},  # barra invisible
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  UMBRALES['riesgo_bajo']  * 100],
                 'color': 'rgba(56,161,105,0.15)'},
                {'range': [UMBRALES['riesgo_bajo']  * 100,
                           UMBRALES['riesgo_medio'] * 100],
                 'color': 'rgba(236,201,75,0.15)'},
                {'range': [UMBRALES['riesgo_medio'] * 100, 100],
                 'color': 'rgba(229,62,62,0.15)'},
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]},
    ))

    # Añadir una aguja por titulación usando shapes + annotations
    import math
    for r in resultados:
        pct   = r["pct"]
        color = r["color_comp"]
        nc    = r["titulacion"]
        for pref in ["Grado en ", "Doble Grado en ", "Grado Universitario en "]:
            if nc.startswith(pref):
                nc = nc[len(pref):]
                break
        nc = nc[:25] + "…" if len(nc) > 25 else nc

        # El velocímetro va de 180° (izq, 0%) a 0° (der, 100%)
        # Ángulo en radianes: 180° - (pct/100 * 180°)
        angulo_deg = 180 - (pct / 100 * 180)
        angulo_rad = math.radians(angulo_deg)
        # Centro del velocímetro en coordenadas paper (0-1)
        cx, cy = 0.5, 0.27  # centro aproximado del gauge
        longitud = 0.35

        x1 = cx + longitud * math.cos(angulo_rad)
        y1 = cy + longitud * math.sin(angulo_rad)

        fig.add_shape(
            type="line",
            x0=cx, y0=cy, x1=x1, y1=y1,
            line=dict(color=color, width=3),
            xref="paper", yref="paper",
        )
        # Etiqueta fuera de la aguja
        x_label = cx + (longitud + 0.07) * math.cos(angulo_rad)
        y_label = cy + (longitud + 0.07) * math.sin(angulo_rad)
        fig.add_annotation(
            x=x_label, y=y_label,
            text=f"<b style='color:{color}'>{pct:.1f}%</b><br><span style='font-size:9px'>{nc}</span>",
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=10, color=color),
            align="center",
        )

    # Leyenda textual debajo
    partes_leyenda = []
    for r in resultados:
        col  = r["color_comp"]
        nom  = r["titulacion"].split("Grado en ")[-1][:30]
        pct  = r["pct"]
        partes_leyenda.append(
            f"<span style='color:{col};font-weight:bold;'>■</span> {nom} ({pct:.1f}%)"
        )
    leyenda_html = " &nbsp;·&nbsp; ".join(partes_leyenda)

    fig.update_layout(
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=10),
        height=260,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"<p style='font-size:0.78rem; color:{COLORES['texto_suave']}; "
        f"text-align:center; margin-top:-0.5rem;'>{leyenda_html}</p>",
        unsafe_allow_html=True
    )


def _grafico_indicador_riesgo(prob: float):
    """
    Velocímetro semicircular con la probabilidad de abandono.
    La aguja apunta al valor predicho. Verde → amarillo → rojo.
    Delta muestra la diferencia respecto a la media UJI (29.2%).
    """
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin-bottom:0.3rem;">
        🎯 Indicador de riesgo
    </h4>
    """, unsafe_allow_html=True)

    pct = prob * 100
    _, color, _, _ = _clasificar_riesgo(prob)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={'suffix': '%', 'font': {'size': 36, 'color': color}},
        delta={
            'reference': 29.2,
            'increasing': {'color': COLORES['abandono']},
            'decreasing': {'color': COLORES['exito']},
            'suffix': ' pp vs media UJI',
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': COLORES['texto_suave'],
                'tickvals': [0, 30, 60, 100],
                'ticktext': ['0%', '30%', '60%', '100%'],
            },
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  UMBRALES['riesgo_bajo']  * 100],
                 'color': 'rgba(56,161,105,0.15)'},   # verde suave
                {'range': [UMBRALES['riesgo_bajo']  * 100,
                           UMBRALES['riesgo_medio'] * 100],
                 'color': 'rgba(236,201,75,0.15)'},   # amarillo suave
                {'range': [UMBRALES['riesgo_medio'] * 100, 100],
                 'color': 'rgba(229,62,62,0.15)'},    # rojo suave
            ],
            'threshold': {
                'line': {'color': COLORES['texto'], 'width': 3},
                'thickness': 0.75,
                'value': pct,
            },
        },
        title={
            'text': (
                "Probabilidad de abandono predicha<br>"
                "<span style='font-size:0.8em;color:gray'>"
                "Referencia: media UJI = 29.2%</span>"
            ),
            'font': {'size': 14},
        },
        domain={'x': [0, 1], 'y': [0, 1]},
    ))

    fig.update_layout(
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=60, b=20),
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# GRÁFICO 2: Radar — tu perfil vs éxito Y vs abandono (2 líneas de referencia)
# =============================================================================

def _grafico_radar(perfil: dict, df_ref: pd.DataFrame,
                   contexto: dict, prob: float):
    """
    Gráfico de araña con 3 trazas:
      - Perfil de éxito  (azul) — media de los que NO abandonaron
      - Perfil de abandono (rojo) — media de los que SÍ abandonaron
      - Tu perfil         (verde/amarillo/rojo según riesgo)

    Cada eje es una variable numérica normalizada a 0-1 usando los
    rangos del dataset completo (para comparabilidad).
    """
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin-bottom:0.3rem;">
        🕸️ Tu perfil vs perfiles de referencia
    </h4>
    """, unsafe_allow_html=True)

    df_ctx = contexto['df_contexto']

    # Variables a mostrar — ajustadas según modo (prospecto sin nota_1er_anio)
    vars_radar = ['nota_acceso', 'n_anios_beca', 'edad_acceso',
                  'nota_selectividad', 'tasa_rendimiento']
    if 'nota_1er_anio' in perfil and perfil.get('nota_1er_anio', None) is not None:
        vars_radar = ['nota_acceso', 'nota_1er_anio', 'n_anios_beca',
                      'tasa_rendimiento', 'edad_acceso']

    vars_ok = [v for v in vars_radar if v in df_ctx.columns]

    if len(vars_ok) < 3:
        st.info("No hay suficientes variables para el gráfico de radar.")
        return

    # Perfiles de referencia en el contexto elegido
    if 'abandono' in df_ctx.columns:
        df_exito    = df_ctx[df_ctx['abandono'] == 0]
        df_abandona = df_ctx[df_ctx['abandono'] == 1]
        exito_vals    = df_exito[vars_ok].mean()
        abandono_vals = df_abandona[vars_ok].mean()
    else:
        exito_vals    = df_ctx[vars_ok].mean()
        abandono_vals = df_ctx[vars_ok].mean()

    usuario_vals = pd.Series({v: float(perfil.get(v, df_ref[v].mean()
                                                   if v in df_ref.columns else 0))
                               for v in vars_ok})

    # Normalización a 0-1 usando los rangos del dataset completo
    for v in vars_ok:
        vmin  = df_ref[v].min() if v in df_ref.columns else 0
        vmax  = df_ref[v].max() if v in df_ref.columns else 1
        rango = vmax - vmin if vmax != vmin else 1
        exito_vals[v]    = (exito_vals[v]    - vmin) / rango
        abandono_vals[v] = (abandono_vals[v] - vmin) / rango
        usuario_vals[v]  = (usuario_vals[v]  - vmin) / rango

    nombres_ejes = [NOMBRES_VARIABLES.get(v, v.replace('_', ' ').title())
                    for v in vars_ok]
    # Cerramos el polígono repitiendo el primer valor
    nombres_c    = nombres_ejes + [nombres_ejes[0]]
    exito_c      = list(exito_vals.values)    + [exito_vals.values[0]]
    abandono_c   = list(abandono_vals.values) + [abandono_vals.values[0]]
    usuario_c    = list(usuario_vals.values)  + [usuario_vals.values[0]]

    _, color_usuario, _, _ = _clasificar_riesgo(prob)

    fig = go.Figure()

    # Traza 1: Perfil de éxito (azul)
    fig.add_trace(go.Scatterpolar(
        r=exito_c, theta=nombres_c,
        fill='toself',
        fillcolor='rgba(49,130,206,0.12)',
        line=dict(color=COLORES['primario'], width=2),
        name='No abandona (éxito)',
    ))

    # Traza 2: Perfil de abandono (rojo)
    fig.add_trace(go.Scatterpolar(
        r=abandono_c, theta=nombres_c,
        fill='toself',
        fillcolor='rgba(229,62,62,0.08)',
        line=dict(color=COLORES['abandono'], width=2, dash='dot'),
        name='Abandona',
    ))

    # Traza 3: Tu perfil (color según riesgo)
    fig.add_trace(go.Scatterpolar(
        r=usuario_c, theta=nombres_c,
        fill='toself',
        fillcolor=_hex_to_rgba(color_usuario, 0.18),
        line=dict(color=color_usuario, width=2.5, dash='dash'),
        name='Tu perfil',
    ))

    nombre_ctx = contexto['valor'] or "toda la UJI"
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['0%', '25%', '50%', '75%', '100%'],
                tickfont=dict(size=9),
            ),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    font=dict(size=10)),
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=30, b=70),
        height=360,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"💡 Comparativa sobre alumnos de {nombre_ctx}. "
        "Cuanto más cerca del borde exterior, mejor en esa variable."
    )


# =============================================================================
# GRÁFICO 3: Cascada de contribuciones
#            Con selector de método: rápido (proxy) vs preciso (SHAP)
# =============================================================================

def _grafico_cascada(perfil: dict, df_ref: pd.DataFrame,
                     prob: float, modelo, pipeline):
    """
    Gráfico de cascada (waterfall) que muestra cómo cada variable del
    perfil sube o baja el riesgo respecto a la probabilidad base.

    Método rápido  : diferencia de medias por grupo (instántaneo)
    Método preciso : SHAP TreeExplainer en tiempo real (~1-3s)
    """
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin-bottom:0.3rem;">
        📊 ¿Qué factores influyen en tu riesgo?
    </h4>
    """, unsafe_allow_html=True)

    # Selector de método — justo debajo del título
    metodo = st.radio(
        label="Método de cálculo:",
        options=["⚡ Rápido (estimación instantánea)",
                 "🔬 Preciso (SHAP real, ~2s)"],
        index=0,
        horizontal=True,
        key=f"metodo_cascada_{id(perfil)}_{id(df_ref)}",
        help=(
            "Rápido: diferencia de medias por grupo (aproximación marginal). "
            "Preciso: SHAP TreeExplainer, calcula la contribución exacta "
            "de cada variable para TU perfil concreto."
        ),
    )

    usar_shap = "Preciso" in metodo

    if usar_shap:
        contribuciones = _contribuciones_shap(perfil, df_ref, modelo, pipeline)
    else:
        contribuciones = _contribuciones_proxy(perfil, df_ref)

    if not contribuciones:
        st.info("No hay suficientes datos para calcular las contribuciones.")
        return

    _renderizar_waterfall(contribuciones, prob, df_ref)


def _contribuciones_proxy(perfil: dict, df_ref: pd.DataFrame) -> list[dict]:
    """
    Método rápido: para cada variable, calcula cuánto difiere la tasa
    de abandono media del grupo del usuario respecto a la media general.
    Es una aproximación marginal (no considera interacciones).
    """
    prob_base = df_ref['abandono'].mean() \
        if 'abandono' in df_ref.columns else 0.292

    vars_cascada = ['nota_acceso', 'situacion_laboral', 'n_anios_beca',
                    'nota_1er_anio', 'edad_entrada', 'tasa_rendimiento',
                    'via_acceso', 'nota_selectividad']
    vars_ok = [v for v in vars_cascada if v in perfil and v in df_ref.columns]

    contribuciones = []
    for v in vars_ok:
        val_usuario = perfil[v]
        # Forzar tipo numérico para comparación segura
        try:
            val_usuario = float(val_usuario)
        except (TypeError, ValueError):
            val_usuario = 0.0

        if df_ref[v].dtype.kind in ('f', 'i'):
            q33 = float(df_ref[v].quantile(0.33))
            q66 = float(df_ref[v].quantile(0.66))
            if val_usuario <= q33:
                grupo = df_ref[df_ref[v] <= q33]
                etiqueta = f"bajo ({val_usuario:.1f})"
            elif val_usuario <= q66:
                grupo = df_ref[(df_ref[v] > q33) & (df_ref[v] <= q66)]
                etiqueta = f"medio ({val_usuario:.1f})"
            else:
                grupo = df_ref[df_ref[v] > q66]
                etiqueta = f"alto ({val_usuario:.1f})"
        else:
            grupo    = df_ref[df_ref[v] == val_usuario]
            etiqueta = str(val_usuario)

        contribucion = (grupo['abandono'].mean() - prob_base) \
            if len(grupo) > 0 and 'abandono' in grupo.columns else 0.0

        nombre = NOMBRES_VARIABLES.get(v, v.replace('_', ' ').title())
        contribuciones.append({
            'variable':     nombre,
            'valor':        etiqueta,
            'contribucion': float(contribucion),
        })

    # Ordenar por valor absoluto, top 6
    contribuciones.sort(key=lambda x: abs(x['contribucion']), reverse=True)
    return contribuciones[:6]


def _contribuciones_shap(perfil: dict, df_ref: pd.DataFrame,
                          modelo, pipeline) -> list[dict]:
    """
    Método preciso: SHAP TreeExplainer sobre el estimador final del Stacking.
    Calcula las contribuciones exactas para el perfil del usuario.
    """
    try:
        import shap  # importación lazy — solo si elige método preciso
    except ImportError:
        st.error("❌ La librería SHAP no está instalada. Usa el método rápido.")
        return []

    with st.spinner("🔬 Calculando contribuciones SHAP..."):
        try:
            # Construimos X_usuario como en _calcular_probabilidad
            cols_features = list(pipeline.feature_names_in_)                 if hasattr(pipeline, 'feature_names_in_') else                 [c for c in df_ref.columns if c not in _COLS_META]
            fila = {}
            for col in cols_features:
                if col in perfil:
                    fila[col] = perfil[col]
                elif col in df_ref.columns:
                    if df_ref[col].dtype.kind in ('f', 'i'):
                        fila[col] = df_ref[col].mean()
                    else:
                        fila[col] = df_ref[col].mode()[0] \
                            if df_ref[col].notna().any() else ""
                else:
                    fila[col] = 0

            X_usuario = pd.DataFrame([fila])
            X_prep    = pipeline.transform(X_usuario)

            # Extraemos el modelo base compatible con TreeExplainer:
            #   - Pipeline sklearn (step 'model') → named_steps
            #   - StackingClassifier              → final_estimator_ (CatBoost en Fase 5)
            #   - Cualquier otro                  → el modelo tal cual
            modelo_base = modelo
            if hasattr(modelo, 'named_steps'):
                modelo_base = modelo.named_steps.get('model', modelo)
            elif hasattr(modelo, 'final_estimator_'):
                # StackingClassifier: el meta-modelo es CatBoost → compatible con TreeExplainer
                modelo_base = modelo.final_estimator_

            explainer  = shap.TreeExplainer(modelo_base)
            shap_vals  = explainer.shap_values(X_prep)

            # shap_values puede ser array o lista según la versión de SHAP
            if isinstance(shap_vals, list):
                vals = shap_vals[1][0]  # clase 1 (abandono), primera fila
            else:
                vals = shap_vals[0] if shap_vals.ndim == 1 else shap_vals[0]

            # Nombres de features tras el pipeline
            try:
                feature_names = pipeline.get_feature_names_out()
            except AttributeError:
                feature_names = [f"feat_{i}" for i in range(len(vals))]

            # Construimos contribuciones
            contribuciones = []
            for fname, shap_val in zip(feature_names, vals):
                # Intentamos mapear al nombre original (sin prefijo del pipeline)
                fname_clean = fname.split('__')[-1] if '__' in fname else fname
                nombre      = NOMBRES_VARIABLES.get(fname_clean,
                                                     fname_clean.replace('_', ' ').title())
                contribuciones.append({
                    'variable':     nombre,
                    'valor':        fname_clean,
                    'contribucion': float(shap_val),
                })

            contribuciones.sort(key=lambda x: abs(x['contribucion']), reverse=True)
            return contribuciones[:6]

        except Exception as e:
            st.warning(f"⚠️ No se pudo calcular SHAP: {e}. Usando método rápido.")
            return _contribuciones_proxy(perfil, df_ref)


def _renderizar_waterfall(contribuciones: list[dict], prob: float,
                           df_ref: pd.DataFrame):
    """Renderiza el gráfico de cascada con las contribuciones calculadas."""

    prob_base = df_ref['abandono'].mean() \
        if 'abandono' in df_ref.columns else 0.292

    labels  = ['Base UJI'] + [c['variable'] for c in contribuciones] + ['Tu riesgo']
    valores = [prob_base] + [c['contribucion'] for c in contribuciones] + [prob]
    medidas = ['absolute'] + ['relative'] * len(contribuciones) + ['total']

    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=medidas,
        x=labels,
        y=[v * 100 for v in valores],
        text=[
            f"{v*100:+.1f}%" if 0 < i < len(labels) - 1 else f"{v*100:.1f}%"
            for i, v in enumerate(valores)
        ],
        textposition='outside',
        connector={'line': {'color': COLORES['borde'], 'width': 1}},
        increasing={'marker': {'color': COLORES['abandono']}},
        decreasing={'marker': {'color': COLORES['exito']}},
        totals={'marker': {'color': COLORES['primario']}},
        hovertemplate="<b>%{x}</b><br>Contribución: %{y:.1f} pp<extra></extra>",
    ))

    fig.update_layout(
        yaxis_title="Probabilidad de abandono (%)",
        yaxis=dict(ticksuffix='%'),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=80),
        height=380,
        showlegend=False,
        xaxis=dict(tickangle=-25),
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔴 Rojo = ese factor aumenta tu riesgo respecto a la media.  "
        "🟢 Verde = ese factor lo reduce."
    )


# =============================================================================
# GRÁFICO 4: Percentil
# =============================================================================

def _grafico_percentil(prob: float, df_ref: pd.DataFrame,
                        contexto: dict, modelo, pipeline):
    """
    Histograma del grupo de referencia con la posición del usuario marcada.
    El usuario puede cambiar el grupo con un selector de 3 opciones.
    """
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin-bottom:0.3rem;">
        📍 ¿Dónde estás respecto a otros alumnos?
    </h4>
    """, unsafe_allow_html=True)

    col_rama = 'rama_meta' if 'rama_meta' in df_ref.columns else 'rama'

    # Opciones de grupo de referencia — siempre las 3 si el contexto lo permite
    opciones_grupo = ["Toda la UJI"]
    if contexto['tipo'] in ('rama', 'titulacion') and col_rama in df_ref.columns:
        opciones_grupo.append("Solo mi rama")
    if contexto['tipo'] == 'titulacion' and 'titulacion' in df_ref.columns:
        opciones_grupo.append("Solo mi titulación")

    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        grupo_sel = st.radio(
            label="Comparar con:",
            options=opciones_grupo,
            index=0,
            key=f"grupo_percentil_{contexto['tipo']}_{contexto['valor']}",
            help="Elige el grupo con el que quieres comparar tu posición.",
        )

    # DataFrame del grupo seleccionado
    if grupo_sel == "Toda la UJI":
        df_grupo     = df_ref.copy()
        nombre_grupo = "toda la UJI"
    elif "titulación" in grupo_sel and contexto['valor']:
        df_grupo     = df_ref[df_ref['titulacion'] == contexto['valor']]
        nombre_grupo = contexto['valor']
    else:
        val_rama     = contexto['valor']
        df_grupo     = df_ref[df_ref[col_rama] == val_rama] \
            if val_rama else df_ref.copy()
        nombre_grupo = val_rama or "toda la UJI"

    # Probabilidades del grupo (usamos col precalculada si existe)
    if 'prob_abandono' in df_grupo.columns:
        probs_grupo = df_grupo['prob_abandono'].dropna().values
    else:
        try:
            cols_pipeline = list(pipeline.feature_names_in_)                 if hasattr(pipeline, 'feature_names_in_') else                 [c for c in df_grupo.columns if c not in _COLS_META]
            cols_ok = [c for c in cols_pipeline if c in df_grupo.columns]
            X_prep  = pipeline.transform(df_grupo[cols_ok])
            probs_grupo   = modelo.predict_proba(X_prep)[:, 1]
        except Exception:
            probs_grupo = np.array([0.292])

    # Percentil del usuario
    percentil = float((probs_grupo < prob).mean() * 100)

    with col_info:
        _, color, _, _ = _clasificar_riesgo(prob)
        st.markdown(f"""
        <div style="
            background:{color}15;
            border-left:4px solid {color};
            border-radius:6px;
            padding:0.8rem 1.2rem;
            font-size:0.9rem;
        ">
            Tu riesgo predicho (<strong>{prob*100:.1f}%</strong>) es mayor que el
            <strong>{percentil:.0f}%</strong> de los alumnos de <em>{nombre_grupo}</em>.<br><br>
            Dicho de otra forma: el
            <strong style="color:{color};">{100-percentil:.0f}%</strong>
            de los alumnos tiene un riesgo igual o mayor que el tuyo.
        </div>
        """, unsafe_allow_html=True)

    # Histograma
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=probs_grupo * 100,
        nbinsx=40,
        name=f"Distribución ({nombre_grupo})",
        marker_color=COLORES['primario'],
        opacity=0.6,
        hovertemplate="Rango: %{x:.0f}%<br>Alumnos: %{y}<extra></extra>",
    ))

    # Línea vertical: posición del usuario
    fig.add_vline(
        x=prob * 100,
        line_color=color, line_width=3, line_dash="solid",
        annotation_text=f"  Tú ({prob*100:.1f}%)",
        annotation_position="top right",
        annotation_font_color=color,
        annotation_font_size=13,
    )

    # Línea vertical: media del grupo
    media_grupo = float(probs_grupo.mean() * 100)
    fig.add_vline(
        x=media_grupo,
        line_color=COLORES['texto_suave'], line_width=1.5, line_dash="dash",
        annotation_text=f"  Media ({media_grupo:.1f}%)",
        annotation_position="top left",
        annotation_font_color=COLORES['texto_suave'],
        annotation_font_size=11,
    )

    fig.update_layout(
        xaxis_title="Probabilidad de abandono predicha (%)",
        yaxis_title="Nº alumnos",
        xaxis=dict(range=[0, 100], ticksuffix='%'),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
        height=280,
        showlegend=False,
        bargap=0.05,
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORES['borde'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORES['borde'])

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"📊 Distribución de probabilidades predichas para {nombre_grupo} "
        f"({len(probs_grupo):,} alumnos). La línea de color marca tu posición."
    )



# =============================================================================
# COMPARATIVA DE TITULACIONES
# =============================================================================

def _mostrar_comparativa(perfil: dict, titulaciones: list,
                          df_ref: "pd.DataFrame", modelo, pipeline,
                          df_features: "pd.DataFrame | None" = None):
    """
    Comparativa entre titulaciones seleccionadas.

    DISEÑO:
      Dato central  → tasa histórica de abandono por titulación (varía realmente)
      Dato secundario → tu riesgo personal (calculado UNA sola vez con tu perfil)

    POR QUÉ:
      El modelo no tiene 'titulacion' como feature — solo 'rama'.
      Tu riesgo personal es el mismo para titulaciones de la misma rama.
      La tasa histórica sí varía entre titulaciones y es muy útil para
      un prospecto que quiere saber qué carrera tiene más abandonos reales.
    """
    if len(titulaciones) < 2:
        st.warning("Selecciona al menos 2 titulaciones para ver la comparativa.")
        return

    st.markdown(f"""
    <h3 style="color:{COLORES['primario']}; margin-bottom:0.3rem;">
        Comparativa de titulaciones
    </h3>
    <p style="font-size:0.88rem; color:{COLORES['texto_suave']}; margin-top:0;">
        Abandono histórico real de cada titulación + tu riesgo personal estimado.
    </p>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 1. Recopilar datos históricos + calcular prob por titulación
    #    Cada titulación usa su propio df filtrado → pronósticos distintos.
    # ------------------------------------------------------------------
    col_rama = 'rama_meta' if 'rama_meta' in df_ref.columns else 'rama'
    _PALETA_COMP = ["#3182ce", "#805ad5", "#319795", "#d69e2e", "#e53e3e"]

    datos = []
    for i, tit in enumerate(titulaciones):
        df_tit = df_ref[df_ref["titulacion"] == tit] \
            if "titulacion" in df_ref.columns else df_ref.copy()
        # Fallback: si la titulación no tiene datos, usamos df_ref global
        if len(df_tit) == 0:
            df_tit = df_ref.copy()

        tasa_hist = float(df_tit["abandono"].mean() * 100) \
            if "abandono" in df_tit.columns and len(df_tit) > 0 else 0.0
        n_alumnos = len(df_tit)
        rama = df_tit[col_rama].mode()[0] \
            if col_rama in df_tit.columns and len(df_tit) > 0 else "–"

        # Calcular prob con las medias de ESTA titulación
        prob_tit, err_tit = _calcular_probabilidad(perfil, modelo, pipeline, df_tit)
        if err_tit or prob_tit is None:
            prob_tit = 0.0

        # Corrección heurística: ajusta por tasa histórica tit vs rama
        ctx_tit_tmp = {"tipo": "titulacion", "valor": tit}
        prob_tit, _, _, _ = _ajustar_prob_por_titulacion(prob_tit, ctx_tit_tmp, df_ref)
        nivel_tit, color_tit, _, _ = _clasificar_riesgo(prob_tit)

        # Nombre corto
        nc = tit
        for pref in ["Grado en ", "Doble Grado en ", "Grado Universitario en "]:
            if nc.startswith(pref):
                nc = nc[len(pref):]
                break

        datos.append({
            "titulacion":   tit,
            "nombre":       nc,
            "nombre_40":    nc[:40] + "…" if len(nc) > 40 else nc,
            "tasa_hist":    tasa_hist,
            "n_alumnos":    n_alumnos,
            "rama":         rama,
            "color":        _PALETA_COMP[i % len(_PALETA_COMP)],
            "prob":         prob_tit,
            "nivel":        nivel_tit,
            "color_riesgo": color_tit,
        })

    # Ordenar por tasa histórica de mayor a menor
    datos.sort(key=lambda x: x["tasa_hist"], reverse=True)

    st.divider()

    # ------------------------------------------------------------------
    # 4. Tabla resumen — dato central: abandono histórico
    # ------------------------------------------------------------------
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin:0.5rem 0 0.4rem 0;">
        📋 Abandono histórico por titulación
    </h4>
    <p style="font-size:0.82rem; color:{COLORES['texto_suave']}; margin:0 0 0.6rem 0;">
        Porcentaje real de estudiantes que abandonaron cada titulación
        en el período 2010–2020. Dato independiente de tu perfil.
    </p>
    """, unsafe_allow_html=True)

    # Cabecera
    cols_h = st.columns([3, 1.5, 1.2, 1.8, 1.2])
    for col, h in zip(cols_h, ["Titulación", "Abandono histórico", "Alumnos", "Rama", "Tu riesgo"]):
        col.markdown(
            f"<p style='font-size:0.77rem; font-weight:600; "
            f"color:{COLORES['texto_suave']}; margin:0;'>{h}</p>",
            unsafe_allow_html=True,
        )
    st.markdown(f"<hr style='margin:0.2rem 0; border-color:{COLORES['borde']}'>",
                unsafe_allow_html=True)

    for d in datos:
        # Color de la barra de tasa histórica según nivel
        if d["tasa_hist"] >= 40:
            col_hist = COLORES["abandono"]
        elif d["tasa_hist"] >= 25:
            col_hist = COLORES["advertencia"]
        else:
            col_hist = COLORES["exito"]

        cs = st.columns([3, 1.5, 1.2, 1.8, 1.2])
        cs[0].markdown(
            f"<p style='font-size:0.83rem; color:{COLORES['texto']}; "
            f"margin:0.15rem 0;'>"
            f"<span style='color:{d['color']};'>■</span> {d['nombre_40']}</p>",
            unsafe_allow_html=True,
        )
        cs[1].markdown(
            f"<p style='font-size:0.92rem; font-weight:bold; "
            f"color:{col_hist}; margin:0.15rem 0;'>{d['tasa_hist']:.1f}%</p>",
            unsafe_allow_html=True,
        )
        cs[2].markdown(
            f"<p style='font-size:0.83rem; color:{COLORES['texto_suave']}; "
            f"margin:0.15rem 0;'>{d['n_alumnos']:,}</p>",
            unsafe_allow_html=True,
        )
        cs[3].markdown(
            f"<p style='font-size:0.83rem; color:{COLORES['texto_suave']}; "
            f"margin:0.15rem 0;'>{d['rama']}</p>",
            unsafe_allow_html=True,
        )
        cs[4].markdown(
            f"<p style='font-size:0.83rem; font-weight:bold; "
            f"color:{d['color_riesgo']}; margin:0.15rem 0;'>{d['prob']*100:.1f}%</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<hr style='margin:0.1rem 0; border-color:{COLORES['borde']}'>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # 5. Gráfico de barras — abandono histórico por titulación
    # ------------------------------------------------------------------
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin:1.5rem 0 0.4rem 0;">
        📊 Comparativa visual
    </h4>""", unsafe_allow_html=True)

    fig = go.Figure()
    for d in datos:
        fig.add_trace(go.Bar(
            x=[d["tasa_hist"]],
            y=[d["nombre_40"]],
            orientation="h",
            marker_color=d["color"],
            text=f"{d['tasa_hist']:.1f}%",
            textposition="outside",
            name=d["nombre_40"],
            hovertemplate=(
                f"<b>{d['nombre']}</b><br>"
                f"Abandono histórico: {d['tasa_hist']:.1f}%<br>"
                f"Alumnos: {d['n_alumnos']:,}<br>"
                f"Tu riesgo personal: {d['prob']*100:.1f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Línea vertical: media UJI histórica
    media_uji_hist = float(df_ref["abandono"].mean() * 100)         if "abandono" in df_ref.columns else 29.2
    fig.add_vline(
        x=media_uji_hist,
        line_color=COLORES["texto_suave"], line_dash="dash", line_width=1.5,
        annotation_text=f"Media UJI ({media_uji_hist:.1f}%)",
        annotation_font_size=10,
        annotation_font_color=COLORES["texto_suave"],
    )
    # (línea de riesgo personal eliminada: cada titulación tiene su prob)
    # En su lugar: marcadores de riesgo como scatter sobre las barras
    fig.add_trace(go.Scatter(
        x=[d["prob"] * 100 for d in datos],
        y=[d["nombre_40"] for d in datos],
        mode="markers+text",
        marker=dict(
            symbol="diamond",
            size=12,
            color=[d["color_riesgo"] for d in datos],
            line=dict(color="white", width=1.5),
        ),
        text=[f'{d["prob"]*100:.1f}%' for d in datos],
        textposition="middle right",
        textfont=dict(size=10),
        name="Tu riesgo estimado",
        hovertemplate=(
            "<b>Tu riesgo en %{y}</b><br>"
            "Probabilidad estimada: %{x:.1f}%"
            "<extra></extra>"
        ),
        showlegend=True,
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 105], ticksuffix="%",
                   title="Tasa de abandono histórica (%)"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=120, t=20, b=40),
        height=max(220, len(datos) * 65),
        bargap=0.3,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=COLORES["borde"],
            borderwidth=1,
        ),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORES["borde"])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "📊 Barras = tasa de abandono histórica real (2010–2020). "
        "╌╌╌ Línea gris discontinua = media UJI. "
        "◆ Diamante de color = tu riesgo estimado para este perfil (por titulación)."
    )


    # ------------------------------------------------------------------
    # 6. Detalle expandible por titulación — radar + cascada
    # ------------------------------------------------------------------
    st.divider()
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin:0.5rem 0 0.3rem 0;">
        🔎 Detalle por titulación
    </h4>
    <p style="font-size:0.83rem; color:{COLORES['texto_suave']}; margin:0 0 0.8rem 0;">
        Despliega para ver el perfil de abandono histórico y los factores de riesgo.
    </p>""", unsafe_allow_html=True)

    for idx_d, d in enumerate(datos):
        col_hist_nivel = COLORES["abandono"] if d["tasa_hist"] >= 40 \
            else COLORES["advertencia"] if d["tasa_hist"] >= 25 else COLORES["exito"]
        label = (
            f"<span style='color:{d['color']};'>■</span> "
            f"{d['nombre_40']} — "
            f"Histórico: {d['tasa_hist']:.1f}% · "
            f"Tu riesgo: {d['prob']*100:.1f}% · "
            f"{d['n_alumnos']:,} alumnos"
        )
        with st.expander(f"{d['nombre_40']} — Hist: {d['tasa_hist']:.1f}% · Tu riesgo: {d['prob']*100:.1f}%", expanded=False):
            ctx_tit = {
                "tipo":        "titulacion",
                "valor":       d["titulacion"],
                "df_contexto": df_ref[df_ref["titulacion"] == d["titulacion"]]
                               if "titulacion" in df_ref.columns else df_ref.copy(),
            }
            c1, c2 = st.columns([1, 1])
            with c1:
                _grafico_radar(perfil, df_ref, ctx_tit, d['prob'])
            with c2:
                df_tit_exp = df_ref[df_ref['titulacion'] == d['titulacion']].copy() if 'titulacion' in df_ref.columns else df_ref
                _grafico_cascada(perfil, df_tit_exp, d['prob'], modelo, pipeline)

    # ------------------------------------------------------------------
    # 7. Recomendaciones personalizadas basadas en el perfil
    # ------------------------------------------------------------------
    st.divider()
    _recomendaciones(perfil, datos[0]['prob'] if datos else 0.0, 'prospecto')


# =============================================================================
# RECOMENDACIONES PERSONALIZADAS
# =============================================================================

def _recomendaciones(perfil: dict, prob: float, modo: str):
    """
    Genera recomendaciones concretas basadas en el perfil y el nivel de riesgo.
    Las recomendaciones son distintas para prospecto (antes de entrar) y
    en_curso (ya matriculado): el alumno en curso necesita orientación
    inmediata y acciones concretas, no consejos de acceso.
    """
    st.markdown(f"""
    <h4 style="color:{COLORES['texto']}; margin-bottom:0.5rem;">
        💡 Recomendaciones personalizadas
    </h4>
    """, unsafe_allow_html=True)

    nivel, color, _, _ = _clasificar_riesgo(prob)
    recomendaciones = []

    # ------------------------------------------------------------------
    # MODO EN CURSO — alumno ya matriculado
    # Recomendaciones orientadas a la acción inmediata dentro de la UJI
    # ------------------------------------------------------------------
    if modo == "en_curso":

        nota_1er = perfil.get('nota_1er_anio', 10)
        cred_1er = perfil.get('cred_superados_anio_1er', 40)
        trabaja  = perfil.get('situacion_laboral', 11) != 11
        beca     = perfil.get('n_anios_beca', 0)

        if nota_1er < 5:
            recomendaciones.append({
                'icono': '🆘',
                'titulo': 'Contacta con tu tutor ahora',
                'texto': (
                    'Una nota media del primer año por debajo de 5 es una '
                    'señal de alerta importante. El Servicio de Orientación '
                    'Universitaria (SAE) de la UJI ofrece atención personalizada '
                    'y puede ayudarte a reorganizar tu carga lectiva.'
                ),
            })
        elif nota_1er < 7:
            recomendaciones.append({
                'icono': '📋',
                'titulo': 'Revisión de tu plan de estudios',
                'texto': (
                    'Tu rendimiento tiene margen de mejora. Considera revisar '
                    'con tu tutor si la carga de asignaturas es adecuada a '
                    'tu situación personal. Reducir créditos un año puede '
                    'ser más eficaz que repetir asignaturas.'
                ),
            })

        if cred_1er < 30:
            recomendaciones.append({
                'icono': '📉',
                'titulo': 'Pocos créditos superados',
                'texto': (
                    'Superar menos de 30 créditos en el primer año es un '
                    'factor de riesgo relevante. Habla con tu facultad sobre '
                    'la posibilidad de adaptar tu matrícula o acceder a '
                    'grupos de refuerzo.'
                ),
            })

        if trabaja:
            recomendaciones.append({
                'icono': '⏰',
                'titulo': 'Compatibiliza trabajo y estudio',
                'texto': (
                    'Combinar trabajo y estudios aumenta el riesgo de abandono. '
                    'La UJI ofrece modalidades semipresenciales y horarios '
                    'adaptados. Consulta con tu facultad las opciones '
                    'disponibles para tu titulación.'
                ),
            })

        if beca == 0 and nivel in ('Medio', 'Alto'):
            recomendaciones.append({
                'icono': '🎓',
                'titulo': 'Solicita una beca',
                'texto': (
                    'Los estudiantes becados tienen tasas de abandono '
                    'significativamente más bajas. Si cumples los requisitos '
                    'económicos, solicita la beca del Ministerio o las '
                    'propias de la UJI en la próxima convocatoria.'
                ),
            })

        if nivel == 'Bajo' or not recomendaciones:
            recomendaciones.append({
                'icono': '✅',
                'titulo': 'Vas por buen camino',
                'texto': (
                    'Tu perfil actual muestra factores protectores importantes. '
                    'Mantén el ritmo de estudio y no descuides los primeros '
                    'cuatrimestres — son los más decisivos para consolidar '
                    'el hábito académico.'
                ),
            })

    # ------------------------------------------------------------------
    # MODO PROSPECTO — alumno que todavía no está matriculado
    # Recomendaciones orientadas a la decisión de entrada
    # ------------------------------------------------------------------
    else:

        trabaja = perfil.get('situacion_laboral', 11) != 11
        beca    = perfil.get('n_anios_beca', 0)
        nota    = perfil.get('nota_acceso', 10)
        edad    = perfil.get('edad_entrada', 19)

        if beca == 0:
            recomendaciones.append({
                'icono': '🎓',
                'titulo': 'Infórmate sobre becas',
                'texto': (
                    'Los años con beca son uno de los factores protectores más '
                    'importantes. Consulta las convocatorias del Ministerio y '
                    'las becas propias de la UJI antes de matricularte.'
                ),
            })

        if trabaja:
            recomendaciones.append({
                'icono': '⏰',
                'titulo': 'Planifica tu carga desde el inicio',
                'texto': (
                    'Combinar trabajo y estudios aumenta el riesgo de abandono. '
                    'La UJI ofrece modalidades semipresenciales y horarios '
                    'adaptados. Consulta con tu facultad las opciones disponibles.'
                ),
            })

        if nota < 7:
            recomendaciones.append({
                'icono': '📚',
                'titulo': 'Refuerzo académico desde el inicio',
                'texto': (
                    'Una nota de acceso más baja se puede compensar con una '
                    'buena organización desde el primer cuatrimestre. Los '
                    'servicios de tutoría de la UJI están a tu disposición '
                    'desde el primer día.'
                ),
            })

        if edad > 25:
            recomendaciones.append({
                'icono': '🤝',
                'titulo': 'Aprovecha tu experiencia',
                'texto': (
                    'Los estudiantes maduros tienen más responsabilidades pero '
                    'también más motivación y experiencia vital. La UJI tiene '
                    'programas específicos de acompañamiento para este perfil.'
                ),
            })

        if nivel == 'Bajo' or not recomendaciones:
            recomendaciones.append({
                'icono': '✅',
                'titulo': 'Buen perfil de entrada',
                'texto': (
                    'Tu perfil muestra factores protectores importantes. '
                    'Mantén el mismo nivel de dedicación durante el primer año — '
                    'es el período más crítico para consolidar el hábito de estudio.'
                ),
            })

    # ------------------------------------------------------------------
    # Renderizado de tarjetas (igual para ambos modos)
    # ------------------------------------------------------------------
    cols = st.columns(min(len(recomendaciones), 3))
    for i, rec in enumerate(recomendaciones[:3]):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background:white;
                border:1px solid {COLORES['borde']};
                border-top:3px solid {color};
                border-radius:8px;
                padding:1rem;
                min-height:180px;
            ">
                <div style="font-size:1.5rem;">{rec['icono']}</div>
                <div style="font-weight:bold; font-size:0.9rem;
                            color:{COLORES['texto']}; margin:0.3rem 0;">
                    {rec['titulo']}
                </div>
                <div style="font-size:0.78rem; color:{COLORES['texto_suave']};
                            line-height:1.4;">
                    {rec['texto']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PANTALLA DE INSTRUCCIONES (antes de calcular)
# =============================================================================

def _mostrar_instrucciones():
    """Mensaje orientativo que se muestra antes de que el usuario calcule."""
    st.markdown(f"""
    <div style="text-align:center; padding:3rem 2rem; color:{COLORES['texto_suave']};">
        <div style="font-size:3rem; margin-bottom:1rem;">🔮</div>
        <div style="font-size:1.1rem; font-weight:bold; color:{COLORES['texto']};">
            Rellena tu perfil y pulsa "Calcular mi pronóstico"
        </div>
        <div style="font-size:0.88rem; margin-top:0.5rem;">
            El modelo analizará tu perfil y te mostrará una estimación
            personalizada de tu riesgo de abandono.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# HELPER: clasificar nivel de riesgo
# =============================================================================

def _clasificar_riesgo(prob: float) -> tuple[str, str, str, str]:
    """
    Devuelve (nivel, color, emoji, mensaje) según la probabilidad.
    Usa UMBRALES y COLORES_RIESGO de config_app para coherencia global.
    """
    if prob < UMBRALES['riesgo_bajo']:
        return (
            'Bajo', COLORES_RIESGO['bajo'], '✅',
            'Tu perfil muestra factores protectores importantes. '
            'El riesgo de abandono es reducido según el modelo.',
        )
    elif prob < UMBRALES['riesgo_medio']:
        return (
            'Medio', COLORES_RIESGO['medio'], '⚠️',
            'Hay algunos factores de riesgo en tu perfil. '
            'Con el apoyo adecuado y buena planificación, es muy manejable.',
        )
    else:
        return (
            'Alto', COLORES_RIESGO['alto'], '🔴',
            'Tu perfil presenta varios factores de riesgo. '
            'Te recomendamos consultar con el servicio de orientación de la UJI.',
        )


# =============================================================================
# FIN DE pronostico_shared.py
# =============================================================================
