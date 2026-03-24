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
#   2. Formulario de perfil del alumno
#   3. Cálculo de probabilidad con el modelo
#   4. Gráfico 1 — Indicador de riesgo (velocímetro)
#   5. Gráfico 2 — Radar: tu perfil vs perfil de éxito
#   6. Gráfico 3 — Cascada de contribuciones
#   7. Gráfico 4 — Percentil con selector de grupo de referencia
#   8. Recomendaciones personalizadas
#
# PARÁMETROS DE show_pronostico():
#   modo : str — "prospecto" o "en_curso"
#
# REQUISITOS:
#   - config_app.py accesible
#   - utils/loaders.py disponible
#   - Modelo y pipeline de Fase 5 cargados
#
# GENERA:
#   Página HTML interactiva. No genera ficheros en disco.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Imports internos
# ---------------------------------------------------------------------------
# _path_setup añade app/ a sys.path de forma robusta en Windows/OneDrive
import _path_setup  # noqa: F401

from config_app import COLORES, NOMBRES_VARIABLES, UMBRALES
from utils.loaders import cargar_meta_test, cargar_modelo, cargar_pipeline


# =============================================================================
# FUNCIÓN PRINCIPAL — llamada desde p03 y p04
# =============================================================================

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
            "titulo":      "🔍 Pronóstico para alumno prospecto",
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
    <h2 style="color: {COLORES['primario']}; margin-bottom: 0.2rem;">
        {textos['titulo']}
    </h2>
    <p style="color: {COLORES['texto_suave']}; margin-top: 0; font-size: 0.95rem;">
        {textos['subtitulo']}
    </p>
    <p style="color: {COLORES['texto']}; font-size: 0.88rem; max-width: 700px;">
        {textos['descripcion']}
    </p>
    """, unsafe_allow_html=True)

    # Carga de datos y modelos
    with st.spinner("Cargando modelo..."):
        try:
            df_ref   = cargar_meta_test()   # datos históricos como referencia
            modelo   = cargar_modelo()
            pipeline = cargar_pipeline()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()

    st.divider()

    # --- Paso 1: selector de contexto ---
    contexto = _selector_contexto(df_ref, modo)

    st.divider()

    # --- Paso 2: formulario de perfil ---
    perfil, calcular = _formulario_perfil(modo, df_ref, contexto)

    # Solo calculamos si el usuario pulsa el botón
    if not calcular:
        _mostrar_instrucciones()
        return

    # --- Paso 3: calcular probabilidad ---
    with st.spinner("Calculando pronóstico..."):
        prob, error = _calcular_probabilidad(perfil, modelo, pipeline, df_ref)

    if error:
        st.error(f"❌ Error al calcular: {error}")
        return

    st.divider()

    # --- Pasos 4-8: resultados ---
    _mostrar_resultado_principal(prob, modo)
    st.divider()
    _grafico_indicador_riesgo(prob)
    st.divider()

    col_radar, col_cascada = st.columns([1, 1])
    with col_radar:
        _grafico_radar(perfil, df_ref, contexto, prob)
    with col_cascada:
        _grafico_cascada(perfil, df_ref, prob)

    st.divider()
    _grafico_percentil(prob, df_ref, contexto, modelo, pipeline)
    st.divider()
    _recomendaciones(perfil, prob, modo)

    # Aviso legal / limitaciones
    st.markdown(f"""
    <div style="
        font-size: 0.75rem;
        color: {COLORES['texto_suave']};
        border-top: 1px solid {COLORES['borde']};
        margin-top: 1.5rem;
        padding-top: 0.8rem;
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
        tipo       : "titulacion" | "rama" | "todas"
        valor      : nombre de la titulación/rama, o None
        df_contexto: subconjunto de df_ref según el contexto elegido
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.5rem;">
        1️⃣ ¿Tienes una titulación en mente?
    </h4>
    <p style="font-size: 0.85rem; color: {COLORES['texto_suave']};">
        Opcional. Si eliges una titulación o rama, las comparativas
        se calcularán respecto a alumnos de ese mismo contexto.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        tipo_contexto = st.radio(
            label="Comparar contra:",
            options=["Todas las titulaciones", "Una rama concreta", "Una titulación concreta"],
            index=0,
            key=f"tipo_contexto_{modo}",
            help=(
                "Elige el grupo con el que quieres compararte. "
                "'Todas' usa el conjunto completo de la UJI como referencia."
            )
        )

    with col2:
        valor_contexto = None

        if tipo_contexto == "Una rama concreta" and 'rama' in df_ref.columns:
            ramas = sorted(df_ref['rama'].dropna().unique().tolist())
            valor_contexto = st.selectbox(
                label="Selecciona la rama",
                options=ramas,
                key=f"sel_rama_{modo}",
            )

        elif tipo_contexto == "Una titulación concreta" and 'titulacion' in df_ref.columns:
            titulaciones = sorted(df_ref['titulacion'].dropna().unique().tolist())
            valor_contexto = st.selectbox(
                label="Selecciona la titulación",
                options=titulaciones,
                key=f"sel_tit_{modo}",
                help="Puedes escribir para buscar dentro de la lista"
            )

        # Mostramos info del contexto elegido
        if valor_contexto:
            if tipo_contexto == "Una rama concreta":
                df_ctx = df_ref[df_ref['rama'] == valor_contexto]
            else:
                df_ctx = df_ref[df_ref['titulacion'] == valor_contexto]

            tasa_ctx = (df_ctx['abandono'].sum() / len(df_ctx) * 100) \
                if 'abandono' in df_ctx.columns and len(df_ctx) > 0 else 0

            st.markdown(f"""
            <div style="
                background: {COLORES['fondo']};
                border-left: 3px solid {COLORES['primario']};
                border-radius: 4px;
                padding: 0.6rem 1rem;
                font-size: 0.83rem;
                margin-top: 0.3rem;
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
                background: {COLORES['fondo']};
                border-left: 3px solid {COLORES['texto_suave']};
                border-radius: 4px;
                padding: 0.6rem 1rem;
                font-size: 0.83rem;
                margin-top: 0.3rem;
            ">
                <strong>Contexto:</strong> Toda la UJI<br>
                <strong>Alumnos en histórico:</strong> {len(df_ref):,}<br>
                <strong>Tasa abandono histórica:</strong> {tasa_uji:.1f}%
            </div>
            """, unsafe_allow_html=True)

    # Construimos el dict de contexto
    if tipo_contexto == "Una rama concreta" and valor_contexto:
        tipo_key  = "rama"
        df_ctx    = df_ref[df_ref['rama'] == valor_contexto]
    elif tipo_contexto == "Una titulación concreta" and valor_contexto:
        tipo_key  = "titulacion"
        df_ctx    = df_ref[df_ref['titulacion'] == valor_contexto]
    else:
        tipo_key  = "todas"
        df_ctx    = df_ref.copy()

    return {"tipo": tipo_key, "valor": valor_contexto, "df_contexto": df_ctx}


# =============================================================================
# PASO 2: Formulario de perfil
# =============================================================================

def _formulario_perfil(modo: str, df_ref: pd.DataFrame,
                        contexto: dict) -> tuple[dict, bool]:
    """
    Formulario con los datos del alumno.
    Devuelve (perfil_dict, calcular_bool).
    calcular_bool es True cuando el usuario pulsa el botón.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.5rem;">
        2️⃣ Rellena tu perfil
    </h4>
    """, unsafe_allow_html=True)

    # Calculamos medias del contexto para usar como valores por defecto
    # Así el formulario arranca con valores realistas, no con ceros
    df_ctx = contexto['df_contexto']
    def _media(col, default):
        return float(df_ctx[col].mean()) if col in df_ctx.columns \
            and df_ctx[col].notna().any() else default

    col1, col2, col3 = st.columns(3)

    perfil = {}

    with col1:
        st.markdown("**📋 Datos de acceso**")

        perfil['nota_acceso'] = st.slider(
            label="Nota de acceso (PAU / FP)",
            min_value=5.0,
            max_value=14.0,
            value=round(_media('nota_acceso', 8.0), 1),
            step=0.1,
            help="Nota con la que accediste o accederás a la universidad (0–14)",
            key=f"nota_acceso_{modo}"
        )

        # Opciones de tipo de acceso — valores reales del dataset
        opciones_acceso = ["Bachillerato/PAU", "FP Superior", "Mayores 25 años",
                           "Titulados", "Traslado", "Otro"]
        perfil['tipo_acceso'] = st.selectbox(
            label="Vía de acceso",
            options=opciones_acceso,
            index=0,
            key=f"tipo_acceso_{modo}"
        )

        perfil['edad_acceso'] = st.number_input(
            label="Edad al acceder",
            min_value=17,
            max_value=60,
            value=int(_media('edad_acceso', 19)),
            step=1,
            key=f"edad_{modo}"
        )

    with col2:
        st.markdown("**💼 Situación personal**")

        opciones_laboral = ["No trabaja", "Trabaja a tiempo parcial",
                            "Trabaja a tiempo completo"]
        perfil['situacion_laboral'] = st.selectbox(
            label="Situación laboral",
            options=opciones_laboral,
            index=0,
            help="La situación laboral es el predictor categórico más fuerte del modelo",
            key=f"laboral_{modo}"
        )

        perfil['n_anios_beca'] = st.slider(
            label="Años con beca previstos",
            min_value=0,
            max_value=6,
            value=int(round(_media('n_anios_beca', 2))),
            step=1,
            help="Número de años que prevés tener beca durante la carrera",
            key=f"beca_{modo}"
        )

        opciones_sexo = ["Mujer", "Hombre"]
        perfil['sexo'] = st.selectbox(
            label="Sexo",
            options=opciones_sexo,
            index=0,
            key=f"sexo_{modo}"
        )

    with col3:
        st.markdown("**📚 Rendimiento académico**")

        if modo == "en_curso":
            # Solo disponible si ya está cursando
            perfil['nota_1er_anio'] = st.slider(
                label="Nota media del primer año",
                min_value=0.0,
                max_value=10.0,
                value=round(_media('nota_1er_anio', 6.0), 1),
                step=0.1,
                help="Nota media de las asignaturas cursadas en el primer año",
                key=f"nota_1er_{modo}"
            )

            perfil['creditos_superados'] = st.slider(
                label="Créditos superados",
                min_value=0,
                max_value=80,
                value=int(_media('creditos_superados', 40)),
                step=1,
                help="Total de créditos superados hasta ahora",
                key=f"creditos_{modo}"
            )

            perfil['tasa_rendimiento'] = (
                perfil['creditos_superados'] /
                max(perfil.get('creditos_matriculados', 60), 1)
            )

        else:
            # Para prospecto: solo expectativa de rendimiento
            st.markdown(f"""
            <div style="
                background: {COLORES['fondo']};
                border-radius: 6px;
                padding: 0.8rem;
                font-size: 0.82rem;
                color: {COLORES['texto_suave']};
                margin-top: 0.5rem;
            ">
                📌 Como todavía no estás matriculado/a, no se incluyen
                datos de rendimiento académico.<br><br>
                El modelo estimará tu riesgo basándose únicamente en tu
                perfil de entrada.
            </div>
            """, unsafe_allow_html=True)

            # Valores por defecto para campos que el modelo puede necesitar
            perfil['nota_1er_anio']      = _media('nota_1er_anio', 6.0)
            perfil['creditos_superados'] = _media('creditos_superados', 40.0)
            perfil['tasa_rendimiento']   = _media('tasa_rendimiento', 0.65)

    # Botón de cálculo — centrado
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        calcular = st.button(
            label="🔮 Calcular mi pronóstico",
            type="primary",       # botón azul destacado
            use_container_width=True,
            key=f"btn_calcular_{modo}"
        )

    return perfil, calcular


# =============================================================================
# PASO 3: Calcular probabilidad
# =============================================================================

def _calcular_probabilidad(perfil: dict, modelo, pipeline,
                            df_ref: pd.DataFrame) -> tuple[float, str | None]:
    """
    Construye un DataFrame con el perfil del usuario en el formato
    que espera el pipeline, lo transforma y predice la probabilidad.

    Devuelve (probabilidad, mensaje_error).
    Si hay error, probabilidad es None y mensaje_error contiene el texto.
    """
    try:
        # Tomamos las columnas features del pipeline
        # (las que el pipeline conoce — no metadatos)
        cols_meta = ['abandono', 'titulacion', 'rama', 'anio_cohorte',
                     'sexo', 'nivel_riesgo', 'prob_abandono']
        cols_features = [c for c in df_ref.columns if c not in cols_meta]

        # Construimos una fila con los valores del perfil
        # Para columnas que el usuario no rellenó, usamos la media del contexto
        fila = {}
        for col in cols_features:
            if col in perfil:
                fila[col] = perfil[col]
            elif col in df_ref.columns:
                fila[col] = df_ref[col].mean() \
                    if df_ref[col].dtype in [np.float64, np.int64] \
                    else df_ref[col].mode()[0]
            else:
                fila[col] = 0

        X_usuario = pd.DataFrame([fila])

        # Transformamos y predecimos
        X_prep = pipeline.transform(X_usuario)
        prob   = modelo.predict_proba(X_prep)[0, 1]

        return float(prob), None

    except Exception as e:
        return None, str(e)


# =============================================================================
# RESULTADO PRINCIPAL
# =============================================================================

def _mostrar_resultado_principal(prob: float, modo: str):
    """Banner con el resultado principal antes de los gráficos."""

    pct  = prob * 100
    nivel, color, emoji, mensaje = _clasificar_riesgo(prob)

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}18, {color}08);
        border: 2px solid {color}60;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 0.5rem 0;
    ">
        <div style="font-size: 2.5rem;">{emoji}</div>
        <div style="font-size: 2rem; font-weight: bold; color: {color};">
            {pct:.1f}%
        </div>
        <div style="font-size: 1.1rem; font-weight: bold; color: {COLORES['texto']};">
            Riesgo de abandono: <span style="color: {color};">{nivel}</span>
        </div>
        <div style="font-size: 0.88rem; color: {COLORES['texto_suave']}; margin-top: 0.4rem;">
            {mensaje}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# GRÁFICO 1: Indicador de riesgo (velocímetro)
# =============================================================================

def _grafico_indicador_riesgo(prob: float):
    """
    Velocímetro semicircular que muestra la probabilidad de abandono.
    La aguja apunta al valor predicho. Verde → amarillo → rojo.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.3rem;">
        🎯 Indicador de riesgo
    </h4>
    """, unsafe_allow_html=True)

    pct = prob * 100
    _, color, _, _ = _clasificar_riesgo(prob)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={
            'suffix': '%',
            'font': {'size': 36, 'color': color}
        },
        delta={
            'reference': 29.2,        # tasa media UJI como referencia
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
                'ticktext': ['0%', '30%\nBajo', '60%\nMedio', '100%'],
            },
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                # Zonas de color de fondo del velocímetro
                {'range': [0,  UMBRALES['riesgo_bajo']  * 100],
                 'color': COLORES['exito'] + '30'},       # verde suave
                {'range': [UMBRALES['riesgo_bajo']  * 100,
                           UMBRALES['riesgo_medio'] * 100],
                 'color': COLORES['advertencia'] + '30'}, # amarillo suave
                {'range': [UMBRALES['riesgo_medio'] * 100, 100],
                 'color': COLORES['abandono'] + '30'},    # rojo suave
            ],
            'threshold': {
                'line': {'color': COLORES['texto'], 'width': 3},
                'thickness': 0.75,
                'value': pct,
            },
        },
        title={
            'text': "Probabilidad de abandono predicha<br>"
                    "<span style='font-size:0.8em;color:gray'>"
                    "Referencia: media UJI = 29.2%</span>",
            'font': {'size': 14}
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=60, b=20),
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# GRÁFICO 2: Radar — tu perfil vs perfil de éxito
# =============================================================================

def _grafico_radar(perfil: dict, df_ref: pd.DataFrame,
                   contexto: dict, prob: float):
    """
    Gráfico de araña que compara el perfil del usuario contra el perfil
    medio de los alumnos que NO abandonaron en su contexto de referencia.
    Cada eje es una variable numérica normalizada a 0-1.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.3rem;">
        🕸️ Tu perfil vs perfil de éxito
    </h4>
    """, unsafe_allow_html=True)

    df_ctx = contexto['df_contexto']

    # Variables a mostrar en el radar — solo numéricas relevantes
    vars_radar = ['nota_acceso', 'nota_1er_anio', 'n_anios_beca',
                  'creditos_superados', 'tasa_rendimiento']
    vars_disponibles = [v for v in vars_radar
                        if v in df_ctx.columns and v in perfil]

    if len(vars_disponibles) < 3:
        st.info("No hay suficientes variables para el gráfico de radar.")
        return

    # Perfil de éxito: media de los que NO abandonaron en el contexto
    exito = df_ctx[df_ctx['abandono'] == 0][vars_disponibles].mean() \
        if 'abandono' in df_ctx.columns else df_ctx[vars_disponibles].mean()

    # Valores del usuario
    usuario_vals = pd.Series({v: perfil[v] for v in vars_disponibles})

    # Normalizamos ambos a 0-1 usando los rangos del dataset completo
    for v in vars_disponibles:
        vmin = df_ref[v].min() if v in df_ref.columns else 0
        vmax = df_ref[v].max() if v in df_ref.columns else 1
        rango = vmax - vmin if vmax != vmin else 1
        exito[v]       = (exito[v]       - vmin) / rango
        usuario_vals[v] = (usuario_vals[v] - vmin) / rango

    nombres_ejes = [NOMBRES_VARIABLES.get(v, v.replace('_', ' ').title())
                    for v in vars_disponibles]
    # Cerramos el radar repitiendo el primer valor al final
    nombres_ejes_c = nombres_ejes + [nombres_ejes[0]]
    exito_c        = list(exito.values)        + [exito.values[0]]
    usuario_c      = list(usuario_vals.values) + [usuario_vals.values[0]]

    fig = go.Figure()

    # Área de perfil de éxito (fondo azul)
    fig.add_trace(go.Scatterpolar(
        r=exito_c,
        theta=nombres_ejes_c,
        fill='toself',
        fillcolor=COLORES['primario'] + '25',
        line=dict(color=COLORES['primario'], width=2),
        name='Perfil de éxito (no abandona)',
    ))

    # Área del usuario (rojo o verde según su riesgo)
    _, color_usuario, _, _ = _clasificar_riesgo(prob)
    fig.add_trace(go.Scatterpolar(
        r=usuario_c,
        theta=nombres_ejes_c,
        fill='toself',
        fillcolor=color_usuario + '30',
        line=dict(color=color_usuario, width=2.5, dash='dash'),
        name='Tu perfil',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['0%', '25%', '50%', '75%', '100%'],
                tickfont=dict(size=9),
            ),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=30, b=60),
        height=340,
    )

    st.plotly_chart(fig, use_container_width=True)
    nombre_ctx = contexto['valor'] or "toda la UJI"
    st.caption(f"💡 Perfil de éxito calculado sobre alumnos de {nombre_ctx} que completaron el grado.")


# =============================================================================
# GRÁFICO 3: Cascada de contribuciones
# =============================================================================

def _grafico_cascada(perfil: dict, df_ref: pd.DataFrame, prob: float):
    """
    Gráfico de cascada (waterfall) que muestra cómo cada variable del
    perfil sube o baja el riesgo respecto a la probabilidad base (media UJI).

    Ejemplo visual:
      Base UJI: 29%
      + No tiene beca: +8%
      + Trabaja: +5%
      + Nota acceso alta: -6%
      = Tu riesgo: 36%
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.3rem;">
        📊 ¿Qué factores influyen en tu riesgo?
    </h4>
    """, unsafe_allow_html=True)

    # Probabilidad base = media del dataset de referencia
    prob_base = df_ref['abandono'].mean() \
        if 'abandono' in df_ref.columns else 0.292

    # Calculamos contribución aproximada de cada variable
    # Método: diferencia entre la prob media del grupo con ese valor
    # y la prob base general. Es una aproximación marginal sencilla.
    vars_cascada = ['nota_acceso', 'situacion_laboral', 'n_anios_beca',
                    'nota_1er_anio', 'edad_acceso', 'tasa_rendimiento']
    vars_disponibles = [v for v in vars_cascada if v in perfil and v in df_ref.columns]

    contribuciones = []

    for v in vars_disponibles:
        val_usuario = perfil[v]

        # Para variables numéricas: comparamos el tercil del usuario
        if df_ref[v].dtype in [np.float64, np.int64, float, int]:
            tercil_33 = df_ref[v].quantile(0.33)
            tercil_66 = df_ref[v].quantile(0.66)

            if val_usuario <= tercil_33:
                grupo = df_ref[df_ref[v] <= tercil_33]
                etiqueta_val = f"bajo ({val_usuario:.1f})"
            elif val_usuario <= tercil_66:
                grupo = df_ref[(df_ref[v] > tercil_33) & (df_ref[v] <= tercil_66)]
                etiqueta_val = f"medio ({val_usuario:.1f})"
            else:
                grupo = df_ref[df_ref[v] > tercil_66]
                etiqueta_val = f"alto ({val_usuario:.1f})"
        else:
            # Para categóricas: comparamos el grupo exacto
            grupo       = df_ref[df_ref[v] == val_usuario]
            etiqueta_val = str(val_usuario)

        if len(grupo) > 0 and 'abandono' in grupo.columns:
            prob_grupo    = grupo['abandono'].mean()
            contribucion  = prob_grupo - prob_base
        else:
            contribucion  = 0.0

        nombre = NOMBRES_VARIABLES.get(v, v.replace('_', ' ').title())
        contribuciones.append({
            'variable':    nombre,
            'valor':       etiqueta_val,
            'contribucion': contribucion,
        })

    if not contribuciones:
        st.info("No hay suficientes datos para calcular las contribuciones.")
        return

    # Ordenamos por valor absoluto de contribución (las más importantes primero)
    contribuciones.sort(key=lambda x: abs(x['contribucion']), reverse=True)
    contribuciones = contribuciones[:6]  # top 6

    # Construimos el waterfall
    labels = ['Base UJI'] + [c['variable'] for c in contribuciones] + ['Tu riesgo']
    valores_waterfall = [prob_base] + \
                        [c['contribucion'] for c in contribuciones] + \
                        [prob]

    # Colores: base=gris, positivo=rojo (sube riesgo), negativo=verde (baja riesgo), total=azul
    colores_wf = ['#718096']  # base gris
    for c in contribuciones:
        colores_wf.append(COLORES['abandono'] if c['contribucion'] > 0
                          else COLORES['exito'])
    colores_wf.append(COLORES['primario'])  # total azul

    medidas = ['absolute'] + ['relative'] * len(contribuciones) + ['total']

    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=medidas,
        x=labels,
        y=[v * 100 for v in valores_waterfall],  # convertimos a porcentaje
        text=[f"{v*100:+.1f}%" if i > 0 and i < len(labels)-1
              else f"{v*100:.1f}%" for i, v in enumerate(valores_waterfall)],
        textposition='outside',
        connector={'line': {'color': COLORES['borde'], 'width': 1}},
        increasing={'marker': {'color': COLORES['abandono']}},
        decreasing={'marker': {'color': COLORES['exito']}},
        totals={'marker': {'color': COLORES['primario']}},
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Contribución: %{y:.1f} pp<extra></extra>"
        ),
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
        "💡 Rojo = ese factor aumenta tu riesgo respecto a la media. "
        "Verde = ese factor lo reduce. Los valores son aproximaciones marginales."
    )


# =============================================================================
# GRÁFICO 4: Percentil
# =============================================================================

def _grafico_percentil(prob: float, df_ref: pd.DataFrame,
                        contexto: dict, modelo, pipeline):
    """
    Muestra dónde está el usuario dentro de la distribución de probabilidades
    del grupo de referencia elegido.

    El usuario puede cambiar el grupo de referencia con un selector:
    - Toda la UJI
    - Solo su rama
    - Solo su titulación
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.3rem;">
        📍 ¿Dónde estás respecto a otros alumnos?
    </h4>
    """, unsafe_allow_html=True)

    # Selector del grupo de referencia — el usuario puede cambiarlo
    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        opciones_grupo = ["Toda la UJI"]
        if 'rama' in df_ref.columns and contexto['tipo'] in ('rama', 'titulacion'):
            opciones_grupo.append(f"Solo mi rama")
        if 'titulacion' in df_ref.columns and contexto['tipo'] == 'titulacion':
            opciones_grupo.append(f"Solo mi titulación")

        grupo_ref_sel = st.radio(
            label="Comparar con:",
            options=opciones_grupo,
            index=0,
            key=f"grupo_percentil_{contexto['tipo']}_{contexto['valor']}",
            help="Elige el grupo con el que quieres comparar tu posición"
        )

    # Seleccionamos el DataFrame del grupo de referencia
    if grupo_ref_sel == "Toda la UJI" or len(opciones_grupo) == 1:
        df_grupo = df_ref.copy()
        nombre_grupo = "toda la UJI"
    elif "titulación" in grupo_ref_sel and contexto['valor']:
        df_grupo = df_ref[df_ref['titulacion'] == contexto['valor']]
        nombre_grupo = contexto['valor']
    else:
        df_grupo = df_ref[df_ref['rama'] == contexto['valor']] \
            if contexto['valor'] else df_ref.copy()
        nombre_grupo = contexto['valor'] or "toda la UJI"

    # Calculamos probabilidades del grupo con el modelo
    # (reutilizamos las que ya tiene el df_ref si están disponibles,
    # o calculamos con pipeline+modelo si no)
    if 'prob_abandono' in df_grupo.columns:
        probs_grupo = df_grupo['prob_abandono'].dropna().values
    else:
        try:
            cols_meta     = ['abandono', 'titulacion', 'rama', 'anio_cohorte', 'sexo']
            cols_features = [c for c in df_grupo.columns if c not in cols_meta]
            X_prep        = pipeline.transform(df_grupo[cols_features])
            probs_grupo   = modelo.predict_proba(X_prep)[:, 1]
        except Exception:
            probs_grupo = np.array([0.292])  # fallback a media UJI

    # Calculamos el percentil del usuario dentro del grupo
    percentil = (probs_grupo < prob).mean() * 100

    with col_info:
        _, color, _, _ = _clasificar_riesgo(prob)
        st.markdown(f"""
        <div style="
            background: {color}15;
            border-left: 4px solid {color};
            border-radius: 6px;
            padding: 0.8rem 1.2rem;
            font-size: 0.9rem;
        ">
            Tu riesgo predicho (<strong>{prob*100:.1f}%</strong>) es
            <strong>mayor que el {percentil:.0f}%</strong> de los alumnos
            de <em>{nombre_grupo}</em>.<br><br>
            Dicho de otra forma: el
            <strong style="color:{color};">{100-percentil:.0f}%</strong>
            de los alumnos tiene un riesgo igual o mayor que el tuyo.
        </div>
        """, unsafe_allow_html=True)

    # Histograma del grupo con línea vertical del usuario
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=probs_grupo * 100,
        nbinsx=40,
        name=f'Distribución ({nombre_grupo})',
        marker_color=COLORES['primario'],
        opacity=0.6,
        hovertemplate="Rango: %{x:.0f}%<br>Alumnos: %{y}<extra></extra>",
    ))

    # Línea vertical: posición del usuario
    fig.add_vline(
        x=prob * 100,
        line_color=color,
        line_width=3,
        line_dash="solid",
        annotation_text=f"  Tú ({prob*100:.1f}%)",
        annotation_position="top right",
        annotation_font_color=color,
        annotation_font_size=13,
    )

    # Línea vertical: media del grupo
    media_grupo = probs_grupo.mean() * 100
    fig.add_vline(
        x=media_grupo,
        line_color=COLORES['texto_suave'],
        line_width=1.5,
        line_dash="dash",
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
# RECOMENDACIONES PERSONALIZADAS
# =============================================================================

def _recomendaciones(perfil: dict, prob: float, modo: str):
    """
    Genera recomendaciones concretas basadas en el perfil del usuario
    y su nivel de riesgo. Siempre positivas y orientadas a la acción.
    """
    st.markdown(f"""
    <h4 style="color: {COLORES['texto']}; margin-bottom: 0.5rem;">
        💡 Recomendaciones personalizadas
    </h4>
    """, unsafe_allow_html=True)

    nivel, color, _, _ = _clasificar_riesgo(prob)
    recomendaciones = []

    # Recomendaciones basadas en factores de riesgo detectados
    if perfil.get('n_anios_beca', 0) == 0:
        recomendaciones.append({
            'icono': '🎓',
            'titulo': 'Infórmate sobre becas',
            'texto': (
                'Los años con beca son uno de los factores protectores más '
                'importantes. Consulta las convocatorias del Ministerio y '
                'las becas propias de la UJI antes de matricularte.'
            )
        })

    if perfil.get('situacion_laboral', '') != 'No trabaja':
        recomendaciones.append({
            'icono': '⏰',
            'titulo': 'Gestión del tiempo',
            'texto': (
                'Combinar trabajo y estudios aumenta el riesgo de abandono. '
                'La UJI ofrece modalidades semipresenciales y horarios '
                'adaptados. Consulta con tu facultad las opciones disponibles.'
            )
        })

    if perfil.get('nota_acceso', 10) < 7:
        recomendaciones.append({
            'icono': '📚',
            'titulo': 'Refuerzo académico desde el inicio',
            'texto': (
                'Una nota de acceso más baja se puede compensar con una '
                'buena organización desde el primer cuatrimestre. Los '
                'servicios de tutoría de la UJI están a tu disposición.'
            )
        })

    if modo == 'en_curso' and perfil.get('nota_1er_anio', 10) < 5:
        recomendaciones.append({
            'icono': '🆘',
            'titulo': 'Busca apoyo académico ahora',
            'texto': (
                'Una nota media del primer año por debajo de 5 es una señal '
                'de alerta importante. Contacta con tu tutor académico o '
                'con el servicio de orientación universitaria de la UJI '
                'lo antes posible.'
            )
        })

    if perfil.get('edad_acceso', 19) > 25:
        recomendaciones.append({
            'icono': '🤝',
            'titulo': 'Aprovecha tu experiencia',
            'texto': (
                'Los estudiantes maduros tienen más responsabilidades pero '
                'también más motivación y experiencia vital. La UJI tiene '
                'programas específicos de acompañamiento para este perfil.'
            )
        })

    # Si tiene bajo riesgo, también damos un mensaje positivo
    if nivel == 'Bajo' or not recomendaciones:
        recomendaciones.append({
            'icono': '✅',
            'titulo': 'Buen perfil de entrada',
            'texto': (
                'Tu perfil muestra factores protectores importantes. '
                'Mantén el mismo nivel de dedicación durante el primer año — '
                'es el período más crítico para consolidar el hábito de estudio.'
            )
        })

    # Mostramos las recomendaciones en tarjetas
    cols = st.columns(min(len(recomendaciones), 3))
    for i, rec in enumerate(recomendaciones[:3]):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background: white;
                border: 1px solid {COLORES['borde']};
                border-top: 3px solid {color};
                border-radius: 8px;
                padding: 1rem;
                height: 180px;
                overflow: hidden;
            ">
                <div style="font-size: 1.5rem;">{rec['icono']}</div>
                <div style="font-weight: bold; font-size: 0.9rem;
                            color: {COLORES['texto']}; margin: 0.3rem 0;">
                    {rec['titulo']}
                </div>
                <div style="font-size: 0.78rem; color: {COLORES['texto_suave']};
                            line-height: 1.4;">
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
    <div style="
        text-align: center;
        padding: 3rem 2rem;
        color: {COLORES['texto_suave']};
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🔮</div>
        <div style="font-size: 1.1rem; font-weight: bold; color: {COLORES['texto']};">
            Rellena tu perfil y pulsa "Calcular mi pronóstico"
        </div>
        <div style="font-size: 0.88rem; margin-top: 0.5rem;">
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
    Centralizado aquí para coherencia en toda la página.
    """
    if prob < UMBRALES['riesgo_bajo']:
        return (
            'Bajo',
            COLORES['exito'],
            '✅',
            'Tu perfil muestra factores protectores importantes. '
            'El riesgo de abandono es reducido según el modelo.'
        )
    elif prob < UMBRALES['riesgo_medio']:
        return (
            'Medio',
            COLORES['advertencia'],
            '⚠️',
            'Hay algunos factores de riesgo en tu perfil. '
            'Con el apoyo adecuado y buena planificación, es muy manejable.'
        )
    else:
        return (
            'Alto',
            COLORES['abandono'],
            '🔴',
            'Tu perfil presenta varios factores de riesgo. '
            'Te recomendamos consultar con el servicio de orientación de la UJI.'
        )


# =============================================================================
# FIN DE pronostico_shared.py
# =============================================================================
