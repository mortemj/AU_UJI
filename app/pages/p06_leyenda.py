# =============================================================================
# p06_leyenda.py — Guía semántica, transparencia del modelo y marco ético
# =============================================================================
# Audiencia: tribunal, gestores, cualquier usuario que quiera entender
#            qué significa cada color, símbolo y métrica de la app.
#
# Bloques:
#   A. Paleta de colores y significado semántico
#   B. Transparencia del modelo (métricas dinámicas)
#   C. Marco ético y legal
#   D. Glosario de términos y símbolos
#   E. Sobre el modelo — decisiones metodológicas (placeholder para fases)
# =============================================================================

import sys
import json
from pathlib import Path

import streamlit as st

# --- Path setup ---
_DIR = Path(__file__).resolve().parent.parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from config_app import (
    APP_CONFIG, COLORES, COLORES_RAMAS, COLORES_RIESGO,
    COLORES_SEXO, RAMAS_NOMBRES, RUTAS, UMBRALES,
)


# =============================================================================
# HELPERS
# =============================================================================

def _leer_metricas() -> dict:
    """Lee metricas_modelo.json. Fallback con valores canónicos."""
    try:
        ruta = RUTAS.get("metricas_modelo")
        if ruta and Path(ruta).exists():
            return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "auc": 0.954, "f1": 0.827,
        "baseline_auc": 0.927, "baseline_f1": 0.797,
        "n_alumnos_unicos": 30872, "n_registros": 33621,
        "n_test": 6596, "tasa_abandono": 0.292,
        "modelo_nombre": "Stacking (CatBoost + RF + LogReg)",
        "baseline_nombre": "CatBoost AutoML",
    }


def _chip_color(hex_color: str, label: str, desc: str = "") -> str:
    """Genera HTML de un chip de color con etiqueta y descripción."""
    return f"""
    <div style="display:flex; align-items:center; gap:0.7rem;
                padding:0.5rem 0.8rem; margin:0.3rem 0;
                border-radius:8px; background:{COLORES['fondo']};
                border:1px solid {COLORES['borde']};">
        <div style="width:28px; height:28px; border-radius:6px;
                    background:{hex_color}; flex-shrink:0;
                    box-shadow:0 1px 3px rgba(0,0,0,0.2);"></div>
        <div>
            <span style="font-weight:600; color:{COLORES['texto']};
                         font-size:0.9rem;">{label}</span>
            {"<br><span style='font-size:0.78rem; color:" + COLORES['texto_suave'] + ";'>" + desc + "</span>" if desc else ""}
        </div>
        <code style="margin-left:auto; font-size:0.75rem;
                     color:{COLORES['texto_suave']};">{hex_color}</code>
    </div>"""


def _seccion(icono: str, titulo: str, subtitulo: str = ""):
    """Cabecera de sección con separador."""
    st.markdown(f"""
    <div style="margin:1.5rem 0 0.5rem 0;">
        <h2 style="color:{COLORES['primario']}; margin:0; font-size:1.3rem;">
            {icono} {titulo}
        </h2>
        {"<p style='color:" + COLORES['texto_suave'] + "; font-size:0.85rem; margin:0.2rem 0 0 0;'>" + subtitulo + "</p>" if subtitulo else ""}
    </div>
    <hr style="margin:0.4rem 0 1rem 0; border:none; border-top:2px solid {COLORES['borde']};"/>
    """, unsafe_allow_html=True)


# =============================================================================
# BLOQUES
# =============================================================================

def _bloque_A_colores():
    """Bloque A — Paleta de colores y significado semántico."""
    _seccion("🎨", "Paleta de colores y significado semántico",
             "Cada color en esta app tiene un significado concreto y documentado.")

    # --- Riesgo ---
    st.markdown(f"#### 🚦 Niveles de riesgo de abandono")
    st.markdown(
        f"Los umbrales son: **bajo** < {UMBRALES['riesgo_bajo']*100:.0f}% · "
        f"**medio** {UMBRALES['riesgo_bajo']*100:.0f}–{UMBRALES['riesgo_medio']*100:.0f}% · "
        f"**alto** ≥ {UMBRALES['riesgo_medio']*100:.0f}%. "
        "Se eligieron para equilibrar sensibilidad y especificidad con una tasa de abandono del 29,2%.",
        unsafe_allow_html=False
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(_chip_color(
            COLORES_RIESGO["bajo"], "🟢 Riesgo bajo",
            f"Probabilidad < {UMBRALES['riesgo_bajo']*100:.0f}%"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(_chip_color(
            COLORES_RIESGO["medio"], "🟡 Riesgo medio",
            f"{UMBRALES['riesgo_bajo']*100:.0f}–{UMBRALES['riesgo_medio']*100:.0f}%"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(_chip_color(
            COLORES_RIESGO["alto"], "🔴 Riesgo alto",
            f"Probabilidad ≥ {UMBRALES['riesgo_medio']*100:.0f}%"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Ramas ---
    st.markdown("#### 🎓 Ramas de conocimiento — paleta inspirada en togas doctorales")
    st.markdown(
        "Los colores de cada rama están inspirados en los colores tradicionales de las "
        "togas doctorales españolas, adaptados para legibilidad en pantalla. "
        "**Ningún color de rama coincide con los de riesgo** para evitar confusión semántica.",
        unsafe_allow_html=False
    )

    # Mapa toga → descripción
    _togas = {
        "Ciencias Sociales y Jurídicas": ("toga amarillo/oro",  "ámbar dorado"),
        "Ingeniería y Arquitectura":     ("toga marrón",        "naranja tostado"),
        "Ciencias de la Salud":          ("toga amarillo limón","azul cyan"),
        "Artes y Humanidades":           ("toga morado",        "violeta medio"),
        "Ciencias Experimentales":       ("toga azul marino",   "azul profundo"),
    }
    cols = st.columns(2)
    for i, (nombre, color) in enumerate(COLORES_RAMAS.items()):
        toga_orig, color_nombre = _togas.get(nombre, ("", ""))
        with cols[i % 2]:
            abr = next((k for k, v in RAMAS_NOMBRES.items() if v == nombre), "")
            st.markdown(_chip_color(
                color,
                f"{abr} · {nombre}",
                f"Inspiración: {toga_orig} → {color_nombre}"
            ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Sexo ---
    st.markdown("#### 👥 Sexo — convención ODS 5")
    st.markdown(
        "Colores independientes de las ramas para evitar acoplamiento involuntario. "
        "Alineados con la convención visual internacional y el **ODS 5** (igualdad de género).",
        unsafe_allow_html=False
    )
    cols2 = st.columns(3)
    for i, (label, color) in enumerate(COLORES_SEXO.items()):
        descs = {
            "Mujer":  "Variable protegida ODS 5",
            "Hombre": "Convención visual estándar",
            "Total":  "Línea agregada neutra",
        }
        with cols2[i]:
            st.markdown(_chip_color(color, label, descs.get(label, "")),
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Institucional ---
    st.markdown("#### 🏛️ Colores institucionales")
    cols3 = st.columns(2)
    with cols3[0]:
        st.markdown(_chip_color(
            COLORES["primario"], "Azul institucional UJI",
            "Cabeceras, botones principales, énfasis"
        ), unsafe_allow_html=True)
    with cols3[1]:
        st.markdown(_chip_color(
            COLORES["primario_claro"], "Azul claro accent",
            "Hover, accents secundarios"
        ), unsafe_allow_html=True)


def _bloque_B_modelo():
    """Bloque B — Transparencia del modelo."""
    _seccion("📊", "Transparencia del modelo",
             "Métricas reales cargadas dinámicamente desde metricas_modelo.json.")

    m = _leer_metricas()

    # KPIs del modelo
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AUC-ROC", f"{m.get('auc', 0.954):.3f}",
                  help="Área bajo la curva ROC. 1.0 = perfecto, 0.5 = azar.")
    with col2:
        st.metric("F1-Score test", f"{m.get('f1', 0.827):.3f}",
                  help="Media armónica de precisión y recall sobre el conjunto de test.")
    with col3:
        n_test = m.get("n_test", 6596)
        st.metric("Alumnos test", f"{n_test:,}".replace(",", "."),
                  help="Alumnos en el conjunto de test tras filtro 2010–2020.")
    with col4:
        tasa = m.get("tasa_abandono", 0.292) * 100
        st.metric("Tasa abandono", f"{tasa:.1f} %".replace(".", ","),
                  help="Tasa real de abandono en el conjunto de test.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparativa vs baseline
    with st.expander("📈 Comparativa con el baseline AutoML", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div style="padding:1rem; border-radius:8px;
                        border-left:4px solid {COLORES['primario']};
                        background:{COLORES['fondo']};">
                <div style="font-size:0.8rem; color:{COLORES['texto_suave']};
                            text-transform:uppercase; letter-spacing:0.05em;">
                    Modelo final</div>
                <div style="font-size:1.1rem; font-weight:700;
                            color:{COLORES['primario']};">
                    {m.get('modelo_nombre', 'Stacking')}</div>
                <div style="font-size:1.4rem; font-weight:800;
                            color:{COLORES['primario']};">
                    AUC {m.get('auc', 0.954):.3f} · F1 {m.get('f1', 0.827):.3f}</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style="padding:1rem; border-radius:8px;
                        border-left:4px solid {COLORES['texto_suave']};
                        background:{COLORES['fondo']};">
                <div style="font-size:0.8rem; color:{COLORES['texto_suave']};
                            text-transform:uppercase; letter-spacing:0.05em;">
                    Baseline AutoML</div>
                <div style="font-size:1.1rem; font-weight:700;
                            color:{COLORES['texto_suave']};">
                    {m.get('baseline_nombre', 'CatBoost AutoML')}</div>
                <div style="font-size:1.4rem; font-weight:800;
                            color:{COLORES['texto_suave']};">
                    AUC {m.get('baseline_auc', 0.927):.3f} · F1 {m.get('baseline_f1', 0.797):.3f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Qué significan las métricas
    with st.expander("❓ ¿Qué significan AUC y F1?", expanded=False):
        st.markdown(f"""
**AUC-ROC (Área bajo la curva ROC)**
Mide la capacidad del modelo para distinguir entre alumnos que van a abandonar y los que no.
Un AUC de **{m.get('auc', 0.954):.3f}** significa que, dado un alumno que abandona y otro que no,
el modelo asigna mayor probabilidad de abandono al primero en el {m.get('auc', 0.954)*100:.1f}% de los casos.
Rango: 0.5 (azar) → 1.0 (perfecto).

**F1-Score**
Media armónica entre precisión (¿de los que predigo que abandonan, cuántos realmente abandonan?)
y recall (¿de los que realmente abandonan, cuántos detecto?).
Con una tasa de abandono del {tasa:.1f}%, el F1 es más informativo que la exactitud (accuracy).
Un F1 de **{m.get('f1', 0.827):.3f}** indica un equilibrio sólido entre los dos errores posibles.
        """)

    # Limitaciones
    with st.expander("⚠️ Limitaciones del modelo", expanded=False):
        st.markdown(f"""
1. **Datos hasta 2020.** El modelo refleja patrones de 2010–2020. Cambios estructurales posteriores
   (pandemia, nuevos grados) pueden no estar representados.

2. **Censura temporal cohorte 2020.** Los alumnos que iniciaron en 2020 tienen menos de 4 años
   de observación — su tasa de abandono observada subestima el valor real.
   La solución metodológica correcta es el análisis de supervivencia (Kaplan-Meier),
   propuesto como línea de ampliación del TFM.

3. **Resultados orientativos.** Las predicciones son probabilidades estadísticas,
   no deterministas. No deben usarse como único criterio de decisión sobre ningún estudiante.

4. **Grupos pequeños (N < 30).** Las estimaciones para grupos con pocos alumnos
   tienen mayor varianza estadística y deben interpretarse con cautela.
        """)



def _bloque_B2_titulaciones():
    """Bloque B2 — Catálogo de titulaciones por rama."""
    _seccion("🎓", "Catálogo de titulaciones por rama de conocimiento",
             "Titulaciones disponibles en la app · Datos UJI 2010–2020 · Nombres según SIA 2025")

    # Catálogo: nombre corto → (abreviatura rama, color fondo, color texto, color borde)
    _catalogo = {
        "Ciencias Sociales y Jurídicas": {
            "abr": "SO", "color": "#d97706", "bg": "rgba(217,119,6,0.05)", "txt": "#92400e",
            "tits": [
                "Administración y Dirección de Empresas",
                "Doble ADE + Derecho",
                "Ciencias Actividad Física y Deporte",
                "Criminología y Seguridad",
                "Comunicación Audiovisual",
                "Derecho", "Economía", "Finanzas y Contabilidad",
                "Gestión y Administración Pública",
                "International Business Economics",
                "Maestro Educ. Infantil", "Maestro Educ. Primaria",
                "Doble Maestro Infantil+Primaria",
                "Marketing", "Periodismo",
                "Publicidad y Comunicación Corporativa",
                "Relaciones Laborales y RRHH", "Turismo",
            ],
        },
        "Artes y Humanidades": {
            "abr": "HU", "color": "#7c3aed", "bg": "rgba(124,58,237,0.05)", "txt": "#4c1d95",
            "tits": [
                "Estudios Ingleses", "Historia y Patrimonio",
                "Humanidades", "Traducción e Interpretación",
            ],
        },
        "Ingeniería y Arquitectura": {
            "abr": "TE", "color": "#c2410c", "bg": "rgba(194,65,12,0.05)", "txt": "#7c2d12",
            "tits": [
                "Arquitectura Técnica",
                "Diseño y Desarrollo de Videojuegos",
                "Ing. Diseño Industrial y Desarrollo de Productos",
                "Ing. Eléctrica", "Ing. Informática", "Ing. Mecánica",
                "Ing. Química", "Ing. Tecnologías Industriales",
                "Inteligencia Robótica", "Matemática Computacional",
            ],
            "historicas": ["Ing. Agroalimentaria y del Medio Rural"],
        },
        "Ciencias Experimentales": {
            "abr": "EX", "color": "#1d4ed8", "bg": "rgba(29,78,216,0.05)", "txt": "#1e3a8a",
            "tits": ["Química", "Bioquímica y Biología Molecular"],
        },
        "Ciencias de la Salud": {
            "abr": "SA", "color": "#0891b2", "bg": "rgba(8,145,178,0.05)", "txt": "#164e63",
            "tits": ["Enfermería", "Medicina", "Psicología"],
        },
    }

    for nombre_rama, datos in _catalogo.items():
        color   = datos["color"]
        bg      = datos["bg"]
        txt     = datos["txt"]
        abr     = datos["abr"]
        tits    = datos["tits"]
        hist    = datos.get("historicas", [])
        n_total = len(tits) + len(hist)

        # Cabecera de rama
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:6px 10px;
                    border-radius:8px;background:rgba(0,0,0,0.03);
                    border-left:4px solid {color};margin-bottom:5px;">
            <div style="width:14px;height:14px;border-radius:3px;
                        background:{color};flex-shrink:0;"></div>
            <span style="font-size:0.85rem;font-weight:500;color:{txt};">{nombre_rama}</span>
            <span style="font-size:0.75rem;color:{COLORES['texto_suave']};
                         margin-left:auto;font-family:monospace;">
                {abr} · {n_total} titulaciones
            </span>
        </div>""", unsafe_allow_html=True)

        # Chips de titulaciones
        chips_html = " ".join([
            f'''<span style="font-size:0.75rem;padding:2px 8px;border-radius:10px;
                        border:1px solid {color};color:{txt};background:{bg};
                        display:inline-block;margin:2px;">{t}</span>'''
            for t in tits
        ])
        # Chips históricas (gris + aviso)
        chips_hist = " ".join([
            f'''<span style="font-size:0.75rem;padding:2px 8px;border-radius:10px;
                        border:1px solid {COLORES['borde']};
                        color:{COLORES['texto_suave']};
                        background:{COLORES['fondo']};
                        display:inline-block;margin:2px;">
                {t} ⚠️ histórica</span>'''
            for t in hist
        ])

        st.markdown(
            f'''<div style="padding:0 8px 8px 8px;line-height:2;">
                {chips_html}{chips_hist}
            </div>''',
            unsafe_allow_html=True
        )
        if hist:
            st.caption(
                f"⚠️ {', '.join(hist)}: sin docencia activa — "
                "aparece en análisis histórico pero no se ofrece para pronóstico."
            )

    st.markdown(f"""
    <div style="font-size:0.75rem;color:{COLORES['texto_suave']};padding:4px 8px;
                border-radius:6px;background:{COLORES['fondo']};margin-top:4px;">
        Fuente: <code>meta_test_app.parquet</code> ·
        Clasificación oficial UJI/ANECA ·
        Nombres actualizados según SIA 2025
    </div>""", unsafe_allow_html=True)


def _bloque_C_etica():
    """Bloque C — Marco ético y legal."""
    _seccion("⚖️", "Marco ético y legal",
             "Esta aplicación opera bajo un marco normativo explícito.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="padding:1rem; border-radius:8px;
                    border-left:4px solid {COLORES['primario']};
                    background:{COLORES['fondo']}; margin-bottom:0.8rem;">
            <div style="font-weight:700; color:{COLORES['primario']};">
                🔒 RGPD — Reglamento UE 2016/679</div>
            <div style="font-size:0.85rem; color:{COLORES['texto']}; margin-top:0.4rem;">
                Todos los datos son anonimizados. Ningún alumno es identificable.
                Los registros se identifican únicamente por un código interno ficticio.
            </div>
        </div>
        <div style="padding:1rem; border-radius:8px;
                    border-left:4px solid {COLORES['advertencia']};
                    background:{COLORES['fondo']}; margin-bottom:0.8rem;">
            <div style="font-weight:700; color:{COLORES['texto']};">
                🤖 AI Act — Reglamento UE 2024/1689</div>
            <div style="font-size:0.85rem; color:{COLORES['texto']}; margin-top:0.4rem;">
                Los sistemas de IA aplicados a educación se clasifican como
                <strong>alto riesgo</strong>. Esta app requiere supervisión humana
                explícita y no puede sustituir el juicio académico.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="padding:1rem; border-radius:8px;
                    border-left:4px solid {COLORES_SEXO['Mujer']};
                    background:{COLORES['fondo']}; margin-bottom:0.8rem;">
            <div style="font-weight:700; color:{COLORES_SEXO['Mujer']};">
                👥 ODS 5 — Igualdad de género</div>
            <div style="font-size:0.85rem; color:{COLORES['texto']}; margin-top:0.4rem;">
                El sexo es una variable protegida analizada explícitamente
                en la página de equidad. El modelo muestra rendimiento
                homogéneo entre grupos de sexo (diferencia F1: 1,8 pp).
            </div>
        </div>
        <div style="padding:1rem; border-radius:8px;
                    border-left:4px solid {COLORES['exito']};
                    background:{COLORES['fondo']}; margin-bottom:0.8rem;">
            <div style="font-weight:700; color:{COLORES['exito']};">
                📋 UNESCO — Ética de la IA (2021)</div>
            <div style="font-size:0.85rem; color:{COLORES['texto']}; margin-top:0.4rem;">
                Recomendación sobre la ética de la inteligencia artificial.
                Principios de transparencia, responsabilidad y no discriminación
                aplicados al diseño de esta herramienta.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "⚠️ **Uso responsable:** esta herramienta es de apoyo a la decisión, "
        "nunca sustituto del juicio humano. Cualquier uso institucional debe ir "
        "acompañado de supervisión humana y revisión periódica.",
        icon=None
    )


def _bloque_D_glosario():
    """Bloque D — Glosario de términos y símbolos."""
    _seccion("🔣", "Glosario de términos y símbolos",
             "Definiciones precisas de los conceptos usados en la app.")

    terminos = [
        ("Abandono (definición estricta)",
         "Baja definitiva del grado sin traslado ni cambio de titulación. "
         "No incluye traslados a otro grado ni interrupciones temporales."),
        ("Riesgo predicho (%)",
         "Probabilidad que el modelo asigna a un alumno de abandonar el grado, "
         "basada en su perfil. Es una estimación estadística, no una certeza."),
        ("Abandono real (%)",
         "Porcentaje de alumnos que realmente abandonaron en el conjunto de test "
         "(datos históricos 2010–2020). No es una predicción."),
        ("Disparate Impact (DI)",
         "Ratio entre la tasa de predicción positiva del grupo menos favorecido "
         "y la del más favorecido. Valor ideal: cercano a 1,0. "
         "Por debajo de 0,8 → señal de discriminación estadística (regla del 80%, EEOC)."),
        ("Falso positivo (FP)",
         "El modelo predice abandono pero el alumno completa el grado. "
         "Consecuencia: intervención innecesaria."),
        ("Falso negativo (FN)",
         "El alumno abandona pero el modelo no lo detecta. "
         "Consecuencia: alumno en riesgo sin apoyo."),
        ("AUC-ROC",
         "Área bajo la curva ROC. Mide la capacidad discriminativa del modelo "
         "independientemente del umbral de clasificación. Rango: 0,5 (azar) → 1,0 (perfecto)."),
        ("F1-Score",
         "Media armónica entre precisión y recall. Más informativo que la exactitud "
         "cuando las clases están desbalanceadas (29,2% abandono vs 70,8% no abandono)."),
        ("Stacking",
         "Ensamble de modelos que combina las predicciones de varios modelos base "
         "(CatBoost + Random Forest) usando un meta-learner (Regresión Logística) "
         "para obtener la predicción final."),
        ("SHAP",
         "SHapley Additive exPlanations. Método para explicar cuánto contribuye "
         "cada variable a la predicción de un alumno concreto. "
         "Basado en la teoría de juegos cooperativos (Shapley, 1953)."),
        ("Kaplan-Meier",
         "Estimador de supervivencia que tiene en cuenta la censura temporal. "
         "Permite estimar el tiempo hasta el abandono incluso para cohortes "
         "que no han completado el período de observación. "
         "Propuesto como línea de ampliación del TFM."),
        ("D_strict",
         "Dataset de producción del TFM: 33.621 registros × 19 features. "
         "Construido con auditoría de leakage estricta — excluye variables "
         "que podrían filtrar información del futuro al modelo."),
    ]

    # Mostrar en 2 columnas
    mid = len(terminos) // 2 + len(terminos) % 2
    col1, col2 = st.columns(2)

    for i, (term, defn) in enumerate(terminos):
        target = col1 if i < mid else col2
        with target:
            st.markdown(f"""
            <div style="padding:0.6rem 0.8rem; margin:0.3rem 0;
                        border-radius:6px; background:{COLORES['fondo']};
                        border-left:3px solid {COLORES['primario']};">
                <div style="font-weight:700; font-size:0.88rem;
                            color:{COLORES['primario']};">{term}</div>
                <div style="font-size:0.82rem; color:{COLORES['texto']};
                            margin-top:0.2rem;">{defn}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Iconos usados en la app
    st.markdown("#### 🗂️ Iconos y su significado")
    iconos = [
        ("🏛️", "Visión institucional — KPIs globales y tendencias"),
        ("📚", "Por titulación — análisis detallado por grado"),
        ("🔍", "Futuro estudiante — pronóstico antes de matricularse"),
        ("📊", "Alumno en curso — pronóstico con datos actuales"),
        ("⚖️", "Equidad y diversidad — análisis de fairness"),
        ("🟢", "Riesgo bajo — probabilidad de abandono < 30%"),
        ("🟡", "Riesgo medio — probabilidad 30–60%"),
        ("🔴", "Riesgo alto — probabilidad ≥ 60%"),
        ("📋", "Nota metodológica — información técnica del bloque"),
        ("⚠️", "Aviso — interpretar con cautela"),
        ("💡", "Recomendación personalizada"),
        ("🎓", "Titulación o dato académico"),
    ]
    cols_ic = st.columns(3)
    for i, (icono, desc) in enumerate(iconos):
        with cols_ic[i % 3]:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.5rem;
                        padding:0.4rem 0.6rem; margin:0.2rem 0;
                        border-radius:6px; background:{COLORES['fondo']};
                        border:1px solid {COLORES['borde']};">
                <span style="font-size:1.2rem;">{icono}</span>
                <span style="font-size:0.8rem; color:{COLORES['texto']};">{desc}</span>
            </div>""", unsafe_allow_html=True)


def _bloque_placeholder_E():
    """Bloque E — Placeholder para decisiones metodológicas (se completa con las fases)."""
    _seccion("🧠", "Sobre el modelo — decisiones metodológicas",
             "Este bloque se completará al cerrar las fases 1–6 del proyecto.")

    st.markdown(f"""
    <div style="padding:1.2rem; border-radius:8px;
                border:2px dashed {COLORES['borde']};
                background:{COLORES['fondo']}; text-align:center;">
        <div style="font-size:1.5rem;">🔧</div>
        <div style="font-weight:600; color:{COLORES['texto_suave']}; margin:0.3rem 0;">
            Pendiente de completar</div>
        <div style="font-size:0.85rem; color:{COLORES['texto_suave']};">
            Aquí se documentarán: pipeline completo · por qué Stacking · por qué 19 features ·
            decisión D_strict · por qué no incluir titulación · Kaplan-Meier como extensión
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def show():
    """Renderiza la página de guía semántica y transparencia."""

    # Cabecera
    st.markdown(f"""
    <div style="margin-bottom:0.5rem;">
        <h1 style="color:{COLORES['primario']}; margin:0; font-size:1.8rem;">
            📖 Guía semántica y transparencia
        </h1>
        <p style="color:{COLORES['texto_suave']}; margin:0.3rem 0 0 0; font-size:0.9rem;">
            Qué significa cada color, símbolo y métrica de esta aplicación ·
            Marco ético y legal · Glosario de términos
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Índice rápido
    with st.expander("📑 Índice de esta página", expanded=False):
        st.markdown("""
- 🎨 **Paleta de colores** — riesgo, ramas (togas), sexo, institucional
- 📊 **Transparencia del modelo** — métricas, comparativa baseline, limitaciones
- 🎓 **Catálogo de titulaciones** — todas las titulaciones por rama de conocimiento
- ⚖️ **Marco ético y legal** — RGPD, AI Act, ODS 5, UNESCO
- 🔣 **Glosario** — términos técnicos e iconos
- 🧠 **Decisiones metodológicas** — pendiente de completar con las fases
        """)

    # Bloques
    _bloque_A_colores()
    _bloque_B_modelo()
    _bloque_B2_titulaciones()
    _bloque_C_etica()
    _bloque_D_glosario()
    _bloque_placeholder_E()

    # Pie de página
    st.divider()
    st.markdown(f"""
    <div style="text-align:center; padding:0.5rem;
                color:{COLORES['texto_suave']}; font-size:0.8rem;">
        {APP_CONFIG['autora']} · {APP_CONFIG['tipo_trabajo']} ·
        {APP_CONFIG['universidad_master']} + {APP_CONFIG['universidad_datos']} ·
        {APP_CONFIG['año']} ·
        <a href="mailto:{APP_CONFIG['email_master']}"
           style="color:{COLORES['primario']};">{APP_CONFIG['email_master']}</a>
    </div>
    """, unsafe_allow_html=True)
