# =============================================================================
# config_app.py
# Fichero central de configuración de la app Streamlit
#
# ¿QUÉ HACE ESTE FICHERO?
#   Define en un único lugar todas las constantes, rutas y ajustes que
#   necesita la app. El resto de ficheros lo importan con:
#       from config_app import RUTAS, COLORES, APP_CONFIG
#   Así, si algo cambia (una ruta, un color), solo lo tocas aquí.
#
# ¿POR QUÉ ES LO PRIMERO QUE SE ESCRIBE?
#   Porque todos los demás ficheros dependen de él. Sin esta base,
#   cada fichero haría sus propias suposiciones y la app sería frágil.
# =============================================================================

from pathlib import Path

# =============================================================================
# 1. ROOT — Localizar la carpeta raíz del proyecto
# =============================================================================
# Necesitamos saber dónde está instalado el proyecto en el ordenador de
# cualquier persona que lo ejecute. No podemos hardcodear
# "C:/Users/mjmor/..." porque en otro ordenador esa ruta no existe.
#
# La estrategia: este fichero (config_app.py) está en app/.
# Subimos un nivel con .parent y llegamos a la raíz del proyecto (AU_UJI/).
#
# Path(__file__)        → ruta completa a este fichero: .../app/config_app.py
# .resolve()            → convierte a ruta absoluta (sin ../.. relativos)
# .parent               → sube un nivel: de app/ a AU_UJI/
#
# Luego verificamos que la carpeta src/ exista ahí como comprobación
# de seguridad (igual que hacemos en los notebooks del TFM).

def _detectar_root() -> Path:
    """
    Sube niveles desde este fichero hasta encontrar la carpeta src/.
    Lanza un error claro si no la encuentra, en lugar de fallar misteriosamente.
    """
    candidato = Path(__file__).resolve().parent  # empieza en app/
    for _ in range(5):                           # sube hasta 5 niveles
        if (candidato / "src").exists():
            return candidato
        candidato = candidato.parent
    raise FileNotFoundError(
        "No se encontró la carpeta src/. "
        "Asegúrate de ejecutar la app desde dentro del proyecto AU_UJI/."
    )

ROOT = _detectar_root()


# =============================================================================
# 2. RUTAS — Dónde están los datos y modelos que necesita la app
# =============================================================================
# Usamos pathlib (Path) en lugar de strings de texto porque:
#   - Funciona igual en Windows, Mac y Linux (maneja / y \ automáticamente)
#   - Podemos concatenar carpetas con el operador /  (muy legible)
#   - Tiene métodos útiles: .exists(), .stem, .suffix, etc.

RUTAS = {
    # --- Modelo y pipeline ---
    # El modelo entrenado en Fase 5 (CatBoost con balanceo)
    "modelo": ROOT / "data" / "05_modelado" / "models" / "CatBoost__balanced.pkl",

    # El pipeline de preprocesamiento (imputer + encoder + scaler)
    # Se aplica ANTES de pasar datos al modelo
    "pipeline": ROOT / "data" / "05_modelado" / "pipeline_preprocesamiento.pkl",

    # --- Resultados de Fase 6 (interpretabilidad) ---
    # Valores SHAP globales calculados sobre el conjunto de test
    "shap_global": ROOT / "results" / "fase6" / "shap_global_catboost.pkl",

    # Métricas de equidad (fairness) por subgrupos
    "fairness": ROOT / "results" / "fase6" / "fairness_metricas.parquet",

    # --- Datos de evaluación ---
    # Metadatos del test: titulacion, rama, sexo, per_id_ficticio, abandono...
    # Generado por f6_m00_preparacion.ipynb (NO contiene features del modelo)
    "meta_test": ROOT / "data" / "06_evaluacion" / "meta_test.parquet",

    # Features del test preprocesadas — las 19 variables que usa el pipeline
    # Generado en Fase 5. Se cruza con meta_test por índice en loaders.py
    "X_test_prep": ROOT / "data" / "05_modelado" / "X_test_prep.parquet",

    # Fichero puente: índice posicional + per_id_ficticio (Fase 6 celda 8b)
    # Permite join robusto en loaders.py — estable con nuevos datos Excel
    "X_test_ids": ROOT / "data" / "05_modelado" / "X_test_prep_ids.parquet",

    # --- Dataset completo (para joins con titulación) ---
    "df_alumno": ROOT / "data" / "00_raw" / "df_alumno.parquet",
}


# =============================================================================
# 3. COLORES — Paleta visual de la app
# =============================================================================
# Centralizamos los colores para que toda la app tenga coherencia visual.
# Si mañana quieres cambiar el azul por otro tono, lo cambias aquí una vez.

COLORES = {
    "primario":      "#3182ce",   # azul institucional (botones, títulos)
    "abandono":      "#e53e3e",   # rojo para riesgo de abandono
    "exito":         "#38a169",   # verde para éxito / bajo riesgo
    "advertencia":   "#d69e2e",   # amarillo para riesgo medio
    "fondo":         "#f7fafc",   # gris muy claro para fondos de tarjetas
    "texto":         "#2d3748",   # gris oscuro para texto principal
    "texto_suave":   "#718096",   # gris medio para texto secundario
    "borde":         "#e2e8f0",   # gris claro para bordes y separadores
}


# =============================================================================
# 3b. COLORES_RAMAS — Paleta fija por rama de conocimiento (Opción C)
# =============================================================================
# Cada rama tiene un color distinto y reconocible en todos los gráficos.
# Si el nombre exacto de la rama cambia en los datos, actualízalo aquí.
# Estos colores se usan en p01, p02 y p05 para barras, radar y filtros.

# Mapeo de abreviaturas a nombres completos — igual que en Fase 4 (f4_m04)
RAMAS_NOMBRES = {
    "SO": "Ciencias Sociales y Jurídicas",
    "TE": "Ingeniería y Arquitectura",
    "SA": "Ciencias de la Salud",
    "HU": "Artes y Humanidades",
    "EX": "Ciencias Experimentales",
}

# Colores por nombre completo de rama (Opción C — colores vivos)
# Paleta Dark24 (px.colors.qualitative.Dark24) — elegante y sofisticado
# Para cambiar: sustituir los hex aquí, efecto en toda la app
COLORES_RAMAS = {
    "Ingeniería y Arquitectura":     "#2E91E5",  # Dark24 azul
    "Ciencias Sociales y Jurídicas": "#1CA71C",  # Dark24 verde
    "Ciencias de la Salud":          "#E15F99",  # Dark24 rosa
    "Ciencias Experimentales":       "#FB0D0D",  # Dark24 rojo
    "Artes y Humanidades":           "#DA16FF",  # Dark24 violeta
}


# =============================================================================
# 3c. COLORES_RIESGO — Colores por nivel de riesgo (bajo / medio / alto)
# =============================================================================
# Definidos aquí una vez — usados en toda la app (donut, histograma,
# indicador de riesgo p03/p04, tabla titulaciones, p05 equidad...).
# Para cambiar un color: modifícalo aquí y se actualiza en toda la app.

COLORES_RIESGO = {
    "bajo":  "#38a169",   # verde
    "medio": "#ECC94B",   # amarillo limón
    "alto":  "#e53e3e",   # rojo
}


# =============================================================================
# 4. APP_CONFIG — Metadatos generales de la aplicación
# =============================================================================
# Información que aparece en el título del navegador, la barra lateral, etc.

APP_CONFIG = {
    "titulo":        "Predicción de Abandono — UJI",
    "subtitulo":     "TFM · Universitat Jaume I · María José Morte",
    "icono":         "🎓",         # aparece en la pestaña del navegador
    "layout":        "wide",       # "wide" = aprovecha todo el ancho de pantalla
                                   # alternativa: "centered" (columna central)
    "sidebar_state": "expanded",   # la barra lateral empieza abierta
}


# =============================================================================
# 5. PESTAÑAS — Definición de las páginas de la app
# =============================================================================
# Cada pestaña tiene un nombre, un icono y una descripción corta.
# Esta lista la usará main.py para construir la navegación lateral.
# Añadir una pestaña nueva = añadir un diccionario a esta lista.

PESTANAS = [
    {
        "id":          "institucional",
        "titulo":      "Visión institucional",
        "icono":       "🏛️",
        "descripcion": "KPIs globales y tendencias de abandono en la UJI",
        "perfil":      "Gestores y dirección académica",
    },
    {
        "id":          "titulacion",
        "titulo":      "Por titulación",
        "icono":       "📚",
        "descripcion": "Análisis detallado por grado universitario",
        "perfil":      "Profesores y coordinadores de titulación",
    },
    {
        "id":          "prospecto",
        "titulo":      "Alumno prospecto",
        "icono":       "🔍",
        "descripcion": "Pronóstico para alumnos antes de matricularse",
        "perfil":      "Futuros estudiantes y orientadores",
    },
    {
        "id":          "en_curso",
        "titulo":      "Alumno en curso",
        "icono":       "📊",
        "descripcion": "Pronóstico para alumnos ya matriculados",
        "perfil":      "Estudiantes matriculados y tutores académicos",
    },
    {
        "id":          "equidad",
        "titulo":      "Equidad y diversidad",
        "icono":       "⚖️",
        "descripcion": "Análisis de fairness por género y rama de conocimiento",
        "perfil":      "Todos los perfiles",
    },
]


# =============================================================================
# 6. UMBRALES — Criterios para clasificar el nivel de riesgo
# =============================================================================
# El modelo devuelve una probabilidad entre 0 y 1.
# Estos umbrales definen cuándo consideramos que el riesgo es bajo/medio/alto.
# Son ajustables: si el tribunal o los gestores prefieren otros valores,
# solo hay que cambiarlos aquí.

UMBRALES = {
    "riesgo_bajo":   0.30,   # prob < 0.30 → riesgo bajo (verde)
    "riesgo_medio":  0.60,   # 0.30 ≤ prob < 0.60 → riesgo medio (amarillo)
                             # prob ≥ 0.60 → riesgo alto (rojo)
}


# =============================================================================
# 7. VARIABLES — Nombres legibles de las features del modelo
# =============================================================================
# El modelo trabaja con nombres técnicos (nota_1er_anio, n_anios_beca...).
# En la interfaz mostramos nombres comprensibles para cualquier usuario.

NOMBRES_VARIABLES = {
    "nota_acceso":          "Nota de acceso a la universidad",
    "nota_1er_anio":        "Nota media del primer año",
    "n_anios_beca":         "Años con beca",
    "creditos_superados":   "Créditos superados",
    "creditos_matriculados":"Créditos matriculados",
    "tasa_rendimiento":     "Tasa de rendimiento académico",
    "situacion_laboral":    "Situación laboral",
    "rama":                 "Rama de conocimiento",
    "tipo_acceso":          "Vía de acceso a la universidad",
    "sexo":                 "Sexo",
    "edad_acceso":          "Edad al acceder a la universidad",
    "anio_cohorte":         "Año de inicio de estudios",
}


# =============================================================================
# 8. VERIFICACIÓN — Comprobar que los ficheros clave existen al arrancar
# =============================================================================
# Esta función se llama desde main.py al iniciar la app.
# Si falta algún fichero crítico, avisa claramente en lugar de fallar
# con un error críptico de Python más adelante.

def verificar_ficheros_criticos() -> list[str]:
    """
    Comprueba que existen los ficheros imprescindibles para la app.
    Devuelve una lista de mensajes de error (vacía si todo está bien).
    """
    criticos = ["modelo", "pipeline", "meta_test"]
    errores = []
    for nombre in criticos:
        ruta = RUTAS[nombre]
        if not ruta.exists():
            errores.append(f"❌ No encontrado: {ruta}")
    return errores


# =============================================================================
# FIN DE config_app.py
# Para importar en otro fichero:
#   from config_app import RUTAS, COLORES, APP_CONFIG, PESTANAS, UMBRALES
# =============================================================================
