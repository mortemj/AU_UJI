# =============================================================================
# loaders.py
# Módulo de carga de datos y modelos para la app Streamlit
#
# ¿QUÉ HACE ESTE FICHERO?
#   Contiene funciones que cargan desde disco los ficheros que necesita
#   la app: el modelo entrenado, el pipeline de preprocesamiento, los
#   valores SHAP, las métricas de fairness y los datos de test.
#
# ¿POR QUÉ ESTÁ SEPARADO DE main.py?
#   Por organización y reutilización. Cualquier página de la app puede
#   hacer "from utils.loaders import cargar_modelo" sin duplicar código.
#
# ¿QUÉ ES EL CACHÉ DE STREAMLIT? (concepto clave para novatas)
#   Recuerda que Streamlit re-ejecuta el script entero cada vez que
#   el usuario hace algo. Sin caché, cargaría el modelo desde disco
#   en cada clic — muy lento (el modelo pesa varios MB).
#
#   El decorador @st.cache_resource le dice a Streamlit:
#   "la primera vez que alguien llame a esta función, ejecuta el código
#   y guarda el resultado en memoria. Las siguientes veces, devuelve
#   directamente lo guardado sin volver a ejecutar nada."
#
#   Hay dos tipos de caché en Streamlit:
#   - @st.cache_resource → para objetos grandes que NO son datos:
#     modelos, pipelines, conexiones. Se comparte entre todos los usuarios.
#   - @st.cache_data     → para datos (DataFrames, listas, dicts).
#     Cada usuario tiene su propia copia en memoria.
#
# REQUISITOS:
#   - config_app.py debe estar en app/ (un nivel arriba)
#   - Los ficheros de modelo y datos deben existir en las rutas de RUTAS
#
# GENERA:
#   Funciones importables desde cualquier página de la app.
#
# SIGUIENTE:
#   utils/predictor.py — lógica de predicción usando modelo + pipeline
# =============================================================================

import sys
from pathlib import Path

import joblib        # para cargar ficheros .pkl (modelos y pipelines)
import pandas as pd  # para cargar ficheros .parquet (datos)
import streamlit as st  # necesario para los decoradores de caché

# ---------------------------------------------------------------------------
# Aseguramos que Python puede encontrar config_app.py
# ---------------------------------------------------------------------------
# config_app.py está en app/, y este fichero está en app/utils/.
# Python no sabe automáticamente que debe buscar un nivel arriba.
# Con esta línea le decimos: "busca también en la carpeta padre (app/)".
#
# Path(__file__)         → ruta a este fichero: .../app/utils/loaders.py
# .resolve().parent      → carpeta utils/
# .parent                → carpeta app/   ← aquí está config_app.py

_DIR_APP = Path(__file__).resolve().parent.parent
if str(_DIR_APP) not in sys.path:
    sys.path.insert(0, str(_DIR_APP))

from config_app import RUTAS  # importamos solo las rutas que necesitamos aquí


# =============================================================================
# FUNCIÓN 1: Cargar el modelo entrenado
# =============================================================================
# El modelo es un objeto CatBoostClassifier guardado con joblib en Fase 5.
# Pesa varios MB. Con @st.cache_resource solo se carga UNA vez por sesión.

@st.cache_resource(show_spinner="Cargando modelo de predicción...")
def cargar_modelo():
    """
    Carga el modelo CatBoost entrenado desde disco.

    Returns
    -------
    modelo : objeto CatBoostClassifier (o similar)
        El modelo listo para hacer predicciones con .predict() y
        .predict_proba().

    Raises
    ------
    FileNotFoundError
        Si el fichero .pkl no existe en la ruta esperada.
    """
    ruta = RUTAS["modelo"]

    # Comprobación amigable: si el fichero no existe, error claro
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en:\n{ruta}\n\n"
            "Verifica que has ejecutado los notebooks de Fase 5 y que "
            "el fichero CatBoost__balanced.pkl está en data/05_modelado/models/"
        )

    # joblib.load() deserializa el objeto Python guardado en el .pkl
    modelo = joblib.load(ruta)
    return modelo


# =============================================================================
# FUNCIÓN 2: Cargar el pipeline de preprocesamiento
# =============================================================================
# El pipeline transforma los datos brutos del usuario (con missing values,
# variables categóricas, escalas distintas) al formato que espera el modelo.
# SIEMPRE hay que aplicar el pipeline ANTES de llamar al modelo.

@st.cache_resource(show_spinner="Cargando pipeline de preprocesamiento...")
def cargar_pipeline():
    """
    Carga el pipeline de preprocesamiento entrenado desde disco.

    El pipeline incluye: imputación de valores perdidos, codificación
    de variables categóricas y escalado numérico.

    Returns
    -------
    pipeline : objeto sklearn Pipeline
        Listo para transformar datos con .transform().
    """
    ruta = RUTAS["pipeline"]

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el pipeline en:\n{ruta}\n\n"
            "Verifica que el fichero pipeline_preprocesamiento.pkl "
            "está en data/05_modelado/"
        )

    pipeline = joblib.load(ruta)
    return pipeline


# =============================================================================
# FUNCIÓN 3: Cargar los datos de test con metadatos
# =============================================================================
# meta_test.parquet es el conjunto de test generado en f6_m00_preparacion.
# Contiene las features + metadatos (titulación, cohorte, rama, abandono real).
# Lo usamos para mostrar estadísticas reales en la app.

@st.cache_data(show_spinner="Cargando datos de evaluación...")
def cargar_meta_test() -> pd.DataFrame:
    """
    Carga el conjunto de test con metadatos desde disco.

    Contiene 6.725 observaciones con features del modelo más columnas
    adicionales: titulacion, rama, anio_cohorte, abandono (etiqueta real).

    Returns
    -------
    df : pd.DataFrame
        DataFrame con todas las columnas de meta_test.parquet.
    """
    ruta = RUTAS["meta_test"]

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró meta_test en:\n{ruta}\n\n"
            "Ejecuta primero el notebook f6_m00_preparacion.ipynb "
            "para generar este fichero."
        )

    df = pd.read_parquet(ruta)
    return df


# =============================================================================
# FUNCIÓN 4: Cargar los valores SHAP globales
# =============================================================================
# Los valores SHAP explican qué variables influyen más en las predicciones.
# Se calcularon en Fase 6 (f6_m01) sobre el modelo CatBoost.
# Son opcionales: si no existen, la app funciona igualmente sin ellos.

@st.cache_data(show_spinner="Cargando valores SHAP...")
def cargar_shap_global():
    """
    Carga los valores SHAP globales calculados en Fase 6.

    Returns
    -------
    shap_values : objeto shap.Explanation o None
        Los valores SHAP listos para visualizar.
        Devuelve None si el fichero no existe (en lugar de lanzar error),
        para que la app pueda funcionar aunque Fase 6 no esté completa.
    """
    ruta = RUTAS["shap_global"]

    if not ruta.exists():
        # Advertencia visible en la app, pero no error fatal
        st.warning(
            "⚠️ No se encontraron los valores SHAP. "
            "El análisis de importancia de variables no estará disponible. "
            f"Ruta esperada: {ruta}"
        )
        return None

    shap_values = joblib.load(ruta)
    return shap_values


# =============================================================================
# FUNCIÓN 5: Cargar las métricas de fairness (equidad)
# =============================================================================
# fairness_metricas.parquet contiene métricas de equidad por subgrupos
# (género, rama) calculadas en Fase 6 (f6_m05).
# También opcional: la pestaña de equidad la necesita, el resto no.

@st.cache_data(show_spinner="Cargando métricas de equidad...")
def cargar_fairness() -> pd.DataFrame | None:
    """
    Carga las métricas de fairness calculadas en Fase 6.

    Returns
    -------
    df : pd.DataFrame o None
        DataFrame con métricas de equidad por subgrupo.
        Devuelve None si el fichero no existe.
    """
    ruta = RUTAS["fairness"]

    if not ruta.exists():
        st.warning(
            "⚠️ No se encontraron las métricas de equidad. "
            "La pestaña de equidad no estará disponible. "
            f"Ruta esperada: {ruta}"
        )
        return None

    df = pd.read_parquet(ruta)
    return df


# =============================================================================
# FUNCIÓN 6: Cargar todo de una vez (función de conveniencia)
# =============================================================================
# En vez de llamar a las 5 funciones por separado, main.py puede llamar
# a esta única función al arrancar y obtenerlo todo.
# Como cada función individual ya tiene caché, no hay penalización
# por llamarla varias veces.

def cargar_todo() -> dict:
    """
    Carga todos los recursos necesarios para la app de una sola vez.

    Útil para el arranque inicial en main.py. Maneja errores de forma
    centralizada y devuelve un diccionario con todo disponible.

    Returns
    -------
    recursos : dict con claves:
        - "modelo"    : modelo CatBoost
        - "pipeline"  : pipeline sklearn
        - "meta_test" : DataFrame con datos de test
        - "shap"      : valores SHAP (o None)
        - "fairness"  : DataFrame fairness (o None)
        - "ok"        : True si los recursos críticos cargaron bien
        - "errores"   : lista de mensajes de error (vacía si todo ok)
    """
    recursos = {
        "modelo":    None,
        "pipeline":  None,
        "meta_test": None,
        "shap":      None,
        "fairness":  None,
        "ok":        False,
        "errores":   [],
    }

    # Recursos críticos: si fallan, la app no puede funcionar
    try:
        recursos["modelo"] = cargar_modelo()
    except FileNotFoundError as e:
        recursos["errores"].append(str(e))

    try:
        recursos["pipeline"] = cargar_pipeline()
    except FileNotFoundError as e:
        recursos["errores"].append(str(e))

    try:
        recursos["meta_test"] = cargar_meta_test()
    except FileNotFoundError as e:
        recursos["errores"].append(str(e))

    # Recursos opcionales: si fallan, la app funciona en modo reducido
    # (las funciones individuales ya muestran st.warning por su cuenta)
    recursos["shap"]     = cargar_shap_global()
    recursos["fairness"] = cargar_fairness()

    # La app está "ok" solo si los tres recursos críticos cargaron bien
    recursos["ok"] = len(recursos["errores"]) == 0

    return recursos


# =============================================================================
# FIN DE loaders.py
#
# Para usar estas funciones desde una página de la app:
#   from utils.loaders import cargar_modelo, cargar_meta_test
#   modelo    = cargar_modelo()
#   meta_test = cargar_meta_test()
# =============================================================================
