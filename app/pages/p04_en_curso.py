# =============================================================================
# p04_en_curso.py
# Pestaña 4 — Pronóstico para alumno en curso
#
# ¿QUÉ HACE ESTE FICHERO?
#   Wrapper fino que llama a pronostico_shared.show_pronostico()
#   en modo "en_curso" (con nota del primer año y créditos superados).
#
# TODO el contenido real está en utils/pronostico_shared.py
# Este fichero solo existe para que la pestaña tenga su propio
# nombre y posición en el menú lateral.
#
# SIGUIENTE:
#   pages/p05_equidad.py
# =============================================================================

import sys
from pathlib import Path

import streamlit as st

# _path_setup añade app/ a sys.path de forma robusta en Windows/OneDrive
import _path_setup  # noqa: F401

from utils.pronostico_shared import show_pronostico


def show():
    """Renderiza el pronóstico en modo en curso (alumno ya matriculado)."""
    show_pronostico(modo="en_curso")
