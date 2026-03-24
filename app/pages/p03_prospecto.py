# =============================================================================
# p03_prospecto.py
# Pestaña 3 — Pronóstico para alumno prospecto
#
# ¿QUÉ HACE ESTE FICHERO?
#   Wrapper fino que llama a pronostico_shared.show_pronostico()
#   en modo "prospecto" (sin nota del primer año).
#
# TODO el contenido real está en utils/pronostico_shared.py
# Este fichero solo existe para que la pestaña tenga su propio
# nombre y posición en el menú lateral.
#
# SIGUIENTE:
#   pages/p04_en_curso.py
# =============================================================================

import sys
from pathlib import Path

import streamlit as st

# _path_setup añade app/ a sys.path de forma robusta en Windows/OneDrive
import _path_setup  # noqa: F401

from utils.pronostico_shared import show_pronostico


def show():
    """Renderiza el pronóstico en modo prospecto (antes de matricularse)."""
    show_pronostico(modo="prospecto")
