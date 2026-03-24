# =============================================================================
# _path_setup.py
# Configura sys.path para que utils/ y config_app.py sean siempre localizables
#
# ¿POR QUÉ EXISTE ESTE FICHERO?
#   Streamlit en Windows (especialmente con OneDrive) puede ejecutar las
#   páginas de pages/ desde un directorio de trabajo distinto a app/.
#   Esto hace que "from utils.loaders import ..." falle con ModuleNotFoundError.
#
#   Este fichero se importa al inicio de CADA página con:
#       import _path_setup  # noqa
#   y garantiza que app/ está siempre en sys.path, sin importar desde
#   dónde ejecute Streamlit.
#
# DÓNDE VA:
#   app/_path_setup.py  (misma carpeta que main.py y config_app.py)
# =============================================================================

import os
import sys

# Calculamos la ruta a app/ de forma robusta
# os.path.abspath(__file__)  → ruta absoluta a este fichero (_path_setup.py)
# os.path.dirname(...)       → carpeta que lo contiene (app/)
_DIR_APP = os.path.dirname(os.path.abspath(__file__))

# Lo añadimos al inicio de sys.path si no está ya
# insert(0, ...) = máxima prioridad — Python busca aquí primero
if _DIR_APP not in sys.path:
    sys.path.insert(0, _DIR_APP)
