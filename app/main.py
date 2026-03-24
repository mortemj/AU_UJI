# =============================================================================
# main.py — versión estable con sidebar y radio buttons
# =============================================================================

import sys
from pathlib import Path
import streamlit as st

_DIR_APP = Path(__file__).resolve().parent
if str(_DIR_APP) not in sys.path:
    sys.path.insert(0, str(_DIR_APP))

from config_app import APP_CONFIG, COLORES, PESTANAS, verificar_ficheros_criticos

st.set_page_config(
    page_title=APP_CONFIG["titulo"],
    page_icon=APP_CONFIG["icono"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    [data-testid="stSidebarNav"] {{display: none !important;}}
    section[data-testid="stSidebar"] > div:first-child {{padding-top: 0.5rem !important;}}
    .main {{background-color: {COLORES["fondo"]};}}
    .metrica-card {{
        background-color: white;
        border: 1px solid {COLORES["borde"]};
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .sidebar-titulo {{
        font-size: 0.85rem;
        color: {COLORES["texto_suave"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

errores = verificar_ficheros_criticos()
if errores:
    st.error("❌ Error al arrancar la app")
    for e in errores:
        st.markdown(f"- `{e}`")
    st.stop()

# Sidebar
with st.sidebar:
    # Logos universidades
    _ROOT = Path(__file__).resolve().parent.parent
    _logo_uji = _ROOT / "docs" / "assets" / "logo_uji.jpg"
    _logo_uoc = _ROOT / "docs" / "assets" / "logo_uoc.jpg"

    # --- NAVEGACIÓN ARRIBA ---
    st.markdown(f"""
    <div style="text-align:center; padding: 0 0 0.2rem 0;">
        <div style="font-size:2rem;">{APP_CONFIG["icono"]}</div>
        <div style="font-size:1rem; font-weight:bold; color:{COLORES["primario"]};">
            Abandono UJI
        </div>
        <div style="font-size:0.72rem; color:{COLORES["texto_suave"]};">
            {APP_CONFIG["subtitulo"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="sidebar-titulo">Navegación</p>', unsafe_allow_html=True)

    opcion_inicio = "🏠  Inicio"
    opciones = [opcion_inicio] + [f"{p['icono']}  {p['titulo']}" for p in PESTANAS]

    pagina_seleccionada = st.radio(
        label="Selecciona una sección:",
        options=opciones,
        index=0,
        label_visibility="collapsed",
    )


    idx_pagina = opciones.index(pagina_seleccionada)

    st.divider()

    # --- LOGOS ABAJO ---
    if _logo_uji.exists():
        st.image(str(_logo_uji), width=160)
    if _logo_uoc.exists():
        st.image(str(_logo_uoc), width=160)

    st.markdown(f"""
    <div style="font-size:0.7rem; color:{COLORES["texto_suave"]}; text-align:center; margin-top:0.8rem;">
        TFM · UOC + UJI · 2025<br>María José Morte Ruiz
    </div>
    """, unsafe_allow_html=True)

# Enrutamiento
id_pagina = "inicio" if idx_pagina == 0 else PESTANAS[idx_pagina - 1]["id"]

if id_pagina == "inicio":
    from pages import p00_inicio as pagina
    pagina.show()
elif id_pagina == "institucional":
    from pages import p01_institucional as pagina
    pagina.show()
elif id_pagina == "titulacion":
    from pages import p02_titulacion as pagina
    pagina.show()
elif id_pagina == "prospecto":
    from pages import p03_prospecto as pagina
    pagina.show()
elif id_pagina == "en_curso":
    from pages import p04_en_curso as pagina
    pagina.show()
elif id_pagina == "equidad":
    from pages import p05_equidad as pagina
    pagina.show()
