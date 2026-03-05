# ============================================================================
# VERIFICAR LIBRERÍAS INSTALADAS
# ============================================================================
# Ejecuta este script en tu entorno para ver qué tienes instalado
# ============================================================================

import subprocess
import sys

# Librerías que usamos en Fase 2
LIBRERIAS_FASE2 = [
    'pandas',
    'numpy', 
    'pyarrow',
    'plotly',
    'altair',
    'matplotlib',
    'missingno',
    'scipy',
    'tqdm',
    'Jinja2',
    'sweetviz',
    'seaborn',
]

print('=' * 70)
print('VERIFICACIÓN DE LIBRERÍAS - FASE 2 EDA')
print('=' * 70)
print(f'Python: {sys.version}')
print('=' * 70)

instaladas = []
no_instaladas = []

for lib in LIBRERIAS_FASE2:
    try:
        if lib == 'Jinja2':
            import jinja2
            version = jinja2.__version__
        else:
            mod = __import__(lib)
            version = getattr(mod, '__version__', 'sin versión')
        instaladas.append((lib, version))
        print(f'  ✅ {lib}: {version}')
    except ImportError:
        no_instaladas.append(lib)
        print(f'  ❌ {lib}: NO INSTALADA')

print('=' * 70)
print(f'RESUMEN: {len(instaladas)} instaladas, {len(no_instaladas)} faltantes')
print('=' * 70)

if no_instaladas:
    print('\n📦 Para instalar las faltantes con pip:')
    print(f'   pip install {" ".join(no_instaladas)}')
    print('\n📦 Para instalar las faltantes con conda:')
    print(f'   conda install {" ".join(no_instaladas)}')
