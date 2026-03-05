# ============================================================================
# PROGRESS.PY - Barra de progreso con color para notebooks
# ============================================================================
# Ubicación: src/utils/progress.py
# 
# Uso:
#   from src.utils import progreso
#   for item in progreso(lista, "📖 Leyendo datos"):
#       ...
# ============================================================================

def progreso(iterable, desc="Procesando", total=None):
    """
    Barra de progreso con color para Jupyter Notebooks.
    
    Detecta automáticamente el entorno (Jupyter/terminal) y usa
    la mejor visualización disponible.
    
    Parameters
    ----------
    iterable : iterable#
        Elementos a iterar
    desc : str
        Descripción que aparece antes de la barra
    total : int, optional
        Total de elementos (se calcula automáticamente si no se indica)
    
    Returns
    -------
    tqdm
        Iterador con barra de progreso
    
    Examples
    --------
    >>> from src.utils import progreso
    >>> for tabla in progreso(tablas, "📖 Leyendo Excels"):
    ...     procesar(tabla)
    """
    try:
        # Intentar usar tqdm.notebook (mejor para Jupyter)
        from tqdm.notebook import tqdm
    except ImportError:
        try:
            # Fallback a tqdm normal con color
            from tqdm import tqdm
            return tqdm(iterable, desc=desc, total=total, colour='#3182ce')
        except ImportError:
            # Sin tqdm, devolver iterable normal
            print(f"⚠️ tqdm no instalado. Ejecuta: pip install tqdm")
            return iterable
    
    return tqdm(iterable, desc=desc, total=total)


def progreso_manual(total, desc="Procesando"):
    """
    Barra de progreso manual para actualizar paso a paso.
    
    Útil cuando no se puede usar un bucle for simple.
    
    Parameters
    ----------
    total : int
        Número total de pasos
    desc : str
        Descripción de la tarea
    
    Returns
    -------
    tqdm
        Objeto tqdm para actualizar manualmente con .update(1)
    
    Examples
    --------
    >>> pbar = progreso_manual(10, "📊 Generando reportes")
    >>> for i in range(10):
    ...     hacer_algo()
    ...     pbar.update(1)
    >>> pbar.close()
    """
    try:
        from tqdm.notebook import tqdm
    except ImportError:
        try:
            from tqdm import tqdm
            return tqdm(total=total, desc=desc, colour='#3182ce')
        except ImportError:
            print(f"⚠️ tqdm no instalado")
            return None
    
    return tqdm(total=total, desc=desc)
