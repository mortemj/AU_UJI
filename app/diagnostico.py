import os
from pathlib import Path
from config_app import ROOT

ruta_xtest = ROOT / "data" / "05_modelado" / "X_test.parquet"
print('Ruta:', ruta_xtest)
print('Existe:', ruta_xtest.exists())
if ruta_xtest.exists():
    print('Tamaño:', os.path.getsize(ruta_xtest), 'bytes')

# También miramos X_test_prep
ruta_prep = ROOT / "data" / "05_modelado" / "X_test_prep.parquet"
print()
print('Ruta prep:', ruta_prep)
print('Existe prep:', ruta_prep.exists())
if ruta_prep.exists():
    print('Tamaño prep:', os.path.getsize(ruta_prep), 'bytes')
    import pandas as pd
    df = pd.read_parquet(ruta_prep)
    print('Shape prep:', df.shape)
    print('Columnas:', sorted(df.columns.tolist()))