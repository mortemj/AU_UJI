import pandas as pd
from pathlib import Path
from config_app import ROOT

# Cargamos los dos ficheros
X_test_prep = pd.read_parquet(ROOT / "data" / "05_modelado" / "X_test_prep.parquet")
meta_actual  = pd.read_parquet(ROOT / "data" / "06_evaluacion" / "meta_test.parquet")

print("X_test_prep shape:", X_test_prep.shape)
print("meta_actual shape:", meta_actual.shape)
print()
print("Índices X_test_prep:", X_test_prep.index[:5].tolist())
print("Índices meta_actual:", meta_actual.index[:5].tolist())
print()

# Columnas de metadatos que queremos añadir a las features
cols_meta = ['abandono', 'titulacion', 'rama', 'sexo', 'via_acceso', 
             'curso_aca_ini', 'per_id_ficticio']
cols_meta_ok = [c for c in cols_meta if c in meta_actual.columns]
print("Cols meta disponibles:", cols_meta_ok)