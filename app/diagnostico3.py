import pandas as pd
import joblib
from config_app import RUTAS

# Cargamos los dos ficheros
meta  = pd.read_parquet(RUTAS["meta_test"])
xtest = pd.read_parquet(RUTAS["X_test_prep"])

print("Columnas X_test_prep:", sorted(xtest.columns.tolist()))
print()
print("Columnas meta_test:", sorted(meta.columns.tolist()))
print()

# Columnas que están en los DOS ficheros (posibles duplicados)
duplicadas = [c for c in meta.columns if c in xtest.columns]
print("Columnas en AMBOS ficheros:", duplicadas)
print()

# Lo que espera el pipeline
pipeline = joblib.load(RUTAS["pipeline"])
print("Features del pipeline:", sorted(pipeline.feature_names_in_.tolist()))