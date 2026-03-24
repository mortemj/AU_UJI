import pandas as pd
from pathlib import Path

meta = pd.read_parquet(
    Path("C:/Users/mjmor/OneDrive - Universitat Jaume I/2.- AU_UJI/data/06_evaluacion/meta_test.parquet")
)
print("Ramas exactas en tus datos:")
print(sorted(meta['rama'].dropna().unique().tolist()))