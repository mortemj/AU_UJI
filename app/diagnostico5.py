import sys
sys.path.insert(0, r"C:\Users\mjmor\OneDrive - Universitat Jaume I\2.- AU_UJI\app")
from utils.loaders import cargar_meta_test
df = cargar_meta_test()
print("Shape:", df.shape)
print("Columnas:", sorted(df.columns.tolist()))
print()
print("rama (primeros 5):", df['rama'].head().tolist())
if 'rama_meta' in df.columns:
    print("rama_meta (primeros 5):", df['rama_meta'].head().tolist())
else:
    print("rama_meta: NO EXISTE")