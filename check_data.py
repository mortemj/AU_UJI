import pandas as pd
for f in ['df_exp_automl_target.parquet', 'df_eda_con_target.parquet', 'df_exp_target_eda.parquet']:
    df = pd.read_parquet(f'data/03_features/{f}')
    print(f, df.shape, list(df.columns))
    print()