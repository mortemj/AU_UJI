# ============================================================================
# CONFIG_MODELADO.PY — Configuración central de la Fase 5: Modelado Clásico
# ============================================================================
# TFM: Predicción de Abandono Universitario
# Autora: María José Morte | UOC + UJI
#
# PROPÓSITO:
#   Este archivo es la ÚNICA fuente de verdad para todos los parámetros
#   de modelado. Ningún notebook de Fase 5 contiene valores hardcodeados:
#   todo —algoritmos, métricas, folds, semillas, colores— se lee aquí.
#
# ¿Por qué centralizar?
#   - Cambiar CV_FOLDS de 5 a 10 afecta a TODOS los módulos con un solo edit.
#   - El tribunal puede ver de un vistazo todas las decisiones metodológicas.
#   - Facilita la reproducibilidad: random_state siempre consistente.
#
# ¿Quién lo usa?
#   - f5_m00_indice.ipynb    → lee MODULOS_FASE5, BASELINE_AUTOML
#   - f5_m01_lineales.ipynb  → lee ALGORITMOS_LINEALES, CV_CONFIG, METRICAS
#   - f5_m02_arboles.ipynb   → lee ALGORITMOS_ARBOLES, CV_CONFIG, METRICAS
#   - f5_m03_boosting.ipynb  → lee ALGORITMOS_BOOSTING, CV_CONFIG, METRICAS
#   - f5_m04_otros.ipynb     → lee ALGORITMOS_OTROS, CV_CONFIG, METRICAS
#   - f5_m05_mlp_ebm.ipynb     → lee ALGORITMOS_MLP_EBM
#   - f5_m06_ensambles.ipynb   → lee ALGORITMOS_ENSAMBLES
#   - f5_m07_comparacion.ipynb → lee TODO para la tabla maestra
#
# NOTA: Este archivo se importa a través de src.config_modelado.
#   from src.config_modelado import CV_CONFIG, METRICAS_EVALUAR, ALGORITMOS_LINEALES
# ============================================================================

from typing import Dict, List, Any


# ============================================================================
# 1. CONFIGURACIÓN DE VALIDACIÓN CRUZADA
# ============================================================================
# Parámetros compartidos por TODOS los módulos de modelado.
#
# ¿Por qué estas decisiones?
#   - RANDOM_STATE=42: convención estándar para reproducibilidad.
#   - TEST_SIZE=0.2: split 80/20, equilibrio entre train suficiente y test fiable.
#   - CV_FOLDS=5: compromiso entre varianza del estimador y coste computacional.
#     Con 33.621 muestras, cada fold tiene ~5.400 muestras — estadísticamente robusto.
#   - ESTRATIFICADO=True: esencial con desbalance (29% abandono).

CV_CONFIG: Dict[str, Any] = {
    'random_state': 42,       # Semilla global — garantiza reproducibilidad total
    'test_size': 0.20,        # 20% test ≈ 6.724 muestras
    'cv_folds': 5,            # 5-Fold Stratified CV sobre train
    'estratificado': True,    # Mantiene proporción abandono/no-abandono en cada fold
    'n_jobs': -1,             # Usa todos los cores disponibles
    'verbose': 0,             # Sin output de scikit-learn (el notebook lo gestiona)
}


# ============================================================================
# 2. ESTRATEGIAS DE BALANCE DE CLASES
# ============================================================================
# El dataset tiene 29.2% abandono — desbalance moderado que puede sesgar
# modelos hacia la clase mayoritaria (no abandono).
#
# Se evalúan 3 estrategias para comparar su impacto en F1-macro:
#   A) Sin ajuste: línea base, ve el desbalance real
#   B) class_weight="balanced": penaliza errores en clase minoritaria
#   C) SMOTE: genera muestras sintéticas de la clase minoritaria
#
# La estrategia ganadora se documenta en f5_m05_comparativa.

ESTRATEGIAS_BALANCE: List[Dict[str, Any]] = [
    {
        'id': 'sin_ajuste',
        'nombre': 'Sin ajuste',
        'descripcion': 'Distribución original del dataset (29.2% abandono)',
        'class_weight': None,
        'usar_smote': False,
        'color': '#3182ce',
        'emoji': '⚖️',
    },
    {
        'id': 'balanced',
        'nombre': 'class_weight="balanced"',
        'descripcion': 'Peso inversamente proporcional a la frecuencia de clase',
        'class_weight': 'balanced',
        'usar_smote': False,
        'color': '#38a169',
        'emoji': '⚖️',
    },
    {
        'id': 'smote',
        'nombre': 'SMOTE',
        'descripcion': 'Synthetic Minority Over-sampling Technique sobre train',
        'class_weight': None,
        'usar_smote': True,
        'color': '#805ad5',
        'emoji': '🔄',
    },
]


# ============================================================================
# 3. MÉTRICAS DE EVALUACIÓN
# ============================================================================
# Métricas calculadas para TODOS los modelos. Orden de prioridad:
#   1. F1-macro: métrica principal — equilibra precisión y recall en ambas clases.
#      Elegida sobre accuracy porque el dataset está desbalanceado.
#   2. AUC-ROC: capacidad discriminativa del modelo, independiente del umbral.
#   3. Precision/Recall: para analizar el trade-off en contexto educativo.
#      (Recall alto = detectar más abandonos; Precision alta = menos falsos positivos)
#   4. Kappa de Cohen: acuerdo más allá del azar, útil con desbalance.
#   5. Tiempo: relevante para despliegue en producción institucional.

METRICAS_EVALUAR: List[Dict[str, Any]] = [
    {
        'id': 'f1_macro',
        'nombre': 'F1-macro',
        'descripcion': 'Media armónica de Precision y Recall sobre ambas clases',
        'sklearn_key': 'f1_macro',    # Clave para cross_val_score o classification_report
        'principal': True,            # Métrica de ranking principal
        'formato': '.4f',
        'emoji': '🎯',
        'color': '#3182ce',
    },
    {
        'id': 'auc_roc',
        'nombre': 'AUC-ROC',
        'descripcion': 'Área bajo la curva ROC — capacidad discriminativa',
        'sklearn_key': 'roc_auc',
        'principal': False,
        'formato': '.4f',
        'emoji': '📈',
        'color': '#38a169',
    },
    {
        'id': 'precision_macro',
        'nombre': 'Precision macro',
        'descripcion': 'Proporción de predicciones positivas correctas',
        'sklearn_key': 'precision_macro',
        'principal': False,
        'formato': '.4f',
        'emoji': '🔍',
        'color': '#805ad5',
    },
    {
        'id': 'recall_macro',
        'nombre': 'Recall macro',
        'descripcion': 'Proporción de positivos reales detectados',
        'sklearn_key': 'recall_macro',
        'principal': False,
        'formato': '.4f',
        'emoji': '🔔',
        'color': '#ed8936',
    },
    {
        'id': 'accuracy',
        'nombre': 'Accuracy',
        'descripcion': 'Proporción de predicciones correctas (informativa, no principal)',
        'sklearn_key': 'accuracy',
        'principal': False,
        'formato': '.4f',
        'emoji': '✅',
        'color': '#319795',
    },
    {
        'id': 'kappa',
        'nombre': 'Kappa de Cohen',
        'descripcion': 'Acuerdo inter-rater normalizado — robusto ante desbalance',
        'sklearn_key': 'cohen_kappa',  # Calcular manualmente con cohen_kappa_score
        'principal': False,
        'formato': '.4f',
        'emoji': '🤝',
        'color': '#e53e3e',
    },
    {
        'id': 'tiempo_s',
        'nombre': 'Tiempo (s)',
        'descripcion': 'Tiempo de entrenamiento en segundos — relevante para producción',
        'sklearn_key': None,           # Se mide con time.time()
        'principal': False,
        'formato': '.2f',
        'emoji': '⏱️',
        'color': '#718096',
    },
]

# Métrica principal (la usada para ranking y selección)
METRICA_PRINCIPAL = next(m for m in METRICAS_EVALUAR if m['principal'])


# ============================================================================
# 4. BASELINE AUTOML DE REFERENCIA
# ============================================================================
# Resultados del screening AutoML (168 modelos, 4 frameworks).
# Fase 5 debe superar o igualar este baseline para justificarse.
#
# Fuente: fase_automl/m06_comparativa — CatBoost_BAG_L2 sobre D_strict
# Dataset: dataset_final_tfm.parquet (33.621 × 20)

BASELINE_AUTOML: Dict[str, Any] = {
    'modelo': 'CatBoost_BAG_L2',
    'framework': 'AutoGluon',
    'dataset': 'D_strict',
    'f1_macro': 0.7970,
    'auc_roc': 0.93,
    'descripcion': (
        'Mejor modelo del screening AutoML (168 modelos, 4 frameworks). '
        'Familia Gradient Boosting confirmada como óptima. '
        'Fase 5 parte de este baseline como referencia mínima.'
    ),
    'color': '#e53e3e',    # Rojo para línea de referencia en gráficos
    'linestyle': '--',     # Línea discontinua en curvas comparativas
}


# ============================================================================
# 5. ALGORITMOS POR MÓDULO
# ============================================================================
# Cada módulo de Fase 5 tiene su lista de algoritmos.
# La estructura permite que f5_m05_comparativa ensamble la tabla maestra
# automáticamente sin conocer los detalles de cada módulo.
#
# ¿Por qué estos algoritmos?
#   - Cobertura de las principales familias del ML supervisado clásico.
#   - Coherente con literatura de predicción de abandono universitario.
#   - EBM (InterpretML) incluido en m04 como puente hacia Fase 6.

ALGORITMOS_LINEALES: List[Dict[str, Any]] = [
    # LogisticRegression: baseline interpretable por excelencia.
    # Variantes L1/L2/ElasticNet cubren regularización y selección de features.
    {
        'id': 'lr_l2',
        'nombre': 'Logistic Regression (L2)',
        'clase': 'LogisticRegression',
        'params': {'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs',
                   'max_iter': 1000, 'random_state': 42},
        'descripcion': 'Regularización Ridge — penaliza pesos grandes, shrinkage suave',
        'emoji': '📈',
        'color': '#3182ce',
    },
    {
        'id': 'lr_l1',
        'nombre': 'Logistic Regression (L1)',
        'clase': 'LogisticRegression',
        'params': {'penalty': 'l1', 'C': 1.0, 'solver': 'liblinear',
                   'max_iter': 1000, 'random_state': 42},
        'descripcion': 'Regularización Lasso — puede llevar coeficientes a cero (selección)',
        'emoji': '📈',
        'color': '#2b6cb0',
    },
    {
        'id': 'lr_elasticnet',
        'nombre': 'Logistic Regression (ElasticNet)',
        'clase': 'LogisticRegression',
        'params': {'penalty': 'elasticnet', 'C': 1.0, 'solver': 'saga',
                   'l1_ratio': 0.5, 'max_iter': 1000, 'random_state': 42},
        'descripcion': 'Combinación L1+L2 — equilibrio entre shrinkage y selección',
        'emoji': '📈',
        'color': '#1a365d',
    },
    {
        'id': 'ridge',
        'nombre': 'Ridge Classifier',
        'clase': 'RidgeClassifier',
        'params': {'alpha': 1.0, 'random_state': 42},
        'descripcion': 'Regresión Ridge adaptada a clasificación — muy rápido',
        'emoji': '📏',
        'color': '#4299e1',
    },
]

ALGORITMOS_ARBOLES: List[Dict[str, Any]] = [
    {
        'id': 'dt',
        'nombre': 'Decision Tree',
        'clase': 'DecisionTreeClassifier',
        'params': {'max_depth': None, 'random_state': 42},
        'descripcion': 'Árbol sin podar — referencia de árbol simple, interpretable',
        'emoji': '🌳',
        'color': '#38a169',
    },
    {
        'id': 'rf',
        'nombre': 'Random Forest',
        'clase': 'RandomForestClassifier',
        'params': {'n_estimators': 200, 'max_depth': None,
                   'n_jobs': -1, 'random_state': 42},
        'descripcion': 'Ensemble de árboles con bagging — robusto y con feature importance',
        'emoji': '🌲',
        'color': '#2f855a',
    },
    {
        'id': 'et',
        'nombre': 'Extra Trees',
        'clase': 'ExtraTreesClassifier',
        'params': {'n_estimators': 200, 'max_depth': None,
                   'n_jobs': -1, 'random_state': 42},
        'descripcion': 'Random Forest con splits completamente aleatorios — menor varianza',
        'emoji': '🌿',
        'color': '#276749',
    },
]

ALGORITMOS_BOOSTING: List[Dict[str, Any]] = [
    # Familia ganadora del AutoML. Se busca superar F1=0.7970.
    {
        'id': 'xgboost',
        'nombre': 'XGBoost',
        'clase': 'XGBClassifier',
        'modulo': 'xgboost',
        'params': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6,
                   'use_label_encoder': False, 'eval_metric': 'logloss',
                   'n_jobs': -1, 'random_state': 42},
        'descripcion': 'Gradient Boosting con regularización L1/L2 y manejo nativo de nulos',
        'emoji': '🚀',
        'color': '#805ad5',
    },
    {
        'id': 'lightgbm',
        'nombre': 'LightGBM',
        'clase': 'LGBMClassifier',
        'modulo': 'lightgbm',
        'params': {'n_estimators': 200, 'learning_rate': 0.1,
                   'num_leaves': 31, 'n_jobs': -1,
                   'random_state': 42, 'verbose': -1},
        'descripcion': 'Leaf-wise boosting — más rápido que XGBoost en datasets grandes',
        'emoji': '💡',
        'color': '#553c9a',
    },
    {
        'id': 'catboost',
        'nombre': 'CatBoost',
        'clase': 'CatBoostClassifier',
        'modulo': 'catboost',
        'params': {'iterations': 200, 'learning_rate': 0.1, 'depth': 6,
                   'random_seed': 42, 'verbose': 0},
        'descripcion': 'Boosting simétrico con manejo nativo de categóricas — ganador AutoML',
        'emoji': '🐱',
        'color': '#44337a',
    },
]

ALGORITMOS_OTROS: List[Dict[str, Any]] = [
    {
        'id': 'knn',
        'nombre': 'K-Nearest Neighbors',
        'clase': 'KNeighborsClassifier',
        'modulo': 'sklearn.neighbors',
        'params': {'n_neighbors': 5, 'weights': 'uniform', 'n_jobs': -1},
        'descripcion': 'Basado en instancias — sensible a escala, sin entrenamiento explícito',
        'emoji': '📍',
        'color': '#ed8936',
    },
    {
        'id': 'gnb',
        'nombre': 'Gaussian Naive Bayes',
        'clase': 'GaussianNB',
        'modulo': 'sklearn.naive_bayes',
        'params': {},
        'descripcion': 'Asume independencia y distribución normal — muy rápido',
        'emoji': '🎲',
        'color': '#c05621',
    },
    {
        'id': 'bnb',
        'nombre': 'Bernoulli Naive Bayes',
        'clase': 'BernoulliNB',
        'modulo': 'sklearn.naive_bayes',
        'params': {},
        'descripcion': 'Variante para features binarias — complementa GaussianNB',
        'emoji': '🎯',
        'color': '#dd6b20',
    },
    {
        'id': 'mlp',
        'nombre': 'MLP Classifier',
        'clase': 'MLPClassifier',
        'modulo': 'sklearn.neural_network',
        'params': {'hidden_layer_sizes': (100,), 'max_iter': 500,
                   'random_state': 42, 'early_stopping': True},
        'descripcion': 'Red neuronal densa — puente entre ML clásico y deep learning',
        'emoji': '🧠',
        'color': '#e53e3e',
    },
    {
        'id': 'ebm',
        'nombre': 'EBM (InterpretML)',
        'clase': 'ExplainableBoostingClassifier',
        'modulo': 'interpret.glassbox',
        'params': {'random_state': 42},
        'descripcion': (
            'Explainable Boosting Machine — GAM moderno con interpretabilidad nativa. '
            'Puente hacia Fase 6: el único modelo "caja de cristal" del conjunto.'
        ),
        'emoji': '🔬',
        'color': '#319795',
    },
]

# Mapa completo: módulo → lista de algoritmos
# Permite a f5_m05_comparativa iterar sobre todos sin conocer los detalles


# ============================================================================
# 5b. ALGORITMOS MLP + EBM (módulo m05)
# ============================================================================
ALGORITMOS_MLP_EBM: List[Dict[str, Any]] = [
    {
        'id': 'mlp',
        'nombre': 'MLP Classifier',
        'clase': 'MLPClassifier',
        'modulo': 'sklearn.neural_network',
        'params': {'hidden_layer_sizes': (100, 50), 'activation': 'relu',
                   'solver': 'adam', 'max_iter': 500, 'early_stopping': True,
                   'random_state': 42},
        'descripcion': 'Red neuronal multicapa — captura relaciones no lineales complejas',
        'emoji': '🧠',
        'color': '#3182ce',
    },
    {
        'id': 'ebm',
        'nombre': 'EBM (InterpretML)',
        'clase': 'ExplainableBoostingClassifier',
        'modulo': 'interpret.glassbox',
        'params': {'max_bins': 256, 'interactions': 10, 'learning_rate': 0.01,
                   'max_rounds': 5000, 'random_state': 42},
        'descripcion': 'Explainable Boosting Machine — precisión de boosting con interpretabilidad nativa',
        'emoji': '🔍',
        'color': '#805ad5',
    },
]

# ============================================================================
# 5c. ALGORITMOS ENSAMBLES (módulo m06)
# ============================================================================
ALGORITMOS_ENSAMBLES: List[Dict[str, Any]] = [
    {
        'id': 'voting',
        'nombre': 'VotingClassifier',
        'clase': 'VotingClassifier',
        'modulo': 'sklearn.ensemble',
        'params': {'voting': 'soft'},
        'descripcion': 'Votación suave de modelos base heterogéneos',
        'emoji': '🗳️',
        'color': '#3182ce',
    },
    {
        'id': 'stacking',
        'nombre': 'StackingClassifier',
        'clase': 'StackingClassifier',
        'modulo': 'sklearn.ensemble',
        'params': {},
        'descripcion': 'Meta-modelo LogReg que aprende a combinar predicciones base',
        'emoji': '🔗',
        'color': '#38a169',
    },
    {
        'id': 'bagging',
        'nombre': 'BaggingClassifier',
        'clase': 'BaggingClassifier',
        'modulo': 'sklearn.ensemble',
        'params': {'n_estimators': 100, 'random_state': 42},
        'descripcion': 'Bootstrap aggregating con base SVM — reduce varianza',
        'emoji': '🎒',
        'color': '#ed8936',
    },
    {
        'id': 'adaboost',
        'nombre': 'AdaBoost',
        'clase': 'AdaBoostClassifier',
        'modulo': 'sklearn.ensemble',
        'params': {'n_estimators': 200, 'learning_rate': 0.1, 'random_state': 42},
        'descripcion': 'Boosting adaptativo clásico con árboles de decisión',
        'emoji': '⚡',
        'color': '#d69e2e',
    },
]

ALGORITMOS_POR_MODULO: Dict[str, List[Dict]] = {
    'm01_lineales':  ALGORITMOS_LINEALES,
    'm02_arboles':   ALGORITMOS_ARBOLES,
    'm03_boosting':  ALGORITMOS_BOOSTING,
    'm04_otros':     ALGORITMOS_OTROS,
    'm05_mlp_ebm':   ALGORITMOS_MLP_EBM,
    'm06_ensambles': ALGORITMOS_ENSAMBLES,
}

# Lista plana de todos los algoritmos (para comparativa y conteos)
TODOS_LOS_ALGORITMOS: List[Dict] = (
    ALGORITMOS_LINEALES
    + ALGORITMOS_ARBOLES
    + ALGORITMOS_BOOSTING
    + ALGORITMOS_MLP_EBM
    + ALGORITMOS_ENSAMBLES
    + ALGORITMOS_OTROS
)


# ============================================================================
# 6. MÓDULOS DE FASE 5 (para generar tarjetas HTML en m00)
# ============================================================================
# Derivado automáticamente de las listas de algoritmos — sin hardcodes.

MODULOS_FASE5: List[Dict[str, Any]] = [
    {
        'id': 'f5_m01',
        'nombre': 'Modelos Lineales',
        'archivo_html': 'm01_lineales.html',
        'emoji': '📈',
        'color': '#3182ce',
        'algoritmos': ALGORITMOS_LINEALES,
        'n_algoritmos': len(ALGORITMOS_LINEALES),
        'descripcion': (
            f'{len(ALGORITMOS_LINEALES)} modelos: '
            + ', '.join(a['nombre'] for a in ALGORITMOS_LINEALES)
            + '. Baseline interpretable y referencia para modelos complejos.'
        ),
    },
    {
        'id': 'f5_m02',
        'nombre': 'Árboles',
        'archivo_html': 'm02_arboles.html',
        'emoji': '🌲',
        'color': '#38a169',
        'algoritmos': ALGORITMOS_ARBOLES,
        'n_algoritmos': len(ALGORITMOS_ARBOLES),
        'descripcion': (
            f'{len(ALGORITMOS_ARBOLES)} modelos: '
            + ', '.join(a['nombre'] for a in ALGORITMOS_ARBOLES)
            + '. Modelos basados en árboles con interpretabilidad nativa.'
        ),
    },
    {
        'id': 'f5_m03',
        'nombre': 'Gradient Boosting',
        'archivo_html': 'm03_boosting.html',
        'emoji': '🚀',
        'color': '#805ad5',
        'algoritmos': ALGORITMOS_BOOSTING,
        'n_algoritmos': len(ALGORITMOS_BOOSTING),
        'descripcion': (
            f'{len(ALGORITMOS_BOOSTING)} modelos: '
            + ', '.join(a['nombre'] for a in ALGORITMOS_BOOSTING)
            + f'. Familia ganadora del AutoML. '
            f'Baseline a superar: F1={BASELINE_AUTOML["f1_macro"]}.'
        ),
    },
    {
        'id': 'f5_m04',
        'nombre': 'Otros Algoritmos',
        'archivo_html': 'm04_otros.html',
        'emoji': '🧪',
        'color': '#ed8936',
        'algoritmos': ALGORITMOS_OTROS,
        'n_algoritmos': len(ALGORITMOS_OTROS),
        'descripcion': (
            f'{len(ALGORITMOS_OTROS)} modelos: '
            + ', '.join(a['nombre'] for a in ALGORITMOS_OTROS)
            + '. Cobertura completa del espacio de algoritmos clásicos.'
        ),
    },
    {
        'id': 'f5_m05',
        'nombre': 'MLP + EBM',
        'archivo_html': 'm05_mlp_ebm.html',
        'emoji': '🧠',
        'color': '#3182ce',
        'algoritmos': ALGORITMOS_MLP_EBM,
        'n_algoritmos': len(ALGORITMOS_MLP_EBM),
        'descripcion': (
            f'{len(ALGORITMOS_MLP_EBM)} modelos: '
            + ', '.join(a['nombre'] for a in ALGORITMOS_MLP_EBM)
            + '. Redes neuronales y boosting interpretable.'
        ),
    },
    {
        'id': 'f5_m06',
        'nombre': 'Ensambles',
        'archivo_html': 'm06_ensambles.html',
        'emoji': '🔗',
        'color': '#38a169',
        'algoritmos': ALGORITMOS_ENSAMBLES,
        'n_algoritmos': len(ALGORITMOS_ENSAMBLES),
        'descripcion': (
            f'{len(ALGORITMOS_ENSAMBLES)} modelos: '
            + ', '.join(a['nombre'] for a in ALGORITMOS_ENSAMBLES)
            + '. Ensambles heterogéneos — Voting, Stacking, Bagging, AdaBoost.'
        ),
    },
    {
        'id': 'f5_m07',
        'nombre': 'Comparativa Final',
        'archivo_html': 'm07_comparacion.html',
        'emoji': '🏆',
        'color': '#e53e3e',
        'algoritmos': [],
        'n_algoritmos': len(TODOS_LOS_ALGORITMOS),
        'descripcion': (
            f'Tabla maestra de {len(TODOS_LOS_ALGORITMOS)} modelos. '
            'Curvas ROC comparativas. Ranking final. '
            'Selección del top-3 para Fase 6 (interpretabilidad + fairness).'
        ),
    },
]


# ============================================================================
# 7. RUTAS DE RESULTADOS
# ============================================================================
# Nombres de ficheros donde cada módulo guarda sus resultados.
# f5_m05_comparativa los carga todos para construir la tabla maestra.

RUTAS_RESULTADOS: Dict[str, str] = {
    'm01_lineales':  'resultados/f5_m01_lineales.parquet',
    'm02_arboles':   'resultados/f5_m02_arboles.parquet',
    'm03_boosting':  'resultados/f5_m03_boosting.parquet',
    'm04_otros':     'resultados/f5_m04_otros.parquet',
}

# Nombre del archivo de métricas JSON (sistema src.metricas)
METRICAS_JSON_KEY = 'fase5_resultados'


# ============================================================================
# 8. FUNCIONES DE UTILIDAD (sin dependencias externas)
# ============================================================================

def get_algoritmo_por_id(alg_id: str) -> Dict[str, Any]:
    """
    Busca un algoritmo por su id en todas las listas.

    Parameters
    ----------
    alg_id : str
        Identificador del algoritmo (ej: 'xgboost', 'lr_l2')

    Returns
    -------
    Dict con la configuración del algoritmo, o {} si no se encuentra.
    """
    for alg in TODOS_LOS_ALGORITMOS:
        if alg['id'] == alg_id:
            return alg
    return {}


def get_metrica_principal() -> Dict[str, Any]:
    """Devuelve la métrica marcada como principal."""
    return METRICA_PRINCIPAL


def resumen_config() -> None:
    """Imprime un resumen de la configuración para verificación en notebooks."""
    total = len(TODOS_LOS_ALGORITMOS)
    print('=' * 65)
    print('CONFIG_MODELADO — Fase 5: Modelado Clásico')
    print('=' * 65)
    print(f'  CV:              {CV_CONFIG["cv_folds"]}-Fold Stratified | '
          f'random_state={CV_CONFIG["random_state"]} | '
          f'test_size={CV_CONFIG["test_size"]}')
    print(f'  Estrategias:     {len(ESTRATEGIAS_BALANCE)} '
          f'({", ".join(e["id"] for e in ESTRATEGIAS_BALANCE)})')
    print(f'  Métricas:        {len(METRICAS_EVALUAR)} '
          f'| Principal: {METRICA_PRINCIPAL["nombre"]}')
    print(f'  Baseline AutoML: {BASELINE_AUTOML["modelo"]} '
          f'F1={BASELINE_AUTOML["f1_macro"]} | AUC={BASELINE_AUTOML["auc_roc"]}')
    print(f'  Algoritmos:      {total} en total')
    for modulo in MODULOS_FASE5[:-1]:  # Excluye comparativa
        print(f'    {modulo["emoji"]} {modulo["nombre"]}: '
              f'{modulo["n_algoritmos"]} modelos')
    print('=' * 65)
