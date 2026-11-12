"""Configuración compartida del TP integrador (dataset Abalone).

Acá viven las rutas, la validación cruzada, las métricas y las 10 configuraciones
de modelos de la Tarea 2. La Tarea 3 (stacking) importa `construir_modelos()` desde
este módulo para usar exactamente los mismos 10 modelos débiles, sin duplicar código
y sin reutilizar los .pkl entrenados con todo el dataset (eso filtraría información
dentro de la cross-validation).
"""

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- Rutas (relativas a la raíz del repo, funcionan desde cualquier directorio) ---
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "Abalone" / "Datos.txt"
DIR_PROCESSED = RAIZ / "Abalone" / "processed"
DIR_MODELOS = RAIZ / "Abalone" / "models"
DIR_IMAGENES = RAIZ / "output_images"
RUTA_MINABLE = DIR_PROCESSED / "abalone_minable.csv"

# --- Dataset ---
COLUMNAS = [
    "Sex", "Length", "Diameter", "Height", "Whole_weight",
    "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings",
]
COLS_NUMERICAS = COLUMNAS[1:-1]  # las 7 medidas físicas (sin Sex ni Rings)
VALORES_NULOS = ["None", "?", "NA", ""]
ORDEN_CLASES = ["Joven", "Medio", "Viejo"]

# --- Validación cruzada ---
# Estratificada y con mezcla: la clase "Viejo" tiene muy pocos casos (menos del 1 %),
# y sin estratificar/mezclar un fold podría quedar casi sin ejemplos de esa clase.
SEMILLA = 42
N_FOLDS = 5


def get_cv():
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)


# --- Métricas ---
# Las 4 métricas de clase más dos de diagnóstico del desbalance:
# balanced_accuracy y el recall de cada clase por separado.
def _recall_clase(clase):
    return make_scorer(recall_score, labels=[clase], average="macro", zero_division=0)


SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    **{f"recall_{c}": _recall_clase(c) for c in ORDEN_CLASES},
}

# --- Tarea 2: 5 configuraciones por técnica ---
# Las configuraciones 1-3 no ponderan las clases; las 4 y 5 usan class_weight="balanced"
# para compensar el desbalance (Viejo < 1 %). Sin ponderar, casi ningún modelo predice
# "Viejo" (recall 0). Para volver al esquema anterior, quitá "class_weight" de las confs 4 y 5.
TREE_CONFIGS = [
    {"criterion": "gini", "max_depth": 5, "min_samples_split": 100, "min_samples_leaf": 40, "max_leaf_nodes": 6, "random_state": SEMILLA},
    {"criterion": "entropy", "max_depth": 5, "min_samples_split": 100, "min_samples_leaf": 40, "max_leaf_nodes": 6, "random_state": SEMILLA},
    {"criterion": "log_loss", "max_depth": 8, "min_samples_split": 80, "min_samples_leaf": 30, "max_leaf_nodes": 10, "random_state": SEMILLA},
    {"criterion": "gini", "max_depth": 4, "min_samples_split": 120, "min_samples_leaf": 50, "max_leaf_nodes": 5, "class_weight": "balanced", "random_state": SEMILLA},
    {"criterion": "entropy", "max_depth": None, "min_samples_split": 90, "min_samples_leaf": 35, "max_leaf_nodes": 8, "class_weight": "balanced", "random_state": SEMILLA},
]

LOGREG_CONFIGS = [
    {"C": 0.1, "solver": "lbfgs", "max_iter": 1000},
    {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    {"C": 10.0, "solver": "lbfgs", "max_iter": 1000},
    {"C": 1.0, "solver": "saga", "penalty": "l1", "max_iter": 2000, "class_weight": "balanced"},
    {"C": 0.5, "solver": "newton-cg", "max_iter": 1000, "class_weight": "balanced"},
]


def construir_arbol(config):
    return DecisionTreeClassifier(**config)


def construir_logreg(config):
    # El Pipeline estandariza dentro de cada fold (sin filtrar información del fold de test).
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(**config)),
    ])


def construir_modelos():
    """Devuelve los 10 modelos como lista de (nombre, estimador) sin entrenar.

    El formato (nombre, estimador) es el que pide StackingClassifier(estimators=...).
    """
    modelos = []
    for i, cfg in enumerate(TREE_CONFIGS, start=1):
        modelos.append((f"DecisionTree_Conf_{i}", construir_arbol(cfg)))
    for i, cfg in enumerate(LOGREG_CONFIGS, start=1):
        modelos.append((f"LogReg_Conf_{i}", construir_logreg(cfg)))
    return modelos
