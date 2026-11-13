# Etapa 3 - Stacking

import os
import warnings
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold

# Hacemos que no nos imprima los warnings
warnings.filterwarnings("ignore")

# INCISO A
# Tomo como estimadores base los modelos entrenados en la Etapa 2
estimadores_base = [
    ('DecisionTree_1', DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=100, min_samples_leaf=40, max_leaf_nodes=6, random_state=42)),
    ('DecisionTree_2', DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=100, min_samples_leaf=40, max_leaf_nodes=6, random_state=42)),
    ('DecisionTree_3', DecisionTreeClassifier(criterion='log_loss', max_depth=8, min_samples_split=80, min_samples_leaf=30, max_leaf_nodes=10, random_state=42)),
    ('DecisionTree_4', DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=120, min_samples_leaf=50, max_leaf_nodes=5, random_state=42)),
    ('DecisionTree_5', DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=90, min_samples_leaf=35, max_leaf_nodes=8, random_state=42)),
    ('LogReg_1', Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000))])),
    ('LogReg_2', Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000))])),
    ('LogReg_3', Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(C=10.0, solver='lbfgs', max_iter=1000))])),
    ('LogReg_4', Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(C=1.0, solver='saga', penalty='l1', max_iter=2000))])),
    ('LogReg_5', Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(C=0.5, solver='newton-cg', max_iter=1000))]))
]

df = pd.read_csv("Abalone/processed/abalone_minable.csv")
# Axis = 1 es para marcar que es una columna
X = df.drop("Rings", axis=1)
T = df["Rings"]

# Hiperparámetros del meta-modelo, las usamos para que nuestro meta-estimador pruebe distintas intensidades de regularización
configuraciones = [
    {'C': 0.01, 'max_iter': 2000},
    {'C': 0.1,  'max_iter': 2000},
    {'C': 1.0,  'max_iter': 2000},
    {'C': 10.0, 'max_iter': 2000},
    {'C': 100.0,'max_iter': 2000}
]

# Configuramos CV-externa, que es la que separa para el stacking el train y el test manteniendo proporción de clases
cv_externo = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Realizamos las 5 ejecuciones del Stacking
resultados = []
for i, config in enumerate(configuraciones):
    print(f"\n--- Stacking: Configuración {i+1} ---")

    # Acá cv=5 hace la CV-interna (out-of-fold prediction)
    stack = StackingClassifier(estimators=estimadores_base, final_estimator=LogisticRegression(**config), cv=5, stack_method="predict_proba")
    scores = cross_validate(stack, X, T, cv=cv_externo, scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"])
    accuracy_promedio = scores['test_accuracy'].mean()
    precision_promedio = scores['test_precision_macro'].mean()
    recall_promedio = scores['test_recall_macro'].mean()
    f1_promedio = scores['test_f1_macro'].mean()

    # Guardo el resultado para esta configuración y los muestro
    resultados.append({"Configuracion": i, "C": config["C"], "Accuracy": accuracy_promedio, "Precision_Macro": precision_promedio, "Recall_Macro": recall_promedio, "F1_Macro": f1_promedio})
    print(f"Accuracy promedio: {accuracy_promedio:.4f}")
    print(f"Precision Macro promedio: {precision_promedio:.4f}")
    print(f"Recall Macro promedio: {recall_promedio:.4f}")
    print(f"F1 Macro promedio: {f1_promedio:.4f}")

# Creamos un Dataframe con los resultados para ordenarlos y guardarlos en un CSV
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values(by="F1_Macro", ascending=False)
os.makedirs("Abalone/processed", exist_ok=True)
df_resultados.to_csv("Abalone/processed/resultados_stacking.csv", index=False)

# Mostramos los resultados finales
print("\n--- RESULTADOS FINALES DEL STACKING ---")
print(df_resultados.to_string(index=False))


# INCISO B
# Calculamos el promedio para cada métrica de todas las ejecuciones independientes
promedios_totales = df_resultados[["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro"]].mean()
print("\n--- PROMEDIO GENERAL DE LAS CINCO EJECUCIONES ---")
print(promedios_totales)
