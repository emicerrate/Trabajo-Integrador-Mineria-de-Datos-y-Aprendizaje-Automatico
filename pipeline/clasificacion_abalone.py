import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold

def entrenar_y_evaluar():
    print("--- CARGA DE DATOS ---")
    ruta_csv = "Abalone/processed/abalone_minable.csv"
    
    if not os.path.exists(ruta_csv):
        print(f"Error: No se encontró {ruta_csv}. Correr el ETL primero.")
        return
        
    df = pd.read_csv(ruta_csv)
    
    X = df.drop(columns=['Rings'])
    y = df['Rings']
    
    print(f"Dataset cargado. Features: {X.shape[1]}, Registros: {X.shape[0]}")

    print("\n--- CONFIGURACIÓN DE MODELOS E HIPERPARÁMETROS ---")
    
    tree_configs = [
        {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 100, 'min_samples_leaf': 40, 'max_leaf_nodes': 6, 'random_state': 42},
        {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 100, 'min_samples_leaf': 40, 'max_leaf_nodes': 6, 'random_state': 42},
        {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 80, 'min_samples_leaf': 30, 'max_leaf_nodes': 10, 'random_state': 42},
        {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 120, 'min_samples_leaf': 50, 'max_leaf_nodes': 5, 'random_state': 42},
        {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 90, 'min_samples_leaf': 35, 'max_leaf_nodes': 8, 'random_state': 42}
    ]

    logreg_configs = [
        {'C': 0.1, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 10.0, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 1.0, 'solver': 'saga', 'penalty': 'l1', 'max_iter': 2000, "random_state": 42},
        {'C': 0.5, 'solver': 'newton-cg', 'max_iter': 1000}
    ]

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_externo = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados_totales = []
    os.makedirs("Abalone/models", exist_ok=True)

    print("\n--- ENTRENAMIENTO Y CROSS-VALIDATION ---")
    
    # Evaluación Árboles de Decisión
    print("Evaluando DecisionTree...")
    for i, config in enumerate(tree_configs):
        modelo = DecisionTreeClassifier(**config)
        nombre_ejecucion = f"DecisionTree_Conf_{i+1}"
        
        scores = cross_validate(modelo, X, y, cv=cv_externo, scoring=scoring)
        
        resultados_totales.append({
            'Algoritmo': 'DecisionTree',
            'Configuración': f"Conf {i+1}",
            'Accuracy': np.mean(scores['test_accuracy']),
            'Precision': np.mean(scores['test_precision_macro']),
            'Recall': np.mean(scores['test_recall_macro']),
            'F1-Score': np.mean(scores['test_f1_macro'])
        })
        
        modelo.fit(X, y)
        joblib.dump(modelo, f"Abalone/models/{nombre_ejecucion}.pkl")
        print(f"[{nombre_ejecucion}] guardado.")

    # Evaluación Regresión Logística (Con StandardScaler en un Pipeline)
    print("\nEvaluando Regresión Logística (con StandardScaler)...")
    for i, config in enumerate(logreg_configs):
        # El Pipeline asegura que la estandarización se aplique y se guarde con el modelo
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(**config))
        ])
        nombre_ejecucion = f"LogReg_Conf_{i+1}"
        
        scores = cross_validate(modelo, X, y, cv=cv_externo, scoring=scoring)
        
        resultados_totales.append({
            'Algoritmo': 'LogisticRegression',
            'Configuración': f"Conf {i+1}",
            'Accuracy': np.mean(scores['test_accuracy']),
            'Precision': np.mean(scores['test_precision_macro']),
            'Recall': np.mean(scores['test_recall_macro']),
            'F1-Score': np.mean(scores['test_f1_macro'])
        })
        
        modelo.fit(X, y)
        joblib.dump(modelo, f"Abalone/models/{nombre_ejecucion}.pkl")
        print(f"[{nombre_ejecucion}] guardado.")

    print("\n--- COMPARACIÓN DE RESULTADOS ---")
    df_resultados = pd.DataFrame(resultados_totales)
    df_resultados = df_resultados.sort_values(by='F1-Score', ascending=False)
    
    print("\nResultados ordenados por mejor F1-Score:")
    print(df_resultados.to_string(index=False))
    
    ruta_resultados = "Abalone/processed/resultados_punto2.csv"
    df_resultados.to_csv(ruta_resultados, index=False)
    print(f"\nCSV generado en: {ruta_resultados}")

if __name__ == '__main__':
    entrenar_y_evaluar()