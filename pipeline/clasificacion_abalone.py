"""Tarea 2: Árbol de decisión vs Regresión logística (5 ejecuciones c/u, CV estratificada).

Qué hace, en orden:
    1. Carga la vista minable generada por etl_abalone.py.
    2. Evalúa las 5 configuraciones de cada técnica con cross-validation (2a).
    3. Promedia las métricas de las 5 ejecuciones de cada técnica (2b).
    4. Compara ambas técnicas (2c): tabla, gráfico y matrices de confusión.
"""

import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_validate

from config import (
    DIR_IMAGENES, DIR_MODELOS, DIR_PROCESSED, LOGREG_CONFIGS, ORDEN_CLASES,
    RUTA_MINABLE, SCORING, TREE_CONFIGS, construir_arbol, construir_logreg, get_cv,
)

sns.set_theme(style="whitegrid")

# Nombre que se muestra, prefijo del .pkl, configuraciones y constructor de cada técnica
TECNICAS = [
    ("DecisionTree", "DecisionTree", TREE_CONFIGS, construir_arbol),
    ("LogisticRegression", "LogReg", LOGREG_CONFIGS, construir_logreg),
]

COLUMNAS_RESUMEN = {
    "test_accuracy": "Accuracy",
    "test_precision_macro": "Precision",
    "test_recall_macro": "Recall",
    "test_f1_macro": "F1-Score",
    "test_balanced_accuracy": "Balanced_Acc",
    "test_recall_Joven": "Recall_Joven",
    "test_recall_Medio": "Recall_Medio",
    "test_recall_Viejo": "Recall_Viejo",
}
METRICAS = list(COLUMNAS_RESUMEN.values())


def evaluar_ejecuciones(X, y):
    """2a: cross-validation de las 10 ejecuciones (5 por técnica)."""
    filas = []
    modelos_por_corrida = {}
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)

    for algoritmo, prefijo, configs, constructor in TECNICAS:
        print(f"Evaluando {algoritmo}...")
        for i, cfg in enumerate(configs, start=1):
            modelo = constructor(cfg)
            scores = cross_validate(modelo, X, y, cv=get_cv(), scoring=SCORING)

            fila = {
                "Algoritmo": algoritmo,
                "Configuración": f"Conf {i}",
                "class_weight": cfg.get("class_weight", "ninguno"),
            }
            fila.update({nombre: scores[k].mean() for k, nombre in COLUMNAS_RESUMEN.items()})
            filas.append(fila)
            modelos_por_corrida[(algoritmo, f"Conf {i}")] = constructor(cfg)

            # Se guarda el modelo entrenado con todos los datos (uso final/demo).
            # OJO: para el stacking NO usar estos .pkl; usar config.construir_modelos().
            modelo.fit(X, y)
            joblib.dump(modelo, DIR_MODELOS / f"{prefijo}_Conf_{i}.pkl")
            print(f"  [{prefijo}_Conf_{i}] F1={fila['F1-Score']:.3f}  Recall_Viejo={fila['Recall_Viejo']:.2f}  guardado.")

    return pd.DataFrame(filas), modelos_por_corrida


def promediar(df_resultados):
    """2b: promedio (y desvío) de las 5 ejecuciones de cada técnica."""
    g = df_resultados.groupby("Algoritmo")[METRICAS]
    media, desvio = g.mean(), g.std()
    resumen = media.join(desvio.add_suffix("_std"))
    resumen.insert(0, "n_ejecuciones", df_resultados.groupby("Algoritmo").size())
    return resumen


def graficar_comparacion(resumen):
    datos = resumen[["Accuracy", "Precision", "Recall", "F1-Score", "Recall_Viejo"]].reset_index()
    largo = datos.melt(id_vars="Algoritmo", var_name="Metrica", value_name="Valor")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=largo, x="Metrica", y="Valor", hue="Algoritmo", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Tarea 2: metricas promedio de las 5 ejecuciones por tecnica")
    plt.tight_layout()
    plt.savefig(DIR_IMAGENES / "comparacion_punto2.png", dpi=120)
    plt.close()


def graficar_matrices_confusion(df_resultados, modelos_por_corrida, X, y):
    """Matriz de confusión (CV, normalizada por fila) del mejor F1 de cada técnica."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (algoritmo, _, _, _) in zip(axes, TECNICAS):
        sub = df_resultados[df_resultados["Algoritmo"] == algoritmo]
        mejor = sub.loc[sub["F1-Score"].idxmax(), "Configuración"]
        pred = cross_val_predict(modelos_por_corrida[(algoritmo, mejor)], X, y, cv=get_cv())
        cm = confusion_matrix(y, pred, labels=ORDEN_CLASES, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=ORDEN_CLASES, yticklabels=ORDEN_CLASES, ax=ax, cbar=False)
        ax.set_title(f"{algoritmo} - {mejor}")
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Real")
    plt.suptitle("Matrices de confusion (cross-validation, normalizadas por clase real)")
    plt.tight_layout()
    plt.savefig(DIR_IMAGENES / "matrices_confusion_punto2.png", dpi=120)
    plt.close()


def entrenar_y_evaluar():
    print("--- CARGA DE DATOS ---")
    if not RUTA_MINABLE.exists():
        print(f"Error: No se encontró {RUTA_MINABLE}. Correr el ETL primero.")
        return
    df = pd.read_csv(RUTA_MINABLE)
    X, y = df.drop(columns=["Rings"]), df["Rings"]
    print(f"Dataset cargado. Features: {X.shape[1]}, Registros: {X.shape[0]}")
    print("Distribucion de clases:", y.value_counts().reindex(ORDEN_CLASES).to_dict())

    print("\n--- 2a. ENTRENAMIENTO Y CROSS-VALIDATION (5 folds estratificados) ---")
    df_resultados, modelos = evaluar_ejecuciones(X, y)
    df_resultados = df_resultados.sort_values("F1-Score", ascending=False)
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(DIR_PROCESSED / "resultados_punto2.csv", index=False)

    print("\nResultados de las 10 ejecuciones (ordenados por F1-Score):")
    print(df_resultados.round(3).to_string(index=False))

    print("\n--- 2b. PROMEDIO DE LAS 5 EJECUCIONES POR TECNICA ---")
    resumen = promediar(df_resultados)
    resumen.round(4).to_csv(DIR_PROCESSED / "resultados_punto2_promedio.csv")
    print(resumen[["n_ejecuciones"] + METRICAS].round(3).to_string())
    print("\nDesvio estandar entre las 5 ejecuciones:")
    print(resumen[[m + "_std" for m in METRICAS]].round(3).to_string())

    # Mismo promedio, separado por régimen de ponderación de clases
    por_cw = df_resultados.groupby(["Algoritmo", "class_weight"])[METRICAS].mean()
    por_cw.round(4).to_csv(DIR_PROCESSED / "resultados_punto2_por_class_weight.csv")
    print("\nPromedio segun class_weight (ninguno = confs 1-3, balanced = confs 4-5):")
    print(por_cw.round(3).to_string())

    print("\n--- 2c. COMPARACION ENTRE TECNICAS ---")
    for m in ["Accuracy", "Precision", "Recall", "F1-Score", "Recall_Viejo"]:
        mejor = resumen[m].idxmax()
        dif = resumen[m].max() - resumen[m].min()
        print(f"  {m:<13} mejor: {mejor:<20} (diferencia {dif:+.3f})")

    graficar_comparacion(resumen)
    graficar_matrices_confusion(df_resultados, modelos, X, y)
    print("\nGuardado: resultados_punto2*.csv en Abalone/processed/, graficos en output_images/.")


if __name__ == "__main__":
    entrenar_y_evaluar()
