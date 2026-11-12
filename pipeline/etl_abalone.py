"""ETL + EDA del dataset Abalone (Tarea 1 y preparación del dataset).

Orden de ejecución dentro del script:
    1. Carga de datos crudos.
    2. EDA detallado sobre los datos originales (con nulos y Rings numérico).
    3. Comparación de métodos de imputación (justifica el criterio elegido).
    4. Imputación de nulos con el método ganador.
    5. Agrupación de Rings en Joven / Medio / Viejo.
    6. One-hot de Sex y guardado de la vista minable.
"""

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guarda las imágenes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

from config import (
    COLS_NUMERICAS, COLUMNAS, DIR_IMAGENES, DIR_PROCESSED, ORDEN_CLASES,
    RUTA_DATOS, RUTA_MINABLE, SEMILLA, VALORES_NULOS,
)

sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------------
# 1. Carga
# ----------------------------------------------------------------------------
def cargar_datos(ruta):
    print("-- 1. EXTRACCION Y CARGA DE DATOS --")
    df = pd.read_csv(ruta, header=None, names=COLUMNAS, na_values=VALORES_NULOS)
    print(f"Dimensiones originales del dataset: {df.shape}")
    return df


# ----------------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------------
def _guardar(nombre):
    DIR_IMAGENES.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(DIR_IMAGENES / nombre, dpi=120)
    plt.close()


def realizar_eda(df):
    """EDA detallado de cada variable, sobre los datos originales (sin imputar)."""
    print("\n--- 2. ANALISIS EXPLORATORIO DE DATOS (EDA) ---")
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

    # --- 2.1 Tipos y nulos ---
    print("\nTipos de dato:")
    print(df.dtypes.to_string())
    nulos = pd.DataFrame({
        "nulos": df.isnull().sum(),
        "porcentaje": (df.isnull().mean() * 100).round(2),
    })
    print("\nValores nulos por columna:")
    print(nulos.to_string())
    nulos.to_csv(DIR_PROCESSED / "eda_nulos.csv")

    # --- 2.2 Variable categórica Sex ---
    print("\nSex (frecuencias):")
    print(df["Sex"].value_counts().to_string())

    # --- 2.3 Variables numéricas: estadísticas, asimetría, ceros y outliers (IQR) ---
    cols = COLS_NUMERICAS + ["Rings"]
    desc = df[cols].describe().T
    desc["asimetria"] = df[cols].skew()
    desc["ceros"] = (df[cols] == 0).sum()
    q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
    iqr = q3 - q1
    fuera = (df[cols] < q1 - 1.5 * iqr) | (df[cols] > q3 + 1.5 * iqr)
    desc["outliers_IQR"] = fuera.sum()
    print("\nEstadisticas descriptivas (numericas):")
    print(desc.round(3).to_string())
    desc.round(4).to_csv(DIR_PROCESSED / "eda_estadisticas.csv")

    # --- 2.4 Rings: valores distintos y categorías resultantes ---
    print(f"\nRings: {df['Rings'].nunique()} valores distintos, rango {df['Rings'].min()}..{df['Rings'].max()}")
    cat = pd.cut(df["Rings"], bins=[0, 10, 20, 29], labels=ORDEN_CLASES)
    dist = cat.value_counts().reindex(ORDEN_CLASES)
    print("\nDistribucion de las categorias (Joven 1-10, Medio 11-20, Viejo 21-29):")
    print(pd.DataFrame({"cantidad": dist, "porcentaje": (dist / len(df) * 100).round(2)}).to_string())

    # --- 2.5 Medidas por Sex (justifica mirar Sex al imputar) ---
    print("\nMediana de cada variable segun Sex:")
    print(df.groupby("Sex")[cols].median().round(3).to_string())

    # --- 2.6 Gráficos ---
    # Valores nulos
    plt.figure(figsize=(8, 4))
    n = nulos["nulos"].sort_values(ascending=False)
    sns.barplot(x=n.values, y=n.index, color="steelblue")
    plt.title("Valores nulos por columna")
    plt.xlabel("Cantidad de nulos")
    _guardar("nulos_por_columna.png")

    # Histogramas de las 7 numéricas + Rings
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(df[c], kde=True, ax=ax, color="steelblue")
        ax.set_title(c)
        ax.set_xlabel("")
    plt.suptitle("Distribucion de cada variable numerica")
    _guardar("histogramas_variables.png")

    # Boxplots por Sex (outliers y diferencias entre grupos)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x="Sex", y=c, order=["F", "M", "I"], hue="Sex", palette="Set2", legend=False, ax=ax)
        ax.set_title(c)
        ax.set_xlabel("")
    plt.suptitle("Variables numericas segun Sex")
    _guardar("boxplots_por_sexo.png")

    # Rings original (28 valores) con los cortes de las categorías
    plt.figure(figsize=(9, 5))
    sns.countplot(x="Rings", data=df, color="steelblue")
    plt.axvline(9.5, color="red", linestyle="--")   # corte Joven | Medio (10 | 11)
    plt.axvline(18.5, color="red", linestyle="--")  # corte Medio | Viejo (20 | 21)
    plt.title("Distribucion de Rings (lineas rojas: cortes de categorias)")
    _guardar("distribucion_rings_original.png")

    # Sex vs categoría
    plt.figure(figsize=(7, 5))
    sns.countplot(
        x=cat.astype(str), hue=df["Sex"], order=ORDEN_CLASES, hue_order=["F", "M", "I"], palette="Set2"
    )
    plt.title("Categoria de edad segun Sex")
    plt.xlabel("Categoria")
    plt.ylabel("Cantidad")
    _guardar("sex_vs_categoria.png")

    # Matriz de correlación con Rings numérico incluido
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlacion - Abalone (incluye Rings numerico)")
    _guardar("matriz_correlacion.png")

    print(f"\nTablas del EDA guardadas en '{DIR_PROCESSED.name}/' y graficos en '{DIR_IMAGENES.name}/'.")


# ----------------------------------------------------------------------------
# 3. Imputación
# ----------------------------------------------------------------------------
def _matriz_imputacion(df):
    """Numéricas + Sex en one-hot. Rings queda afuera a propósito: es el objetivo
    y usarlo para imputar filtraría información de la clase a predecir."""
    sex = pd.get_dummies(df["Sex"], prefix="Sex", dtype=float)
    return pd.concat([df[COLS_NUMERICAS], sex], axis=1)


def _imputador(matriz):
    return IterativeImputer(
        estimator=LinearRegression(), max_iter=10, random_state=SEMILLA,
        min_value=matriz.min().values, max_value=matriz.max().values,  # nunca imputa fuera del rango observado
    )


def comparar_imputacion(df, n_mascara=300):
    """Enmascara valores conocidos y mide qué tan bien los reconstruye cada método.

    Compara: mediana global, mediana por Sex y regresión lineal sobre el resto de
    las medidas (IterativeImputer). Esto sustenta el criterio de imputación elegido.
    """
    print("\n--- 3. COMPARACION DE METODOS DE IMPUTACION ---")
    rng = np.random.default_rng(SEMILLA)
    completas = df.dropna(subset=COLS_NUMERICAS).reset_index(drop=True)
    matriz = _matriz_imputacion(completas)
    filas = []
    for col in COLS_NUMERICAS:
        idx = rng.choice(completas.index, size=n_mascara, replace=False)
        real = completas.loc[idx, col]
        resto = completas.drop(index=idx)

        mae_global = (real - resto[col].median()).abs().mean()
        med_sex = resto.groupby("Sex")[col].median()
        mae_sex = (real - completas.loc[idx, "Sex"].map(med_sex)).abs().mean()

        enmascarada = matriz.copy()
        enmascarada.loc[idx, col] = np.nan
        imp = _imputador(matriz)
        pred = pd.DataFrame(imp.fit_transform(enmascarada), columns=matriz.columns).loc[idx, col]
        mae_reg = (real - pred).abs().mean()

        filas.append({
            "variable": col,
            "nulos_reales": int(df[col].isnull().sum()),
            "MAE_mediana_global": mae_global,
            "MAE_mediana_por_Sex": mae_sex,
            "MAE_regresion": mae_reg,
            "mejora_vs_global_x": mae_global / mae_reg,
        })
    tabla = pd.DataFrame(filas).round(4)
    print(tabla.to_string(index=False))
    tabla.to_csv(DIR_PROCESSED / "eda_comparacion_imputacion.csv", index=False)
    return tabla


def imputar_nulos(df):
    """Imputa los nulos numéricos con regresión lineal sobre el resto de las medidas.

    Criterio elegido a partir del EDA: las medidas están muy correlacionadas entre
    sí (0.8-0.99) y difieren fuertemente según Sex (los infantes son más chicos),
    así que reconstruir cada valor desde las demás medidas es mucho más preciso que
    una mediana global (ver `comparar_imputacion`). Sex no tiene nulos; si los
    hubiera se rellenaría con la moda.
    """
    print("\n--- 4. IMPUTACION DE NULOS ---")
    df = df.copy()
    df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])

    matriz = _matriz_imputacion(df)
    imp = _imputador(matriz)
    imputada = pd.DataFrame(imp.fit_transform(matriz), columns=matriz.columns, index=df.index)
    df[COLS_NUMERICAS] = imputada[COLS_NUMERICAS]

    print(f"Valores nulos despues de la imputacion: {int(df.isnull().sum().sum())}")
    return df


# ----------------------------------------------------------------------------
# 5. Rings -> categorías
# ----------------------------------------------------------------------------
def categorizar_rings(df):
    df = df.copy()
    df["Rings"] = pd.cut(df["Rings"], bins=[0, 10, 20, 29], labels=ORDEN_CLASES).astype(str)
    assert df["Rings"].isin(ORDEN_CLASES).all(), "Hay valores de Rings fuera del rango 1..29"
    return df


# ----------------------------------------------------------------------------
# Ejecución principal
# ----------------------------------------------------------------------------
def main():
    if not RUTA_DATOS.exists():
        print(f"No se encontro el archivo en la ruta: {RUTA_DATOS}")
        return

    df = cargar_datos(RUTA_DATOS)
    realizar_eda(df)
    comparar_imputacion(df)
    df = imputar_nulos(df)

    print("\n--- 5. AGRUPACION DE RINGS ---")
    df = categorizar_rings(df)
    print(df["Rings"].value_counts().reindex(ORDEN_CLASES).to_string())

    plt.figure(figsize=(8, 5))
    sns.countplot(x="Rings", data=df, order=ORDEN_CLASES, hue="Rings", palette="Set2", legend=False)
    plt.title("Distribucion de Categorias de Edad (Rings) en Abalones")
    plt.xlabel("Categoria de Anillos")
    plt.ylabel("Cantidad")
    _guardar("distribucion_rings.png")

    print("\n--- 6. VISTA MINABLE ---")
    df_minable = pd.get_dummies(df, columns=["Sex"], prefix="Sex", drop_first=False)
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_minable.to_csv(RUTA_MINABLE, index=False)
    print(df_minable.head().to_string())
    print(f"\nDataset minable guardado en: {RUTA_MINABLE}")


if __name__ == "__main__":
    main()
