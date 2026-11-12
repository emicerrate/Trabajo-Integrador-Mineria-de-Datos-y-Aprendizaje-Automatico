import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def cargar_y_procesar_abalone(ruta_archivo):
  """Función de Extract, Load y Transform (ELT) para el dataset Abalone."""
  print("-- 1. EXTRACCION Y CARGA DE DATOS --")

  # Nombres estandar de columnas para el dataset Abalone
  columnas = [
      "Sex",
      "Length",
      "Diameter",
      "Height",
      "Whole_weight",
      "Shucked_weight",
      "Viscera_weight",
      "Shell_weight",
      "Rings",
  ]

  # na_values=['None', '?', 'NA'] para atrapar los valores nulos textuales del dataset
  df = pd.read_csv(
      ruta_archivo, header=None, names=columnas, na_values=["None", "?", "NA", ""]
  )

  print(f"Dimensiones originales del dataset: {df.shape}")
  print("\nValores nulos por columna antes de limpiar:")
  print(df.isnull().sum())

  print("\n--- 2. TRANSFORMACION Y LIMPIEZA ---")

  # Tratamiento de valores nulos (Imputacion con la mediana para numéricos o eliminación)
  cols_numericas = [
      "Length",
      "Diameter",
      "Height",
      "Whole_weight",
      "Shucked_weight",
      "Viscera_weight",
      "Shell_weight",
  ]

  for col in cols_numericas:
    mediana_val = df[col].median()
    df[col] = df[col].fillna(mediana_val)

  # Para la columna categórica 'Sex', rellenamos con la moda
  df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])

  # Agrupar Rings en tres categorías (Definida correctamente dentro o bien indentada)
  def categorizar_rings(rings):
    if 1 <= rings <= 10:
      return "Joven"
    elif 11 <= rings <= 20:
      return "Medio"
    elif 21 <= rings <= 29:
      return "Viejo"
    return "Desconocido"  # Por seguridad si hay algún valor fuera de rango

  df["Rings"] = df["Rings"].apply(categorizar_rings)

  print(f"Valores nulos despues de la limpieza: {df.isnull().sum().sum()}")

  # Conversión de variable categorica a numérica (One-Hot Encoding) para la vista minable
  df_minable = pd.get_dummies(df, columns=["Sex"], prefix="Sex", drop_first=False)

  print("\nVista minable generada exitosamente (Primeras filas):")
  print(df_minable.head())

  # --- GUARDAR EL DATASET PROCESADO  ---
  os.makedirs("Abalone/processed", exist_ok=True)
  ruta_salida = "Abalone/processed/abalone_minable.csv"
  df_minable.to_csv(ruta_salida, index=False)
  print(f"Dataset minable guardado exitosamente en: {ruta_salida}")

  return df, df_minable


def realizar_eda(df):
  """Genera gráficos y estadísticas descriptivas para cumplir con el EDA del Inciso 1."""
  print("\n--- 3. ANALISIS EXPLORATORIO DE DATOS (EDA) ---")

  print("\nEstadisticas descriptivas:")
  print(df.describe())

  # Crear directorio para guardar gráficos si no existe
  os.makedirs("output_images", exist_ok=True)

  # Gráfico 1: Distribución de la variable categórica Rings (ahora son etiquetas: Joven, Medio, Viejo)
  plt.figure(figsize=(8, 5))
  sns.countplot(
      x="Rings",
      data=df,
      order=["Joven", "Medio", "Viejo"],
      palette="Set2",
      hue="Rings",
      legend=False,
  )
  plt.title("Distribucion de Categorias de Edad (Rings) en Abalones")
  plt.xlabel("Categoria de Anillos")
  plt.ylabel("Cantidad")
  plt.savefig("output_images/distribucion_rings.png")
  plt.close()

  # Gráfico 2: Matriz de correlación SOLO con las variables numéricas reales
  plt.figure(figsize=(10, 8))
  # Seleccionamos únicamente las columnas que son numéricas en el DataFrame
  df_numerico = df.select_dtypes(include=[np.number])
  matriz_corr = df_numerico.corr()

  sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
  plt.title("Matriz de Correlacion de Variables Numericas - Abalone")
  plt.savefig("output_images/matriz_correlacion.png")
  plt.close()

  print("\nGraficos del EDA guardados en la carpeta 'output_images/'.")
  
# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
  ruta_datos = "Abalone/Datos.txt"

  if os.path.exists(ruta_datos):
    df_original, df_minable = cargar_y_procesar_abalone(ruta_datos)
    realizar_eda(df_original)
  else:
    print(
        f"No se encontro el archivo en la ruta: {ruta_datos}. Verifica el"
        " nombre de la carpeta y archivo."
    )