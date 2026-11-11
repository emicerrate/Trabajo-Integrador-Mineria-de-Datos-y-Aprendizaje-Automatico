import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

def cargar_y_procesar_abalone(ruta_archivo):
    """
    Función de Extract, Load y Transform (ELT) para el dataset Abalone.
    """
    print("-- 1. EXTRACCION Y CARGA DE DATOS --")
    
    # Nombres estandar de columnas para el dataset Abalone
    columnas = [
        'Sex', 'Length', 'Diameter', 'Height', 
        'Whole_weight', 'Shucked_weight', 'Viscera_weight', 
        'Shell_weight', 'Rings'
    ]
    
    # na_values=['None', '?', 'NA'] para atrapar los valores nulos textuales del dataset
    df = pd.read_csv(ruta_archivo, header=None, names=columnas, na_values=['None', '?', 'NA', ''])
    
    print(f"Dimensiones originales del dataset: {df.shape}")
    print("\nValores nulos por columna antes de limpiar:")
    print(df.isnull().sum())
    
    print("\n--- 2. TRANSFORMACION Y LIMPIEZA ---")
    
    # Tratamiento de valores nulos (Imputacion con la mediana para numéricos o eliminación)
    # En este caso, rellenamos las variables numéricas con la mediana de su columna
    cols_numericas = ['Length', 'Diameter', 'Height', 'Whole_weight', 
                      'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    
    for col in cols_numericas:
        mediana_val = df[col].median()
        df[col] = df[col].fillna(mediana_val)
        
    # Para la columna categórica 'Sex', podemos rellenar con la moda o una categoría 'Unknown'
    df['Sex'] = df['Sex'].fillna(df['Sex'].mode()[0])
    
    print(f"Valores nulos despues de la limpieza: {df.isnull().sum().sum()}")
    
    # Conversión de variable categorica a numérica (One-Hot Encoding) para la vista minable
    df_minable = pd.get_dummies(df, columns=['Sex'], prefix='Sex', drop_first=False)
    
    print("\nVista minable generada exitosamente (Primeras filas):")
    print(df_minable.head())  # Reemplazamos 'display(...)' por 'print(...)'
    
    # --- GUARDAR EL DATASET PROCESADO  ---
    os.makedirs('Abalone/processed', exist_ok=True)
    ruta_salida = 'Abalone/processed/abalone_minable.csv'
    df_minable.to_csv(ruta_salida, index=False)
    print(f"Dataset minable guardado exitosamente en: {ruta_salida}")
    
    return df, df_minable
    
def realizar_eda(df):
    """
    Genera gráficos y estadísticas descriptivas para cumplir con el EDA del Inciso 1.
    """
    print("\n--- 3. ANALISIS EXPLORATORIO DE DATOS (EDA) ---")
    
    print("\nEstadisticas descriptivas:")
    print(df.describe())
    
    # Crear directorio para guardar gráficos si no existe
    os.makedirs('output_images', exist_ok=True)
    
    # Gráfico 1: Distribución de la variable objetivo (Rings / Edad)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Rings'], kde=True, color='teal', bins=25)
    plt.title('Distribucion de Anillos (Rings) en Abalones')
    plt.xlabel('Cantidad de Anillos')
    plt.ylabel('Frecuencia')
    plt.savefig('output_images/distribucion_rings.png')
    plt.close()
    
    # Gráfico 2: Matriz de correlación entre variables físicas y anillos
    plt.figure(figsize=(10, 8))
    cols_corr = ['Length', 'Diameter', 'Height', 'Whole_weight', 
                 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
    matriz_corr = df[cols_corr].corr()
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlacion de Variables - Abalone')
    plt.savefig('output_images/matriz_correlacion.png')
    plt.close()
    
    print("\nGraficos del EDA guardados en la carpeta 'output_images/'.")

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # Ajustá la ruta según donde tengas tu archivo de datos dentro de la carpeta Abalone
    ruta_datos = "Abalone/Datos.txt" 
    
    if os.path.exists(ruta_datos):
        df_original, df_minable = cargar_y_procesar_abalone(ruta_datos)
        realizar_eda(df_original)
    else:
        print(f"No se encontro el archivo en la ruta: {ruta_datos}. Verifica el nombre de la carpeta y archivo.")