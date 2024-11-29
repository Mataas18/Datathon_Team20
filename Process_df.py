"""
Prior:
columns = ['brand', 'corporation', 'county', 'indication']
n_samples = [420, 80, 39, 210]

"""

import pandas as pd


def process_dataframe_multiple_columns(df, columns, n_samples):
    """
    Procesa un DataFrame aplicando el criterio de seleccionar las N muestras más frecuentes
    para cada columna dada y renombrar el resto como 'UKW'.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        columns (list): Lista de columnas a procesar.
        n_samples (int): Número de muestras principales a mantener para cada columna.
    
    Returns:
        pd.DataFrame: Nuevo DataFrame con las columnas procesadas.
    """
    df_processed = df.copy()  # Crear una copia del DataFrame para no modificar el original
    
    for idx, column in enumerate(columns):
        # Contar las apariciones de cada valor en la columna
        value_counts = df[column].value_counts()
        
        # Seleccionar las N muestras más frecuentes
        top_values = value_counts.iloc[:n_samples[idx]].index
        
        # Renombrar los valores no seleccionados como 'UKW'
        df_processed[column] = df_processed[column].apply(
            lambda x: x if x in top_values else 'UKW'
        )
    
    return df_processed