"""
Prior:
columns = ['brand', 'corporation', 'county', 'indication']
n_samples = [420, 80, 39, 210]

"""

import ast
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def process_dataframe_multiple_columns(df, columns, n_samples):
    """
    Processes a DataFrame by selecting the top N most frequent samples
    for each given column and renaming the rest as 'UKW'.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to process.
        n_samples (list): Number of top samples to keep for each column.
    
    Returns:
        pd.DataFrame: New DataFrame with processed columns.
    """
    df_processed = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    
    for idx, column in enumerate(columns):
        # Count the occurrences of each value in the column
        value_counts = df[column].value_counts()
        
        # Select the top N most frequent samples
        top_values = value_counts.iloc[:n_samples[idx]].index
        
        # Rename the values not selected as 'UKW'
        df_processed[column] = df_processed[column].apply(
            lambda x: x if x in top_values else 'UKW'
        )
    
    return df_processed

"""
df = convert_categorical_to_numerical(df, 'indication')
"""
def generate_unique_codes(df, column):
    """
    Genera una lista de etiquetas únicas ordenadas a partir de una columna que puede contener
    cadenas simples o listas de valores.

    Args:
        df (pd.DataFrame): El DataFrame que contiene la columna de interés.
        column (str): Nombre de la columna con valores categóricos.

    Returns:
        list: Lista de etiquetas únicas ordenadas.
    """
    # Verificar si los elementos son listas o cadenas simples
    first_value = df[column].iloc[0]
    
    if isinstance(first_value, str):
        try:
            # Intentar interpretar como lista
            parsed_value = ast.literal_eval(first_value)
            if isinstance(parsed_value, list):
                # Si es lista, parsear toda la columna
                parsed_lists = df[column].apply(ast.literal_eval)
                unique_codes = sorted(set(code for sublist in parsed_lists for code in sublist))
            else:
                # Si no es lista, tratar como cadenas simples
                unique_codes = sorted(df[column].unique())
        except (ValueError, SyntaxError):
            # Si no se puede interpretar como lista, tratar como cadenas simples
            unique_codes = sorted(df[column].unique())
    else:
        # Si no es cadena, tratar directamente como valores únicos
        unique_codes = sorted(df[column].unique())
    
    return unique_codes


def convert_categorical_to_multilabel(df, column, unique_codes=None):
    """
    Convierte una columna categórica a un formato multilabel (vector binario) 
    basado en una lista de códigos únicos.

    Args:
        df (pd.DataFrame): DataFrame que contiene la columna de interés.
        column (str): Nombre de la columna con valores categóricos.
        unique_codes (list): Lista de etiquetas únicas para generar el vector multilabel.

    Returns:
        pd.DataFrame: DataFrame con la columna convertida en formato multilabel.
    """
    # Verificar que unique_codes sea proporcionado
    if unique_codes is None:
        raise ValueError('unique_codes must be provided')
    
    # Paso 1: Parsear listas o manejar cadenas simples
    def parse_value(value):
        try:
            parsed = ast.literal_eval(value)  # Intentar interpretar como lista
            if isinstance(parsed, list):
                return parsed  # Si es lista, devolverla
            return [parsed]  # Si no, encapsular como lista
        except (ValueError, SyntaxError):
            return [value]  # Si no se puede interpretar, tratar como cadena simple

    df[f'{column}_parsed'] = df[column].apply(parse_value)
    
    # Paso 2: Crear un vector multilabel para cada fila
    def create_multilabel_vector(labels):
        return [1 if code in labels else 0 for code in unique_codes]
    
    df[f'{column}_multilabel'] = df[f'{column}_parsed'].apply(create_multilabel_vector)
    
    # Paso 3: Eliminar columnas auxiliares si no son necesarias
    df = df.drop(columns=[f'{column}_parsed'])
    df = df.drop(columns=[column])
    
    return df

def date_codification(df):
    """
    Codifies a date column into day, month, and year.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the date column.
    column (str): The name of the column to be transformed.
    Returns:
    pd.DataFrame: The DataFrame with the original column replaced by the day, month, and year columns.
    """
    column = 'date'
    df[column] = pd.to_datetime(df[column])
    df[f'{column}_month'] = df[column].dt.month
    year = df[column].dt.year
    df['date_codification'] = df[f'{column}_month'] + 12 * (year)
    df = df.drop(columns=[column])
    
    return df


def PCA(X, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on the input data.
    Parameters:
    X (pd.DataFrame): The input data.
    Returns:
    pd.DataFrame: The transformed data.
    """
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca

import pandas as pd

def format_dataframe2train(X, fillna_value=0):
    """
    Formatea un DataFrame expandiendo columnas con listas, 
    seleccionando datos numéricos y convirtiéndolos a float.

    Args:
        X (pd.DataFrame): DataFrame de entrada.
        fillna_value (float): Valor para reemplazar NaN. Por defecto, 0.

    Returns:
        pd.DataFrame: DataFrame formateado.
    """
    # Identificar columnas que contienen listas o secuencias
    multilabel_columns = [col for col in X.columns if isinstance(X[col].iloc[0], list)]

    # Expandir columnas con listas en múltiples columnas
    for col in multilabel_columns:
        expanded_cols = pd.DataFrame(X[col].tolist(), index=X.index).add_prefix(f"{col}_")
        X = pd.concat([X, expanded_cols], axis=1)
    
    # Eliminar las columnas originales con listas
    X = X.drop(columns=multilabel_columns)

    # Convertir todo a datos numéricos, rellenando NaN si es necesario
    X = X.apply(pd.to_numeric, errors='coerce')
    if fillna_value is not None:
        X = X.fillna(fillna_value)

    return X
