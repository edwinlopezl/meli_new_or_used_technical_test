import pandas as pd
import unicodedata
import re

class ml:

    @staticmethod
    def normalize_text(text):
        """
        Normalizes text by converting it to lowercase, removing accents, and replacing spaces with underscores.
        
        Args:
            text (str): Input text to normalize.
        
        Returns:
            str: Normalized text or an empty string if the input is None.
        """
        if text is None:
            return ''
        
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = re.sub(r'\s+', '_', text)
        return text

    @staticmethod
    def expand_json_field(df, columns, suffix='_json'):
        """
        Expands JSON-like columns in a DataFrame into separate columns.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of column names containing JSON-like structures.
            suffix (str): Suffix to use for conflicting column names (default is '_json').

        Returns:
            pd.DataFrame: DataFrame with expanded JSON fields.
        """
        df = df.copy()  # Evita modificar el DataFrame original

        for campo in columns:
            # Asegurar que los valores no sean NaN y sean diccionarios
            df[campo] = df[campo].apply(lambda x: x if isinstance(x, dict) else {})

            # Expandir la columna JSON
            nested_df = pd.json_normalize(df[campo], errors='ignore')

            # Renombrar columnas
            nested_df.columns = [f"{campo}_{col.replace('.', '_')}" for col in nested_df.columns]

            # Resetear índice para evitar desalineación
            nested_df.index = df.index  

            # Verificar columnas que ya existen en df
            overlap = df.columns.intersection(nested_df.columns)

            # Si hay conflicto de nombres, agregar sufijo
            if not overlap.empty:
                df = df.join(nested_df, rsuffix=suffix)
            else:
                df = df.join(nested_df)

            # Eliminar la columna original
            df.drop(columns=[campo], inplace=True)

        return df


    @staticmethod
    def expand_nested_fields(df, fields):
        """
        Expands nested list fields containing dictionaries into separate columns with their actual values.

        Args:
            df (pd.DataFrame): Input DataFrame.
            fields (list): List of column names containing nested lists of dictionaries.

        Returns:
            pd.DataFrame: DataFrame with expanded fields.
        """
        df = df.copy()  # Evita modificar el original
        all_new_columns = {}  # Diccionario para almacenar todas las nuevas columnas

        for field in fields:
            unique_keys = set()

            # Extraer todas las claves (incluyendo anidadas)
            for row in df[field].dropna():
                if isinstance(row, list):
                    for item in row:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                if isinstance(value, dict):  # Si hay diccionario anidado
                                    for sub_key in value.keys():
                                        unique_keys.add((key, sub_key))
                                else:
                                    unique_keys.add((key, None))  # No es un diccionario anidado

            # Crear nuevas columnas para cada clave encontrada
            for key_tuple in unique_keys:
                key, sub_key = key_tuple
                if sub_key:
                    col_name = f"{field}_{ml.normalize_text(key)}_{ml.normalize_text(sub_key)}"
                    all_new_columns[col_name] = df[field].apply(
                        lambda items: next((item[key][sub_key] for item in items if isinstance(item, dict) and key in item and isinstance(item[key], dict) and sub_key in item[key]), None)
                        if isinstance(items, list) else None
                    )
                else:
                    col_name = f"{field}_{ml.normalize_text(key)}"
                    all_new_columns[col_name] = df[field].apply(
                        lambda items: next((item[key] for item in items if isinstance(item, dict) and key in item), None)
                        if isinstance(items, list) else None
                    )

        # Agregar todas las nuevas columnas al DataFrame
        df = df.assign(**all_new_columns)

        # Eliminar las columnas originales solo al final
        df.drop(columns=fields, inplace=True)

        return df




    @staticmethod
    def expand_list_fields(df, fields):
        """
        Expands list fields into separate binary columns indicating presence of each unique value.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            fields (list): List of column names containing lists of values.
        
        Returns:
            pd.DataFrame: DataFrame with expanded fields.
        """
        for field in fields:
            unique_values = set()
            for row in df[field].dropna():
                if isinstance(row, list):
                    for value in row:
                        if isinstance(value, str):
                            unique_values.add(value)
            
            mapping = {val: ml.normalize_text(val) for val in unique_values}
            
            for original_value, norm_value in mapping.items():
                new_col_name = f"{field}_{norm_value}"
                df[new_col_name] = df[field].apply(
                    lambda items: "yes" if isinstance(items, list) and original_value in items else "no"
                )
            df = df.drop([field], axis=1)
        
        return df


    @staticmethod
    def expand_datetime_columns(df, columns):
        """
        Expands datetime columns into separate columns for year, month, day, hour, minute, and second.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of column names containing datetime values.
        
        Returns:
            pd.DataFrame: DataFrame with expanded datetime fields.
        """
        for col in columns:
            df[col] = pd.to_datetime(df[col])
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_minute"] = df[col].dt.minute
            df[f"{col}_second"] = df[col].dt.second
            df.drop(columns=col, inplace=True)
        return df
