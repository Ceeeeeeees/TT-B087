import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.preprocessing import LabelEncoder

def balancear_datos(df, metodo='manual', columna_objetivo='rendimiento_general_categoria', decimales=3):
    """
    Función para balancear datos utilizando diferentes técnicas
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame de pandas con los datos a balancear
    metodo : str, default='manual'
        Método de balanceo a utilizar: 'manual', 'smote', 'smotetomek', o 'smoteenn'
    columna_objetivo : str, default='rendimiento_general_categoria'
        Nombre de la columna objetivo para balancear
    decimales : int, default=3
        Número de decimales a mantener en las variables flotantes
        
    Retorna:
    --------
    DataFrame
        DataFrame balanceado
    """
    print(f"Distribución original de clases:\n{Counter(df[columna_objetivo])}")
    
    # Crear una copia del dataframe original
    df_original = df.copy()
    
    # Separar variable objetivo
    y = df[columna_objetivo].copy()
    
    # Comprobar si la variable objetivo es categórica
    if y.dtype == 'object':
        print("Codificando variable objetivo categórica...")
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y)
        # Guardar el mapeo para luego recuperar las etiquetas originales
        mapeo_clases = dict(zip(le_y.transform(le_y.classes_), le_y.classes_))
        print(f"Mapeo de clases: {mapeo_clases}")
    else:
        y_encoded = y.copy()
        le_y = None
    
    # Separar ID si existe
    tiene_id = 'id_alumno' in df.columns
    if tiene_id:
        ids_originales = df['id_alumno'].copy()
        X = df.drop(['id_alumno', columna_objetivo], axis=1).copy()
    else:
        X = df.drop(columna_objetivo, axis=1).copy()
    
    # Redondear variables flotantes antes del procesamiento
    cols_float = X.select_dtypes(include=['float']).columns
    for col in cols_float:
        X[col] = X[col].round(decimales)
    
    # Procesar columnas categóricas
    cols_categoricas = X.select_dtypes(include=['object']).columns.tolist()
    if cols_categoricas:
        print(f"Detectadas columnas categóricas: {cols_categoricas}")
        # Guardar el mapeo de columnas categóricas para recuperarlas después
        mapeos_categoricos = {}
        
        for col in cols_categoricas:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            mapeos_categoricos[col] = dict(zip(le.transform(le.classes_), le.classes_))
    else:
        mapeos_categoricos = None
    
    # Aplicar técnica de balanceo según el método seleccionado
    if metodo == 'manual':
        X_resampled, y_resampled = _balanceo_manual(X, y_encoded)
    elif metodo == 'smote':
        print("Aplicando SMOTE...")
        try:
            smote = SMOTE(random_state=0)
            X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
        except ValueError as e:
            print(f"Error al aplicar SMOTE: {e}")
            print("Verificando tipos de datos:")
            print(X.dtypes)
            raise
    elif metodo == 'smotetomek':
        print("Aplicando SMOTETomek...")
        try:
            smotetomek = SMOTETomek(random_state=0, sampling_strategy="all")
            X_resampled, y_resampled = smotetomek.fit_resample(X, y_encoded)
        except ValueError as e:
            print(f"Error al aplicar SMOTETomek: {e}")
            print("Verificando tipos de datos:")
            print(X.dtypes)
            raise
    elif metodo == 'smoteenn':
        print("Aplicando SMOTEENN...")
        try:
            smoteenn = SMOTEENN(random_state=0, sampling_strategy="all")
            X_resampled, y_resampled = smoteenn.fit_resample(X, y_encoded)
        except ValueError as e:
            print(f"Error al aplicar SMOTEENN: {e}")
            print("Verificando tipos de datos:")
            print(X.dtypes)
            raise
    else:
        raise ValueError(f"Método '{metodo}' no reconocido. Opciones válidas: 'manual', 'smote', 'smotetomek', 'smoteenn'")
    
    # Convertir a DataFrame manteniendo las columnas originales
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    
    # Redondear todos los valores flotantes después del balanceo
    cols_float_resampled = X_resampled_df.select_dtypes(include=['float']).columns
    for col in cols_float_resampled:
        X_resampled_df[col] = X_resampled_df[col].round(decimales)
    
    # Restaurar las columnas categóricas originales
    if mapeos_categoricos:
        for col, mapeo in mapeos_categoricos.items():
            X_resampled_df[col] = X_resampled_df[col].map(mapeo)
    
    # Restaurar la variable objetivo original
    if le_y is not None:
        y_resampled_original = pd.Series([mapeo_clases[y_val] for y_val in y_resampled], name=columna_objetivo)
    else:
        y_resampled_original = pd.Series(y_resampled, name=columna_objetivo)
    
    # Aplicar correcciones a los datos sintéticos
    X_resampled_df = _corregir_datos_sinteticos(X_resampled_df, df_original, decimales)
    
    # Unir características y variable objetivo
    df_balanceado = pd.concat([X_resampled_df, y_resampled_original], axis=1)
    
    # Añadir IDs si existían
    if tiene_id:
        df_balanceado = _asignar_ids(df_balanceado, ids_originales)
    
    print(f"\nDistribución después del balanceo ({metodo}):")
    print(Counter(y_resampled_original))
    
    return df_balanceado

def _balanceo_manual(X, y):
    """Implementación del balanceo manual (sobremuestreo + submuestreo)"""
    # Obtener la distribución de clases actual
    conteo_clases = Counter(y)
    clase_mayoritaria = conteo_clases.most_common(1)[0][0]
    
    # Definir estrategia para el oversampling
    # Aumentar todas las clases minoritarias a 95% de la clase mayoritaria
    tam_mayoritaria = conteo_clases[clase_mayoritaria]
    estrategia_over = {
        clase: int(0.95 * tam_mayoritaria) 
        for clase, conteo in conteo_clases.items() 
        if conteo < tam_mayoritaria
    }
    
    # Aplicar RandomOverSampler
    ros = RandomOverSampler(sampling_strategy=estrategia_over, random_state=42)
    X_over, y_over = ros.fit_resample(X, y)
    
    # Aplicar RandomUnderSampler si es necesario
    conteo_despues_over = Counter(y_over)
    tam_objetivo = int(conteo_despues_over.most_common(1)[0][1] * 1.0)  # Mantener igual
    
    estrategia_under = {
        clase: min(tam_objetivo, conteo) 
        for clase, conteo in conteo_despues_over.items()
    }
    
    rus = RandomUnderSampler(sampling_strategy=estrategia_under, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_over, y_over)
    
    return X_resampled, y_resampled

def _corregir_datos_sinteticos(df, df_original, decimales=3):
    """
    Corrige los datos sintéticos para mantener la consistencia
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con datos rebalanceados
    df_original : DataFrame
        DataFrame original para obtener información de columnas
    decimales : int
        Número de decimales a mantener en variables flotantes
        
    Retorna:
    --------
    DataFrame
        DataFrame con datos corregidos
    """
    # Redondear todas las columnas flotantes
    cols_float = df.select_dtypes(include=['float']).columns
    for col in cols_float:
        df[col] = df[col].round(decimales)
    
    # Identificar columnas que necesitan corrección específica
    
    # Corregir variables de educación (one-hot encoding)
    cols_educacion_madre = [col for col in df.columns if 'educacion_madre_' in col]
    cols_educacion_padre = [col for col in df.columns if 'educacion_padre_' in col]
    
    # Corregir variables one-hot
    if cols_educacion_madre:
        df = _corregir_one_hot_categorico(df, cols_educacion_madre)
    
    if cols_educacion_padre:
        df = _corregir_one_hot_categorico(df, cols_educacion_padre)
    
    # Corregir variables binarias (posesiones)
    cols_posesion = ['computadora', 'cuarto_estudio', 'television', 'carro']
    for col in cols_posesion:
        if col in df.columns:
            # Verificar si la columna es de tipo objeto o numérica
            if df[col].dtype == 'object':
                # Convertir valores de texto a binario (asumiendo 'No'/'Sí' o similar)
                valores_unicos = df_original[col].unique()
                if len(valores_unicos) == 2:
                    valor_negativo = sorted(valores_unicos)[0]
                    df[col] = df[col].apply(lambda x: 0 if x == valor_negativo else 1)
            else:
                df[col] = df[col].round().clip(0, 1).astype(int)
    
    # Corregir variable de libros (0-6)
    if 'libros' in df.columns:
        if df['libros'].dtype == 'object':
            # Mapeo específico para variable libros (si es categórica)
            mapeo_libros = {
                'Ninguno': 0,
                'Muy pocos (1-10)': 1,
                'Algunos (11-25)': 2,
                'Suficientes (26-100)': 3,
                'Bastantes (101-200)': 4,
                'Muchos (201-500)': 5,
                'Biblioteca grande (más de 500)': 6
            }
            # Aplicar mapeo si los valores coinciden, de lo contrario mantener
            if set(df['libros'].unique()).issubset(set(mapeo_libros.keys())):
                df['libros'] = df['libros'].map(mapeo_libros)
        else:
            df['libros'] = df['libros'].round().clip(0, 6).astype(int)
    
    # Verificar si hay promedios o agregados que deben ser recalculados
    # Por ejemplo, si hay una columna 'promedio' que es la media de otras columnas
    if 'promedio' in df.columns and all(col in df.columns for col in ['matematicas', 'lectura', 'ciencias']):
        df['promedio'] = df[['matematicas', 'lectura', 'ciencias']].mean(axis=1).round(decimales)
    
    return df

def _corregir_one_hot_categorico(df, cols):
    """
    Corrige variables one-hot para asegurar que solo una esté activa,
    verificando si son categoricas o numéricas
    """
    if not cols:
        return df
    
    # Verificar si las columnas son categóricas o numéricas
    if df[cols[0]].dtype == 'object':
        # Caso categórico
        for idx in df.index:
            # Verificar si hay más de un valor que no sea el valor nulo
            valores = df.loc[idx, cols].values
            valor_nulo = None  # Valor que representa "ninguno" o "falso"
            
            # Intentar identificar el valor nulo (más común o primer valor)
            valores_unicos = np.unique(valores)
            if len(valores_unicos) > 1:
                valor_nulo = valores_unicos[0]  # Asumir que el primer valor es el "nulo"
            
            # Contar cuántos valores no son el valor nulo
            if valor_nulo is not None:
                valores_activos = [v for v in valores if v != valor_nulo]
                if len(valores_activos) != 1:
                    # Resetear todos a valor nulo
                    df.loc[idx, cols] = valor_nulo
                    # Seleccionar uno aleatoriamente si había más de uno activo
                    if valores_activos:
                        col_aleatorio = np.random.choice(cols)
                        df.loc[idx, col_aleatorio] = np.random.choice(valores_activos)
    else:
        # Caso numérico (one-hot encoding tradicional)
        for idx in df.index:
            valores = df.loc[idx, cols].values
            if sum(valores) != 1:
                # Restablecer a todos 0
                df.loc[idx, cols] = 0
                # Asignar 1 a la posición con el valor máximo
                max_idx = np.argmax(valores)
                if max_idx < len(cols):
                    df.loc[idx, cols[max_idx]] = 1
                else:
                    # Si no hay un máximo claro, asignar aleatoriamente
                    df.loc[idx, np.random.choice(cols)] = 1
    
    return df

def _asignar_ids(df_balanceado, ids_originales):
    """Asigna IDs a las muestras, generando nuevos IDs para muestras sintéticas"""
    # Determinar cuántas muestras originales vs sintéticas hay
    num_originales = len(ids_originales)
    num_sinteticas = len(df_balanceado) - num_originales
    
    if num_sinteticas > 0:
        # Generar nuevos IDs para muestras sintéticas
        max_id = ids_originales.max()
        nuevos_ids = np.concatenate([
            ids_originales.values[:min(num_originales, len(df_balanceado))],
            np.arange(max_id + 1, max_id + 1 + max(0, len(df_balanceado) - num_originales))
        ])
        df_balanceado['id_alumno'] = nuevos_ids[:len(df_balanceado)]
    else:
        # Si hay undersampling, usar los IDs existentes
        df_balanceado['id_alumno'] = ids_originales.values[:len(df_balanceado)]
    
    return df_balanceado

# Ejemplo de uso:
if __name__ == "__main__":
    try:
        df = pd.read_csv('./alumnosMexico2022ProcesadosCategoricos.csv')
        print("Columnas del dataset:")
        print(df.columns.tolist())
        print("\nTipos de datos:")
        print(df.dtypes)
        
        # Aplicar el balanceo con límite de 3 decimales
        print("\n\n--- Aplicando SMOTETomek ---")
        df_balanceado = balancear_datos(df, metodo='smotetomek', decimales=3)
        df_balanceado.to_csv('alumnos_balanceados_smotetomek.csv', index=False)
        
        print("\n\n--- Aplicando SMOTEENN ---")
        df_balanceado_enn = balancear_datos(df, metodo='smoteenn', decimales=3)
        df_balanceado_enn.to_csv('alumnos_balanceados_smoteenn.csv', index=False)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())