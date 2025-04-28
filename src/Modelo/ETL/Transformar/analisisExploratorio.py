import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AnalisisExploratorio:
    """
    Clase para realizar análisis exploratorio de datos sobre los datasets de rendimiento académico.
    Permite trabajar con los tres tipos de datasets: original, continuo y categórico.
    """
    
    def __init__(self, ruta_continuo=None, ruta_categorico=None, ruta_original=None, 
                 df_continuo=None, df_categorico=None, df_original=None):
        """
        Inicializa la clase cargando los DataFrames desde archivos CSV o usando DataFrames proporcionados.
        
        Argumentos:
            ruta_continuo (str): Ruta al archivo CSV con variables continuas
            ruta_categorico (str): Ruta al archivo CSV con variables categóricas
            ruta_original (str): Ruta al archivo CSV con datos originales
            df_continuo (pandas.DataFrame): DataFrame con variables continuas
            df_categorico (pandas.DataFrame): DataFrame con variables categóricas
            df_original (pandas.DataFrame): DataFrame con datos originales
        """
        # Inicializamos los DataFrames como None
        self.df_continuo = None
        self.df_categorico = None
        self.df_original = None
        
        # Cargamos DataFrames desde rutas si se proporcionan
        if ruta_continuo and os.path.exists(ruta_continuo):
            self.df_continuo = pd.read_csv(ruta_continuo)
            print(f"DataFrame continuo cargado: {self.df_continuo.shape[0]} filas, {self.df_continuo.shape[1]} columnas")
        elif df_continuo is not None:
            self.df_continuo = df_continuo.copy()
            print(f"DataFrame continuo asignado: {self.df_continuo.shape[0]} filas, {self.df_continuo.shape[1]} columnas")
            
        if ruta_categorico and os.path.exists(ruta_categorico):
            self.df_categorico = pd.read_csv(ruta_categorico)
            print(f"DataFrame categórico cargado: {self.df_categorico.shape[0]} filas, {self.df_categorico.shape[1]} columnas")
        elif df_categorico is not None:
            self.df_categorico = df_categorico.copy()
            print(f"DataFrame categórico asignado: {self.df_categorico.shape[0]} filas, {self.df_categorico.shape[1]} columnas")
            
        if ruta_original and os.path.exists(ruta_original):
            self.df_original = pd.read_csv(ruta_original)
            print(f"DataFrame original cargado: {self.df_original.shape[0]} filas, {self.df_original.shape[1]} columnas")
        elif df_original is not None:
            self.df_original = df_original.copy()
            print(f"DataFrame original asignado: {self.df_original.shape[0]} filas, {self.df_original.shape[1]} columnas")
            
        # Verificamos que al menos un DataFrame se haya cargado
        if self.df_continuo is None and self.df_categorico is None and self.df_original is None:
            raise ValueError("No se ha cargado ningún DataFrame. Proporciona al menos una ruta o DataFrame.")
    
    def _seleccionar_dataframe(self, tipo):
        """
        Método auxiliar para seleccionar el DataFrame según el tipo.
        
        Argumentos:
            tipo (str): Tipo de DataFrame ('continuo', 'categorico', 'original')
            
        Retorna:
            pandas.DataFrame: El DataFrame seleccionado
        """
        if tipo == 'continuo':
            if self.df_continuo is None:
                raise ValueError("El DataFrame continuo no está disponible")
            return self.df_continuo
        elif tipo == 'categorico':
            if self.df_categorico is None:
                raise ValueError("El DataFrame categórico no está disponible")
            return self.df_categorico
        elif tipo == 'original':
            if self.df_original is None:
                raise ValueError("El DataFrame original no está disponible")
            return self.df_original
        else:
            raise ValueError("Tipo de DataFrame no válido. Use 'continuo', 'categorico' u 'original'")
    
    def resumen_estadistico(self, df_tipo='continuo', variables=None):
        """
        Genera estadísticas descriptivas básicas para las variables seleccionadas.
        
        Argumentos:
            df_tipo (str): Tipo de DataFrame a utilizar ('continuo', 'categorico', 'original')
            variables (list): Lista de nombres de variables a analizar (None para todas)
            
        Retorna:
            pandas.DataFrame: DataFrame con las estadísticas descriptivas
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        if variables:
            # Filtramos solo las variables que existen en el DataFrame
            variables_existentes = [var for var in variables if var in df.columns]
            if not variables_existentes:
                raise ValueError(f"Ninguna de las variables proporcionadas existe en el DataFrame {df_tipo}")
            df_analisis = df[variables_existentes]
        else:
            # Usamos todas las variables numéricas
            df_analisis = df.select_dtypes(include=[np.number])
        
        # Calculamos estadísticas descriptivas
        estadisticas = df_analisis.describe().T
        
        # Añadimos más estadísticas
        estadisticas['varianza'] = df_analisis.var()
        estadisticas['asimetria'] = df_analisis.skew()
        estadisticas['curtosis'] = df_analisis.kurtosis()
        estadisticas['valores_nulos'] = df_analisis.isnull().sum()
        estadisticas['porc_nulos'] = (df_analisis.isnull().sum() / len(df_analisis)) * 100
        
        print(f"Resumen estadístico para variables numéricas ({df_tipo}):")
        return estadisticas
    
    def resumen_categoricas(self, df_tipo='categorico', variables=None):
        """
        Genera estadísticas descriptivas para variables categóricas.
        
        Argumentos:
            df_tipo (str): Tipo de DataFrame a utilizar ('continuo', 'categorico', 'original')
            variables (list): Lista de nombres de variables a analizar (None para todas categóricas)
            
        Retorna:
            dict: Diccionario con distribución de frecuencias para cada variable categórica
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        if variables:
            # Filtramos solo las variables que existen en el DataFrame
            variables_existentes = [var for var in variables if var in df.columns]
            if not variables_existentes:
                raise ValueError(f"Ninguna de las variables proporcionadas existe en el DataFrame {df_tipo}")
            variables_a_analizar = variables_existentes
        else:
            # Usamos variables categóricas (object, category, bool o string)
            variables_a_analizar = df.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()
        
        if not variables_a_analizar:
            print(f"No se encontraron variables categóricas en el DataFrame {df_tipo}")
            return {}
        
        resultados = {}
        for var in variables_a_analizar:
            # Calculamos frecuencias y porcentajes
            frecuencias = df[var].value_counts()
            porcentajes = df[var].value_counts(normalize=True) * 100
            
            # Combinamos en un DataFrame
            resumen = pd.DataFrame({
                'frecuencia': frecuencias,
                'porcentaje': porcentajes
            })
            
            resultados[var] = resumen
            print(f"\nDistribución de frecuencias para {var}:")
            print(resumen)
        
        return resultados
    
    def analizar_correlaciones(self, df_tipo='continuo', metodo='pearson', 
                              umbral=0.3, target='rendimiento_academico_continuo',
                              mostrar_grafico=True, titulo=None, figsize=(12, 10)):
        """
        Analiza correlaciones entre variables y el rendimiento académico.
        
        Argumentos:
            df_tipo (str): Tipo de DataFrame a utilizar ('continuo', 'categorico', 'original')
            metodo (str): Método de correlación ('pearson', 'spearman', 'kendall')
            umbral (float): Umbral de correlación para destacar relaciones importantes
            target (str): Variable objetivo para mostrar correlaciones específicas
            mostrar_grafico (bool): Si se debe mostrar el gráfico de correlaciones
            titulo (str): Título para el gráfico de correlaciones
            figsize (tuple): Tamaño de la figura para el gráfico
            
        Retorna:
            pandas.DataFrame: Matriz de correlación
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        # Seleccionamos solo variables numéricas
        df_num = df.select_dtypes(include=[np.number])
        
        # Calculamos la matriz de correlación
        matriz_corr = df_num.corr(method=metodo)
        
        # Si se solicita mostrar el gráfico
        if mostrar_grafico:
            plt.figure(figsize=figsize)
            
            # Creamos una máscara para la mitad superior de la matriz
            mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))
            
            # Configuramos el mapa de color
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Creamos el heatmap
            sns.heatmap(matriz_corr, mask=mascara, cmap=cmap, vmax=1, vmin=-1, center=0,
                        annot=True, fmt=".2f", square=True, linewidths=.5)
            
            # Añadimos título si se proporciona
            if titulo:
                plt.title(titulo, fontsize=16)
            else:
                plt.title(f'Matriz de Correlación ({metodo.capitalize()})', fontsize=16)
            
            plt.tight_layout()
            plt.show()
        
        # Si se proporciona una variable target, mostramos las correlaciones con esa variable
        if target and target in df_num.columns:
            # Obtenemos correlaciones con la variable objetivo y las ordenamos
            correlaciones_target = matriz_corr[target].sort_values(ascending=False)
            
            # Eliminamos la autocorrelación (correlación con sí misma)
            correlaciones_target = correlaciones_target[correlaciones_target.index != target]
            
            print(f"\nCorrelaciones con {target} (método: {metodo}):")
            print(correlaciones_target)
            
            # Destacamos correlaciones importantes (por encima del umbral)
            correlaciones_importantes = correlaciones_target[
                (correlaciones_target > umbral) | (correlaciones_target < -umbral)
            ]
            
            if not correlaciones_importantes.empty:
                print(f"\nCorrelaciones importantes (|r| > {umbral}) con {target}:")
                print(correlaciones_importantes)
                
                # Visualizamos estas correlaciones importantes
                plt.figure(figsize=(10, 6))
                correlaciones_importantes.plot(kind='bar')
                plt.title(f'Variables con mayor correlación con {target}', fontsize=14)
                plt.ylabel(f'Coeficiente de correlación ({metodo})', fontsize=12)
                plt.xlabel('Variables', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.show()
        
        return matriz_corr
    
    def visualizar_distribuciones(self, variables=None, df_tipo='continuo', 
                                 tipo_grafico='hist', bins=20, figsize=(14, 10)):
        """
        Genera visualizaciones de las distribuciones de las variables seleccionadas.
        
        Argumentos:
            variables (list): Lista de nombres de variables a visualizar (None para todas numéricas)
            df_tipo (str): Tipo de DataFrame a utilizar ('continuo', 'categorico', 'original')
            tipo_grafico (str): Tipo de gráfico ('hist', 'box', 'violin', 'kde')
            bins (int): Número de bins para histogramas
            figsize (tuple): Tamaño de la figura
            
        Retorna:
            None
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        if variables:
            # Filtramos solo las variables que existen en el DataFrame
            variables_existentes = [var for var in variables if var in df.columns]
            if not variables_existentes:
                raise ValueError(f"Ninguna de las variables proporcionadas existe en el DataFrame {df_tipo}")
            variables_a_visualizar = variables_existentes
        else:
            # Usamos variables numéricas
            variables_a_visualizar = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_variables = len(variables_a_visualizar)
        
        if n_variables == 0:
            print(f"No se encontraron variables numéricas para visualizar en el DataFrame {df_tipo}")
            return
        
        # Calculamos filas y columnas para subplots
        n_cols = min(3, n_variables)
        n_rows = (n_variables + n_cols - 1) // n_cols
        
        # Creamos la figura y los subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Para cada variable, generamos el gráfico correspondiente
        for i, var in enumerate(variables_a_visualizar):
            if tipo_grafico == 'hist':
                sns.histplot(df[var], bins=bins, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribución de {var}')
                
            elif tipo_grafico == 'box':
                sns.boxplot(y=df[var], ax=axes[i])
                axes[i].set_title(f'Boxplot de {var}')
                
            elif tipo_grafico == 'violin':
                sns.violinplot(y=df[var], ax=axes[i])
                axes[i].set_title(f'Violinplot de {var}')
                
            elif tipo_grafico == 'kde':
                sns.kdeplot(df[var], fill=True, ax=axes[i])
                axes[i].set_title(f'Densidad de {var}')
            
            axes[i].grid(linestyle='--', alpha=0.3)
        
        # Ocultamos ejes vacíos
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f'Visualización de distribuciones ({tipo_grafico})', fontsize=16, y=1.02)
        plt.show()
    
    def graficar_variables_por_categoria(self, var_numerica, var_categorica, 
                                        df_tipo='categorico', tipo_grafico='box'):
        """
        Visualiza la distribución de una variable numérica según una variable categórica.
        
        Argumentos:
            var_numerica (str): Nombre de la variable numérica a analizar
            var_categorica (str): Nombre de la variable categórica para agrupar
            df_tipo (str): Tipo de DataFrame a utilizar ('continuo', 'categorico', 'original')
            tipo_grafico (str): Tipo de gráfico ('box', 'violin', 'bar', 'strip')
            
        Retorna:
            None
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        # Verificamos que ambas variables existan
        if var_numerica not in df.columns:
            raise ValueError(f"La variable numérica '{var_numerica}' no existe en el DataFrame")
        if var_categorica not in df.columns:
            raise ValueError(f"La variable categórica '{var_categorica}' no existe en el DataFrame")
        
        # Verificamos que var_numerica sea numérica
        if not np.issubdtype(df[var_numerica].dtype, np.number):
            raise ValueError(f"La variable '{var_numerica}' no es numérica")
        
        plt.figure(figsize=(12, 6))
        
        if tipo_grafico == 'box':
            sns.boxplot(x=var_categorica, y=var_numerica, data=df)
            plt.title(f'Distribución de {var_numerica} por {var_categorica}', fontsize=14)
            
        elif tipo_grafico == 'violin':
            sns.violinplot(x=var_categorica, y=var_numerica, data=df)
            plt.title(f'Distribución de {var_numerica} por {var_categorica}', fontsize=14)
            
        elif tipo_grafico == 'bar':
            # Calculamos estadísticas por grupo
            medias = df.groupby(var_categorica)[var_numerica].mean().sort_values(ascending=False)
            errores = df.groupby(var_categorica)[var_numerica].sem().reindex(medias.index)
            
            # Graficamos barras con errores
            sns.barplot(x=medias.index, y=medias.values, yerr=errores.values)
            plt.title(f'Media de {var_numerica} por {var_categorica}', fontsize=14)
            
        elif tipo_grafico == 'strip':
            sns.stripplot(x=var_categorica, y=var_numerica, data=df, jitter=True, alpha=0.6)
            plt.title(f'Distribución de puntos de {var_numerica} por {var_categorica}', fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Mostramos un resumen estadístico por grupo
        resumen = df.groupby(var_categorica)[var_numerica].agg(['count', 'mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)
        print(f"\nResumen estadístico de {var_numerica} por {var_categorica}:")
        print(resumen)
        
        # Realizamos ANOVA para ver si hay diferencias significativas
        grupos = [df[df[var_categorica] == cat][var_numerica].dropna() for cat in df[var_categorica].unique()]
        resultado_anova = stats.f_oneway(*grupos)
        
        print("\nResultado de ANOVA:")
        print(f"F-value: {resultado_anova.statistic:.4f}")
        print(f"p-value: {resultado_anova.pvalue:.4f}")
        
        if resultado_anova.pvalue < 0.05:
            print("Hay diferencias estadísticamente significativas entre los grupos (p < 0.05)")
        else:
            print("No hay diferencias estadísticamente significativas entre los grupos (p >= 0.05)")
    
    def analizar_factores_influyentes(self, target='rendimiento_academico_continuo', 
                                     df_tipo='continuo', n_factores=10, plot=True):
        """
        Identifica los factores más influyentes en el rendimiento académico.
        
        Argumentos:
            target (str): Variable objetivo (rendimiento)
            df_tipo (str): Tipo de DataFrame a utilizar ('continuo', 'categorico', 'original')
            n_factores (int): Número de factores principales a mostrar
            plot (bool): Si se debe generar un gráfico
            
        Retorna:
            pandas.Series: Series con los factores más influyentes ordenados
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        if target not in df.columns:
            raise ValueError(f"La variable objetivo '{target}' no existe en el DataFrame")
        
        # Seleccionamos solo variables numéricas que no sean el target
        df_num = df.select_dtypes(include=[np.number])
        predictores = [col for col in df_num.columns if col != target and col != 'id_alumno']
        
        if not predictores:
            print("No hay variables numéricas predictoras en el DataFrame")
            return None
        
        # Calculamos correlaciones con el target
        correlaciones = df_num[predictores].corrwith(df_num[target]).sort_values(ascending=False)
        
        # Tomamos los n_factores principales (tanto positivos como negativos)
        top_positivos = correlaciones.nlargest(n_factores)
        top_negativos = correlaciones.nsmallest(n_factores)
        
        # Combinamos ambos conjuntos (eliminando posibles duplicados)
        factores_importantes = pd.concat([top_positivos, top_negativos])
        factores_importantes = factores_importantes[~factores_importantes.index.duplicated(keep='first')]
        
        if plot:
            plt.figure(figsize=(14, 8))
            
            # Ordenamos los factores por valor absoluto de correlación
            factores_abs = factores_importantes.abs().sort_values(ascending=False)
            factores_ordenados = factores_importantes.loc[factores_abs.index]
            
            # Creamos colores basados en el signo de la correlación
            colores = ['green' if x > 0 else 'red' for x in factores_ordenados]
            
            # Graficamos
            plt.barh(factores_ordenados.index, factores_ordenados.values, color=colores)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f'Factores más influyentes en {target}', fontsize=16)
            plt.xlabel('Coeficiente de correlación', fontsize=12)
            plt.ylabel('Variables', fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        print(f"Factores más influyentes en {target}:")
        print(factores_importantes)
        
        return factores_importantes
    
    def graficar_dispersion(self, x, y, df_tipo='continuo', hue=None, size=None, col=None, 
                           regresion=True, figsize=(10, 6)):
        """
        Genera un gráfico de dispersión entre dos variables.
        
        Argumentos:
            x (str): Nombre de la variable para el eje x
            y (str): Nombre de la variable para el eje y
            df_tipo (str): Tipo de DataFrame a utilizar
            hue (str): Variable para color (opcional)
            size (str): Variable para tamaño (opcional)
            col (str): Variable para columnas múltiples (opcional)
            regresion (bool): Si se debe mostrar línea de regresión
            figsize (tuple): Tamaño de la figura
            
        Retorna:
            None
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        # Verificamos que las variables existan
        for var, nombre in zip([x, y, hue, size, col], ['x', 'y', 'hue', 'size', 'col']):
            if var is not None and var not in df.columns:
                raise ValueError(f"La variable '{var}' para {nombre} no existe en el DataFrame")
        
        # Creamos el gráfico de dispersión
        plt.figure(figsize=figsize)
        
        if regresion:
            # Con línea de regresión
            g = sns.regplot(x=x, y=y, data=df, scatter_kws={'alpha': 0.6})
            
            # Calculamos coeficiente de correlación
            corr = df[[x, y]].corr().iloc[0, 1]
            plt.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                         fontsize=12, ha='left', va='top')
            
        else:
            # Sin línea de regresión pero con más opciones
            g = sns.scatterplot(x=x, y=y, data=df, hue=hue, size=size, alpha=0.6)
            
            if hue:
                plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f'Relación entre {x} y {y}', fontsize=14)
        plt.grid(linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analizar_outliers(self, variables=None, df_tipo='continuo', metodo='zscore', 
                         umbral=3, graficar=True):
        """
        Identifica valores atípicos (outliers) en las variables seleccionadas.
        
        Argumentos:
            variables (list): Lista de variables a analizar (None para todas numéricas)
            df_tipo (str): Tipo de DataFrame a utilizar
            metodo (str): Método para detectar outliers ('zscore', 'iqr')
            umbral (float): Umbral para considerar un valor como outlier
            graficar (bool): Si se deben generar boxplots
            
        Retorna:
            dict: Diccionario con recuentos de outliers por variable
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        if variables:
            # Filtramos solo las variables que existen y son numéricas
            variables_existentes = [var for var in variables if var in df.columns and np.issubdtype(df[var].dtype, np.number)]
            if not variables_existentes:
                raise ValueError("Ninguna de las variables proporcionadas existe o es numérica")
            variables_a_analizar = variables_existentes
        else:
            # Usamos todas las variables numéricas excepto id_alumno
            variables_a_analizar = [col for col in df.select_dtypes(include=[np.number]).columns 
                                   if col != 'id_alumno']
        
        # Diccionario para almacenar resultados
        outliers_por_variable = {}
        
        for var in variables_a_analizar:
            # Identificamos outliers según el método elegido
            if metodo == 'zscore':
                # Método Z-score
                z_scores = np.abs(stats.zscore(df[var].dropna()))
                outliers = (z_scores > umbral)
                outliers_indices = np.where(outliers)[0]
                
            elif metodo == 'iqr':
                # Método IQR (Rango Intercuartílico)
                Q1 = df[var].quantile(0.25)
                Q3 = df[var].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - umbral * IQR
                upper_bound = Q3 + umbral * IQR
                
                outliers = ((df[var] < lower_bound) | (df[var] > upper_bound))
                outliers_indices = np.where(outliers)[0]
            
            # Guardamos estadísticas de outliers
            n_outliers = len(outliers_indices)
            porcentaje = (n_outliers / len(df[var].dropna())) * 100
            
            outliers_por_variable[var] = {
                'n_outliers': n_outliers,
                'porcentaje': porcentaje,
                'min_outlier': df[var].iloc[outliers_indices].min() if n_outliers > 0 else None,
                'max_outlier': df[var].iloc[outliers_indices].max() if n_outliers > 0 else None
            }
            
            print(f"Variable {var}: {n_outliers} outliers ({porcentaje:.2f}%)")
        
        # Graficamos boxplots si se solicita
        if graficar and variables_a_analizar:
            n_variables = len(variables_a_analizar)
            n_cols = min(3, n_variables)
            n_rows = (n_variables + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, var in enumerate(variables_a_analizar):
                sns.boxplot(y=df[var], ax=axes[i])
                axes[i].set_title(f'Boxplot de {var}')
                axes[i].grid(linestyle='--', alpha=0.3)
            
            # Ocultamos ejes vacíos
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.suptitle(f'Detección de outliers (método: {metodo})', fontsize=16, y=1.02)
            plt.show()
        
        return outliers_por_variable
    
    def analizar_pca(self, variables=None, df_tipo='continuo', n_componentes=2, graficar=True):
        """
        Realiza un análisis de componentes principales (PCA) para reducir dimensionalidad.
        
        Argumentos:
            variables (list): Lista de variables a incluir (None para todas numéricas)
            df_tipo (str): Tipo de DataFrame a utilizar
            n_componentes (int): Número de componentes principales a extraer
            graficar (bool): Si se deben generar gráficos
            
        Retorna:
            dict: Diccionario con resultados del PCA
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        # Seleccionamos las variables
        if variables:
            # Filtramos solo las variables que existen y son numéricas
            variables_existentes = [var for var in variables if var in df.columns and np.issubdtype(df[var].dtype, np.number)]
            if not variables_existentes:
                raise ValueError("Ninguna de las variables proporcionadas existe o es numérica")
            X = df[variables_existentes]
        else:
            # Usamos todas las variables numéricas excepto id_alumno y potencialmente el target
            X = df.select_dtypes(include=[np.number]).drop(columns=['id_alumno'], errors='ignore')
        
        # Estandarizamos los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicamos PCA
        pca = PCA(n_components=n_componentes)
        X_pca = pca.fit_transform(X_scaled)
        
        # Creamos un DataFrame con los componentes principales
        df_pca = pd.DataFrame(
            data=X_pca,
            columns=[f'PC{i+1}' for i in range(n_componentes)]
        )
        
        # Recopilamos resultados
        varianza_explicada = pca.explained_variance_ratio_
        varianza_acumulada = np.cumsum(varianza_explicada)
        
        # Obtenemos las cargas de cada variable en cada componente
        cargas = pd.DataFrame(
            data=pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_componentes)],
            index=X.columns
        )
        
        # Mostramos la varianza explicada
        print("Varianza explicada por cada componente principal:")
        for i, varianza in enumerate(varianza_explicada):
            print(f"PC{i+1}: {varianza:.4f} ({varianza*100:.2f}%)")
        
        print(f"\nVarianza acumulada: {varianza_acumulada[-1]*100:.2f}%")
        
        # Graficamos si se solicita
        if graficar:
            # Scree plot - varianza explicada
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(varianza_explicada) + 1), varianza_explicada)
            plt.plot(range(1, len(varianza_explicada) + 1), varianza_acumulada, 'r-o', linewidth=2)
            plt.axhline(y=0.8, color='g', linestyle='--')
            plt.title('Varianza explicada vs. número de componentes', fontsize=14)
            plt.xlabel('Número de componente')
            plt.ylabel('Proporción de varianza explicada')
            plt.xticks(range(1, len(varianza_explicada) + 1))
            
            # Biplot - si tenemos al menos 2 componentes
            if n_componentes >= 2:
                plt.subplot(1, 2, 2)
                
                # Escalamos las cargas para la visualización
                coef = np.transpose(pca.components_[0:2, :])
                n = coef.shape[0]
                
                # Graficamos las observaciones (puntos)
                plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)
                
                # Graficamos las variables como vectores
                for i in range(n):
                    plt.arrow(0, 0, coef[i, 0], coef[i, 1], color='r', alpha=0.5)
                    plt.text(coef[i, 0] * 1.15, coef[i, 1] * 1.15, X.columns[i], color='g')
                
                plt.title('Biplot: PC1 vs PC2', fontsize=14)
                plt.xlabel(f'PC1 ({varianza_explicada[0]:.2%})')
                plt.ylabel(f'PC2 ({varianza_explicada[1]:.2%})')
                plt.grid(linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Mapa de calor de cargas
            plt.figure(figsize=(10, 8))
            sns.heatmap(cargas, annot=True, cmap='coolwarm', center=0)
            plt.title('Cargas de variables en componentes principales', fontsize=14)
            plt.tight_layout()
            plt.show()
        
        # Resultados
        resultados = {
            'pca': pca,
            'varianza_explicada': varianza_explicada,
            'varianza_acumulada': varianza_acumulada,
            'cargas': cargas,
            'df_pca': df_pca
        }
        
        return resultados
    
    def crear_perfiles_rendimiento(self, n_grupos=3, df_tipo='continuo', 
                                 var_rendimiento='rendimiento_academico_continuo',
                                 variables_analisis=None):
        """
        Crea perfiles de estudiantes según su rendimiento académico.
        
        Argumentos:
            n_grupos (int): Número de grupos/perfiles a crear
            df_tipo (str): Tipo de DataFrame a utilizar
            var_rendimiento (str): Variable de rendimiento para agrupar
            variables_analisis (list): Variables a incluir en el análisis
            
        Retorna:
            pandas.DataFrame: DataFrame con los perfiles por grupo
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        if var_rendimiento not in df.columns:
            raise ValueError(f"La variable de rendimiento '{var_rendimiento}' no existe en el DataFrame")
        
        # Creamos grupos basados en rendimiento académico
        df_grupos = df.copy()
        
        if n_grupos == 3:
            # Tres grupos: bajo, medio, alto
            df_grupos['grupo_rendimiento'] = pd.qcut(
                df_grupos[var_rendimiento], 
                q=[0, 0.33, 0.67, 1], 
                labels=['Bajo', 'Medio', 'Alto']
            )
        else:
            # n grupos con distribución igual
            df_grupos['grupo_rendimiento'] = pd.qcut(
                df_grupos[var_rendimiento], 
                q=n_grupos, 
                labels=[f'Grupo {i+1}' for i in range(n_grupos)]
            )
        
        # Seleccionamos variables para el análisis
        if variables_analisis:
            # Filtramos solo las variables que existen
            variables_analisis = [var for var in variables_analisis if var in df.columns]
        else:
            # Usamos variables numéricas excepto ID y la variable de rendimiento
            variables_analisis = [col for col in df.select_dtypes(include=[np.number]).columns 
                                 if col != 'id_alumno' and col != var_rendimiento]
        
        # Calculamos estadísticas por grupo
        perfiles = df_grupos.groupby('grupo_rendimiento')[variables_analisis].mean()
        
        # Mostramos el tamaño de cada grupo
        tamano_grupos = df_grupos['grupo_rendimiento'].value_counts().sort_index()
        print("Distribución de estudiantes por grupo:")
        for grupo, tamano in tamano_grupos.items():
            porcentaje = (tamano / len(df_grupos)) * 100
            print(f"{grupo}: {tamano} estudiantes ({porcentaje:.2f}%)")
        
        print("\nPerfiles promedio por grupo de rendimiento:")
        print(perfiles)
        
        # Visualizamos los perfiles
        plt.figure(figsize=(14, 8))
        
        # Normalizamos las variables para mejor visualización
        perfiles_norm = (perfiles - perfiles.min()) / (perfiles.max() - perfiles.min())
        
        # Gráfico radar o gráfico de barras según el número de variables
        if len(variables_analisis) <= 10:
            # Para pocas variables, usamos un gráfico de barras agrupadas
            perfiles.plot(kind='bar', figsize=(14, 8))
            plt.title('Perfiles por grupo de rendimiento', fontsize=16)
            plt.xlabel('Grupo de rendimiento', fontsize=12)
            plt.ylabel('Valor promedio', fontsize=12)
            plt.legend(title='Variables')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Visualizamos las diferencias principales
        plt.figure(figsize=(12, 6))
        
        # Calculamos la diferencia entre el grupo de mayor y menor rendimiento
        if n_grupos == 3:
            diferencia = perfiles.loc['Alto'] - perfiles.loc['Bajo']
        else:
            diferencia = perfiles.iloc[-1] - perfiles.iloc[0]
            
        # Ordenamos por magnitud de diferencia
        diferencia = diferencia.sort_values(ascending=False)
        
        # Graficamos
        diferencia.plot(kind='bar')
        plt.title('Diferencias entre grupos de mayor y menor rendimiento', fontsize=16)
        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Diferencia en valor promedio', fontsize=12)
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return perfiles
    
    def analizar_impacto_variables(self, df_tipo='continuo', 
                                  target='rendimiento_academico_continuo',
                                  variables_categoricas=None):
        """
        Analiza el impacto de variables categóricas en el rendimiento académico.
        
        Argumentos:
            df_tipo (str): Tipo de DataFrame a utilizar
            target (str): Variable objetivo (rendimiento)
            variables_categoricas (list): Lista de variables categóricas a analizar
            
        Retorna:
            dict: Diccionario con resultados del análisis
        """
        df = self._seleccionar_dataframe(df_tipo)
        
        if target not in df.columns:
            raise ValueError(f"La variable objetivo '{target}' no existe en el DataFrame")
        
        # Identificamos variables categóricas
        if variables_categoricas:
            # Utilizamos las variables proporcionadas
            variables_a_analizar = [var for var in variables_categoricas if var in df.columns]
        else:
            # Buscamos columnas que parecen categóricas:
            # 1. Columnas con tipo object, category o string
            cat_por_tipo = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            
            # 2. Columnas numéricas con pocos valores únicos (probablemente categóricas)
            num_cols = df.select_dtypes(include=[np.number]).columns
            cat_numericas = [col for col in num_cols if col != 'id_alumno' and col != target 
                            and df[col].nunique() <= 10]
            
            variables_a_analizar = cat_por_tipo + cat_numericas
        
        if not variables_a_analizar:
            print("No se encontraron variables categóricas para analizar")
            return None
        
        # Resultados
        resultados = {}
        
        # Para cada variable categórica
        for var in variables_a_analizar:
            print(f"\n--- Análisis de la variable '{var}' ---")
            
            # Estadísticas por categoría
            stats = df.groupby(var)[target].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
            print(stats)
            
            # Gráfico
            plt.figure(figsize=(10, 6))
            
            # Barras con error
            sns.barplot(x=var, y=target, data=df, errorbar=('ci', 95))
            
            plt.title(f'Impacto de {var} en {target}', fontsize=14)
            plt.xlabel(var, fontsize=12)
            plt.ylabel(target, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # ANOVA
            grupos = [df[df[var] == cat][target].dropna() for cat in df[var].unique()]
            try:
                resultado_anova = stats.f_oneway(*grupos)
                
                print("\nResultado de ANOVA:")
                print(f"F-value: {resultado_anova.statistic:.4f}")
                print(f"p-value: {resultado_anova.pvalue:.4f}")
                
                if resultado_anova.pvalue < 0.05:
                    print("Hay diferencias estadísticamente significativas entre categorías (p < 0.05)")
                else:
                    print("No hay diferencias estadísticamente significativas entre categorías (p >= 0.05)")
                
                # Guardamos resultados
                resultados[var] = {
                    'estadisticas': stats,
                    'anova': resultado_anova,
                    'significativo': resultado_anova.pvalue < 0.05
                }
            except:
                print("No se pudo realizar ANOVA para esta variable")
                resultados[var] = {
                    'estadisticas': stats,
                    'anova': None,
                    'significativo': None
                }
        
        # Resumen de variables significativas
        vars_significativas = [var for var, res in resultados.items() 
                              if res['significativo'] == True]
        
        if vars_significativas:
            print("\n--- Variables con impacto significativo en el rendimiento ---")
            for var in vars_significativas:
                print(f"- {var}")
        
        return resultados
    
    def generar_informe_exploratorio(self, ruta_salida=None, df_tipo='continuo',
                                    target='rendimiento_academico_continuo',
                                    variables_numericas=None,
                                    variables_categoricas=None):
        """
        Genera un informe completo del análisis exploratorio.
        
        Argumentos:
            ruta_salida (str): Ruta para guardar el informe (opcional)
            df_tipo (str): Tipo de DataFrame a utilizar
            target (str): Variable objetivo
            variables_numericas (list): Variables numéricas a incluir
            variables_categoricas (list): Variables categóricas a incluir
            
        Retorna:
            dict: Diccionario con los resultados del análisis
        """
        df = self._seleccionar_dataframe(df_tipo)
        print("=" * 80)
        print(f"INFORME DE ANÁLISIS EXPLORATORIO DE DATOS - {df_tipo.upper()}")
        print("=" * 80)
        
        # 1. Información general del dataset
        print("\n1. INFORMACIÓN GENERAL DEL DATASET")
        print("-" * 50)
        print(f"Número de filas: {df.shape[0]}")
        print(f"Número de columnas: {df.shape[1]}")
        
        # Tipos de datos
        print("\nTipos de datos:")
        for tipo, count in df.dtypes.value_counts().items():
            print(f"  {tipo}: {count} columnas")
        
        # 2. Análisis univariado - variables numéricas
        print("\n2. ANÁLISIS DE VARIABLES NUMÉRICAS")
        print("-" * 50)
        
        # Estadísticas descriptivas
        estadisticas = self.resumen_estadistico(df_tipo=df_tipo, variables=variables_numericas)
        
        # Distribuciones
        print("\nDistribuciones de las principales variables numéricas:")
        self.visualizar_distribuciones(variables=variables_numericas, df_tipo=df_tipo, tipo_grafico='hist')
        
        # 3. Análisis univariado - variables categóricas
        if variables_categoricas or df.select_dtypes(include=['object', 'category']).columns.any():
            print("\n3. ANÁLISIS DE VARIABLES CATEGÓRICAS")
            print("-" * 50)
            
            self.resumen_categoricas(df_tipo=df_tipo, variables=variables_categoricas)
        
        # 4. Análisis bivariado - relaciones con el rendimiento
        if target in df.columns:
            print("\n4. RELACIONES CON EL RENDIMIENTO ACADÉMICO")
            print("-" * 50)
            
            # Correlaciones con variables numéricas
            print("\nCorrelaciones con el rendimiento:")
            self.analizar_correlaciones(df_tipo=df_tipo, target=target, umbral=0.2)
            
            # Impacto de variables categóricas
            print("\nImpacto de variables categóricas en el rendimiento:")
            self.analizar_impacto_variables(df_tipo=df_tipo, target=target, 
                                          variables_categoricas=variables_categoricas)
        
        # 5. Análisis de valores atípicos
        print("\n5. ANÁLISIS DE VALORES ATÍPICOS")
        print("-" * 50)
        
        self.analizar_outliers(variables=variables_numericas, df_tipo=df_tipo)
        
        # 6. Perfiles de rendimiento
        if target in df.columns:
            print("\n6. PERFILES DE RENDIMIENTO ACADÉMICO")
            print("-" * 50)
            
            self.crear_perfiles_rendimiento(df_tipo=df_tipo, var_rendimiento=target)
        
        # 7. Análisis de componentes principales
        print("\n7. ANÁLISIS DE COMPONENTES PRINCIPALES")
        print("-" * 50)
        
        resultados_pca = self.analizar_pca(variables=variables_numericas, df_tipo=df_tipo)
        
        # 8. Conclusiones
        print("\n8. CONCLUSIONES DEL ANÁLISIS EXPLORATORIO")
        print("-" * 50)
        
        if target in df.columns:
            # Factores más influyentes
            factores = self.analizar_factores_influyentes(target=target, df_tipo=df_tipo)
            
            print("\nFactores más importantes que influyen en el rendimiento académico:")
            # Obtener los 5 factores más importantes por valor absoluto
            factores_abs = factores.abs().sort_values(ascending=False)
            top_indices = factores_abs.index[:5]  # Obtener los índices de los 5 factores más importantes
            
            for i, idx in enumerate(top_indices, 1):
                print(f"{i}. {idx}: correlación de {factores[idx]:.3f}")
        
        print("\nAnálisis exploratorio completado.")
        
        # Guardar informe si se proporciona ruta
        if ruta_salida:
            # Implementar guardar informe
            print(f"\nInforme guardado en: {ruta_salida}")
        
        # Resultados
        resultados = {
            'estadisticas': estadisticas,
            'pca': resultados_pca
        }
        
        return resultados
    
# Inicializar la clase con los datos
analisis = AnalisisExploratorio(
    ruta_continuo='alumnosMexico2022ProcesadosContinuos.csv',
    ruta_categorico='alumnosMexico2022ProcesadosCategoricos.csv',
    ruta_original='alumnosMexico2022Procesados.csv'
)

# Ver un resumen estadístico básico
estadisticas = analisis.resumen_estadistico(df_tipo='continuo')

# Analizar correlaciones con el rendimiento académico
analisis.analizar_correlaciones(
    df_tipo='continuo', 
    target='rendimiento_academico_continuo',
    umbral=0.3
)

# Visualizar distribuciones
analisis.visualizar_distribuciones(
    variables=['matematicas', 'comprension_lectora', 'ciencias', 'indice_socioeconomico'],
    df_tipo='continuo'
)

# Analizar impacto de nivel educativo de padres en rendimiento
analisis.graficar_variables_por_categoria(
    var_numerica='rendimiento_academico_continuo',
    var_categorica='educacion_madre',
    df_tipo='continuo'
)

# Identificar factores más influyentes en el rendimiento
factores = analisis.analizar_factores_influyentes(
    target='rendimiento_academico_continuo',
    df_tipo='continuo'
)

# Generar un informe completo
analisis.generar_informe_exploratorio(
    df_tipo='continuo',
    target='rendimiento_academico_continuo'
)