import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as XGBClassifier
import shap
import os
import warnings
warnings.filterwarnings('ignore')

class ModeloRendimientoAcademico:
    def __init__(self, ruta_datos=None, df=None, carpeta_figuras='figuras'):
        """
        Inicializa el modelo con un dataset de estudiantes.
        
        Args:
            ruta_datos (str): Ruta al archivo CSV con los datos procesados
            df (pandas.DataFrame): DataFrame ya preparado (alternativa a ruta_datos)
            carpeta_figuras (str): Carpeta donde guardar las figuras generadas
        """
        if df is not None:
            self.datos = df.copy()
        elif ruta_datos is not None:
            self.datos = pd.read_csv(ruta_datos)
        else:
            raise ValueError("Debe proporcionar un DataFrame o una ruta a los datos")
            
        self.X = None  # Variables predictoras
        self.y = None  # Variable objetivo
        self.X_train = None  # Datos de entrenamiento
        self.X_test = None   # Datos de prueba
        self.y_train = None  # Etiquetas de entrenamiento
        self.y_test = None   # Etiquetas de prueba
        
        # Para los modelos entrenados
        self.modelo_kmeans = None
        self.modelo_pca = None
        self.modelo_arbol = None
        self.modelo_bosque = None
        self.modelo_logistica = None
        self.modelo_xgboost = None
        
        # Resultados de PCA
        self.componentes_principales = None
        
        # Para las etiquetas de cluster
        self.etiquetas_cluster = None
        
        # Carpeta para guardar figuras
        self.carpeta_figuras = carpeta_figuras
        if not os.path.exists(carpeta_figuras):
            os.makedirs(carpeta_figuras)
        
        print(f"Dataset cargado: {self.datos.shape[0]} estudiantes y {self.datos.shape[1]} variables")
        
    def info_datos(self):
        """Muestra información sobre el conjunto de datos"""
        print("Resumen del dataset:")
        print(self.datos.info())
        print("\nEstadísticas descriptivas:")
        print(self.datos.describe())
        print("\nPrimeras 5 filas:")
        print(self.datos.head())
        
        # Verificar valores nulos
        nulos = self.datos.isnull().sum()
        if nulos.sum() > 0:
            print("\nValores nulos por columna:")
            print(nulos[nulos > 0])
            
    def preparar_datos(self, variable_objetivo='matematicas', categorizar=False, 
                       umbral=None, variables_excluir=None):
        """
        Prepara los datos para modelado, separando variables predictoras y objetivo.
        
        Args:
            variable_objetivo (str): Nombre de la columna a predecir
            categorizar (bool): Si es True, convierte la variable objetivo en categórica
            umbral (float): Si categorizar es True, umbral para binarizar la variable
            variables_excluir (list): Lista de variables a excluir del análisis
        """
        print(f"Preparando datos con objetivo: {variable_objetivo}")
        
        # Verificar que la variable objetivo existe
        if variable_objetivo not in self.datos.columns:
            raise ValueError(f"La variable {variable_objetivo} no existe en el dataset")
        
        # Variables a excluir
        if variables_excluir is None:
            variables_excluir = ['id_alumno', variable_objetivo, 'rendimiento_general', 'matematicas', 'comprension_lectora', 'ciencias', 'matematicasCategoria' , 'comprension_lectoraCategoria', 'cienciasCategoria', 'indice_socioeconomicoCategoria']  # Por defecto excluimos el ID
        else:
            variables_excluir = list(variables_excluir) + ['id_alumno'] 
            
        # Filtrar variables existentes
        variables_excluir = [v for v in variables_excluir if v in self.datos.columns]
        
        # Seleccionar variables predictoras
        self.X = self.datos.drop(columns=[variable_objetivo] + variables_excluir)
        
        # Variable objetivo
        if categorizar:
            if umbral is None:
                # Si no se proporciona umbral, usamos la mediana
                umbral = self.datos[variable_objetivo].median()
                print(f"Usando umbral (mediana): {umbral}")
            
            # Crear variable categórica (1: alto rendimiento, 0: bajo rendimiento)
            self.y = (self.datos[variable_objetivo] >= umbral).astype(int)
            print(f"Variable objetivo categorizada: {self.y.value_counts().to_dict()}")
        else:
            self.y = self.datos[variable_objetivo]
        
        # Dividir en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42)
        
        print(f"Variables predictoras: {list(self.X.columns)}")
        print(f"Dimensiones de entrenamiento: {self.X_train.shape}")
        print(f"Dimensiones de prueba: {self.X_test.shape}")
        
        return self.X, self.y
    
    def analisis_cluster(self, n_clusters=3, algoritmo='kmeans', 
                         variables=None, estandarizar=True):
        """
        Realiza análisis de clústeres para identificar patrones naturales.
        
        Args:
            n_clusters (int): Número de clústeres a crear
            algoritmo (str): Algoritmo de clustering ('kmeans' por ahora)
            variables (list): Lista de variables a usar, si None usa todas
            estandarizar (bool): Si se deben estandarizar las variables
            
        Returns:
            numpy.ndarray: Etiquetas de clúster asignadas
        """
        print(f"Realizando análisis de clústeres con {algoritmo}, k={n_clusters}")
        
        # Seleccionar variables para clustering
        if variables is None:
            X_cluster = self.X.copy()
        else:
            # Verificar que las variables existen
            variables_validas = [v for v in variables if v in self.X.columns]
            if len(variables_validas) == 0:
                raise ValueError("Ninguna de las variables seleccionadas está en el dataset")
            X_cluster = self.X[variables_validas].copy()
        
        # Estandarizar si es necesario
        if estandarizar:
            scaler = StandardScaler()
            X_cluster_scaled = scaler.fit_transform(X_cluster)
        else:
            X_cluster_scaled = X_cluster.values
        
        # Aplicar KMeans
        if algoritmo.lower() == 'kmeans':
            self.modelo_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.etiquetas_cluster = self.modelo_kmeans.fit_predict(X_cluster_scaled)
            
            # Añadir etiquetas al DataFrame original
            self.datos['cluster'] = self.etiquetas_cluster
            
            # Calcular la inercia para diferentes valores de k (para el método del codo)
            inercias = []
            k_range = range(1, min(11, len(X_cluster) // 5))  # Máx 10 o menos según datos
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_cluster_scaled)
                inercias.append(kmeans.inertia_)
            
            # Visualizar el método del codo
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, inercias, 'o-')
            plt.xlabel('Número de clusters (k)')
            plt.ylabel('Inercia')
            plt.title('Método del codo para determinar k óptimo')
            plt.grid(True)
            plt.savefig(f'{self.carpeta_figuras}/metodo_codo_kmeans.png')
            plt.close()
        
        # Analizar características de cada clúster
        cluster_stats = self.datos.groupby('cluster').mean()
        print("\nCaracterísticas promedio por clúster:")
        print(cluster_stats)
        
        # Visualizar distribución de clústeres
        plt.figure(figsize=(10, 6))
        sns.countplot(x='cluster', data=self.datos)
        plt.title(f'Distribución de estudiantes por clúster (k={n_clusters})')
        plt.xlabel('Clúster')
        plt.ylabel('Número de estudiantes')
        plt.savefig(f'{self.carpeta_figuras}/distribucion_cluster_k{n_clusters}.png')
        plt.close()
        
        return self.etiquetas_cluster
    
    def analisis_pca(self, n_componentes=2, variables=None):
        """
        Realiza análisis de componentes principales para reducción de dimensionalidad.
        
        Args:
            n_componentes (int): Número de componentes principales a extraer
            variables (list): Lista de variables a incluir, si None usa todas
            
        Returns:
            pandas.DataFrame: DataFrame con los componentes principales
        """
        print(f"Realizando PCA con {n_componentes} componentes")
        
        # Seleccionar variables para PCA
        if variables is None:
            X_pca = self.X.copy()
        else:
            variables_validas = [v for v in variables if v in self.X.columns]
            X_pca = self.X[variables_validas].copy()
        
        # Estandarizar datos
        scaler = StandardScaler()
        X_pca_scaled = scaler.fit_transform(X_pca)
        
        # Aplicar PCA
        self.modelo_pca = PCA(n_components=n_componentes)
        self.componentes_principales = self.modelo_pca.fit_transform(X_pca_scaled)
        
        # Crear DataFrame con componentes
        pca_df = pd.DataFrame(
            data=self.componentes_principales,
            columns=[f'PC{i+1}' for i in range(n_componentes)]
        )
        
        # Varianza explicada
        varianza_explicada = self.modelo_pca.explained_variance_ratio_
        varianza_acumulada = np.cumsum(varianza_explicada)
        
        print(f"Varianza explicada por componente: {varianza_explicada}")
        print(f"Varianza acumulada: {varianza_acumulada}")
        
        # Visualizar varianza explicada
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(varianza_explicada) + 1), varianza_explicada, alpha=0.7)
        plt.step(range(1, len(varianza_acumulada) + 1), varianza_acumulada, where='mid', 
                 label='Varianza acumulada')
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% varianza')
        plt.axhline(y=0.75, color='g', linestyle='--', alpha=0.5, label='75% varianza')
        plt.xlabel('Número de componente')
        plt.ylabel('Ratio de varianza explicada')
        plt.title('Varianza explicada por componentes principales')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.carpeta_figuras}/varianza_explicada_pca.png')
        plt.close()
        
        # Visualizar contribución de cada variable
        if n_componentes >= 2:
            loadings = self.modelo_pca.components_.T * np.sqrt(self.modelo_pca.explained_variance_)
            
            # Crear DataFrame de cargas
            loading_df = pd.DataFrame(
                loadings, 
                columns=[f'PC{i+1}' for i in range(n_componentes)],
                index=X_pca.columns
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(loading_df.iloc[:, :min(n_componentes, 5)], cmap='coolwarm', annot=True, fmt=".2f")
            plt.title('Contribución de variables a los componentes principales')
            plt.tight_layout()
            plt.savefig(f'{self.carpeta_figuras}/contribucion_variables_pca.png')
            plt.close()
            
            # Visualización en 2D si tenemos al menos 2 componentes
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                       c=self.y if hasattr(self.y, 'cat') else self.y, 
                       alpha=0.7, cmap='viridis')
            plt.colorbar(scatter, label='Valor objetivo')
            plt.xlabel(f'PC1 ({varianza_explicada[0]:.2%} varianza)')
            plt.ylabel(f'PC2 ({varianza_explicada[1]:.2%} varianza)')
            plt.title('Proyección de datos en primeros 2 componentes principales')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.carpeta_figuras}/proyeccion_pca_2d.png')
            plt.close()
            
        return pca_df
    
    def entrenar_arbol_decision(self, max_depth=5, min_samples_split=2, 
                               criterio='gini', visualizar=True):
        """
        Entrena un modelo de árbol de decisión.
        
        Args:
            max_depth (int): Profundidad máxima del árbol
            min_samples_split (int): Mínimo número de muestras para dividir
            criterio (str): Criterio de división ('gini' o 'entropy')
            visualizar (bool): Si se debe visualizar el árbol
            
        Returns:
            float: Precisión del modelo en conjunto de prueba
        """
        print(f"Entrenando árbol de decisión (max_depth={max_depth}, criterio={criterio})")
        
        # Verificar que tenemos datos preparados
        if self.X_train is None or self.y_train is None:
            raise ValueError("Primero debe preparar los datos con el método preparar_datos()")
        
        # Crear y entrenar modelo
        self.modelo_arbol = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterio,
            random_state=42
        )
        self.modelo_arbol.fit(self.X_train, self.y_train)
        
        # Evaluar en conjunto de prueba
        y_pred = self.modelo_arbol.predict(self.X_test)
        precision = accuracy_score(self.y_test, y_pred)
        
        print(f"Precisión del árbol: {precision:.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(self.y_test, y_pred))
        
        # Matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de confusión - Árbol de decisión')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(f'{self.carpeta_figuras}/matriz_confusion_arbol.png')
        plt.close()
        
        # Visualizar árbol
        if visualizar and max_depth <= 5:  # Limitar visualización a árboles pequeños
            plt.figure(figsize=(20, 10))
            plot_tree(self.modelo_arbol, filled=True, feature_names=self.X.columns, 
                     class_names=[str(c) for c in self.modelo_arbol.classes_])
            plt.title(f'Árbol de decisión (profundidad={max_depth})')
            plt.tight_layout()
            plt.savefig(f'{self.carpeta_figuras}/arbol_decision_depth{max_depth}.png')
            plt.close()
        
        # Importancia de características
        importancia = pd.DataFrame({
            'característica': self.X.columns,
            'importancia': self.modelo_arbol.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importancia', y='característica', data=importancia)
        plt.title('Importancia de variables - Árbol de decisión')
        plt.tight_layout()
        plt.savefig(f'{self.carpeta_figuras}/importancia_variables_arbol.png')
        plt.close()
        
        return precision
    
    def entrenar_random_forest(self, n_estimators=100, max_depth=None, 
                              min_samples_split=2, n_jobs=-1):
        """
        Entrena un modelo de Random Forest.
        
        Args:
            n_estimators (int): Número de árboles
            max_depth (int): Profundidad máxima de árboles (None para ilimitado)
            min_samples_split (int): Mínimo de muestras para dividir
            n_jobs (int): Número de trabajos paralelos (-1 para todos los cores)
            
        Returns:
            float: Precisión del modelo en conjunto de prueba
        """
        print(f"Entrenando Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")
        
        # Verificar que tenemos datos preparados
        if self.X_train is None or self.y_train is None:
            raise ValueError("Primero debe preparar los datos con el método preparar_datos()")
        
        # Crear y entrenar modelo
        self.modelo_bosque = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=n_jobs
        )
        self.modelo_bosque.fit(self.X_train, self.y_train)
        
        # Evaluar en conjunto de prueba
        y_pred = self.modelo_bosque.predict(self.X_test)
        precision = accuracy_score(self.y_test, y_pred)
        
        print(f"Precisión del Random Forest: {precision:.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(self.y_test, y_pred))
        
        # Matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de confusión - Random Forest')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(f'{self.carpeta_figuras}/matriz_confusion_random_forest.png')
        plt.close()
        
        # Importancia de características
        importancia = pd.DataFrame({
            'característica': self.X.columns,
            'importancia': self.modelo_bosque.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importancia', y='característica', data=importancia)
        plt.title('Importancia de variables - Random Forest')
        plt.tight_layout()
        plt.savefig(f'{self.carpeta_figuras}/importancia_variables_random_forest.png')
        plt.close()
        
        return precision
    
    def entrenar_xgboost(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                        regularizacion_l1=0, regularizacion_l2=1):
        """
        Entrena un modelo XGBoost.
        
        Args:
            n_estimators (int): Número de árboles
            learning_rate (float): Tasa de aprendizaje
            max_depth (int): Profundidad máxima de árboles
            regularizacion_l1 (float): Regularización L1 (alpha)
            regularizacion_l2 (float): Regularización L2 (lambda)
            
        Returns:
            float: Precisión del modelo en conjunto de prueba
        """
        print(f"Entrenando XGBoost (n_estimators={n_estimators}, learning_rate={learning_rate})")
        
        # Verificar que tenemos datos preparados
        if self.X_train is None or self.y_train is None:
            raise ValueError("Primero debe preparar los datos con el método preparar_datos()")
        
        # Crear y entrenar modelo
        self.modelo_xgboost = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            reg_alpha=regularizacion_l1,
            reg_lambda=regularizacion_l2,
            random_state=42
        )
        self.modelo_xgboost.fit(self.X_train, self.y_train)
        
        # Evaluar en conjunto de prueba
        y_pred = self.modelo_xgboost.predict(self.X_test)
        precision = accuracy_score(self.y_test, y_pred)
        
        print(f"Precisión del XGBoost: {precision:.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(self.y_test, y_pred))
        
        # Matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de confusión - XGBoost')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(f'{self.carpeta_figuras}/matriz_confusion_xgboost.png')
        plt.close()
        
        # Importancia de características
        importancia = pd.DataFrame({
            'característica': self.X.columns,
            'importancia': self.modelo_xgboost.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importancia', y='característica', data=importancia)
        plt.title('Importancia de variables - XGBoost')
        plt.tight_layout()
        plt.savefig(f'{self.carpeta_figuras}/importancia_variables_xgboost.png')
        plt.close()
        
        return precision
    
    def entrenar_regresion_logistica(self, regularizacion=1.0, solver='liblinear'):
        """
        Entrena un modelo de regresión logística.
        
        Args:
            regularizacion (float): Parámetro C de regularización (inverso)
            solver (str): Algoritmo de optimización
            
        Returns:
            float: Precisión del modelo en conjunto de prueba
        """
        print(f"Entrenando regresión logística (C={regularizacion}, solver={solver})")
        
        # Verificar que tenemos datos preparados
        if self.X_train is None or self.y_train is None:
            raise ValueError("Primero debe preparar los datos con el método preparar_datos()")
        
        # Crear pipeline con estandarización
        self.modelo_logistica = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(C=regularizacion, solver=solver, random_state=42))
        ])
        self.modelo_logistica.fit(self.X_train, self.y_train)
        
        # Evaluar en conjunto de prueba
        y_pred = self.modelo_logistica.predict(self.X_test)
        precision = accuracy_score(self.y_test, y_pred)
        
        print(f"Precisión de la regresión logística: {precision:.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(self.y_test, y_pred))
        
        # Matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de confusión - Regresión logística')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(f'{self.carpeta_figuras}/matriz_confusion_regresion_logistica.png')
        plt.close()
        
        # Coeficientes (para variables estandarizadas)
        coeficientes = pd.DataFrame({
            'característica': self.X.columns,
            'coeficiente': self.modelo_logistica.named_steps['logistic'].coef_[0]
        }).sort_values('coeficiente', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='coeficiente', y='característica', data=coeficientes)
        plt.title('Coeficientes - Regresión logística')
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{self.carpeta_figuras}/coeficientes_regresion_logistica.png')
        plt.close()
        
        return precision
    
    def comparar_modelos(self, cv=5):
        """
        Compara todos los modelos entrenados usando validación cruzada.
        
        Args:
            cv (int): Número de folds para validación cruzada
            
        Returns:
            pandas.DataFrame: Comparativa de precisión de modelos
        """
        print(f"Comparando modelos con validación cruzada ({cv} folds)")
        
        # Verificar que tenemos datos preparados
        if self.X is None or self.y is None:
            raise ValueError("Primero debe preparar los datos con el método preparar_datos()")
        
        # Definir modelos a comparar
        modelos = {
            'Árbol de decisión': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'Regresión logística': Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(random_state=42))
            ])
        }
        
        # Evaluar modelos con validación cruzada
        resultados = {}
        for nombre, modelo in modelos.items():
            scores = cross_val_score(modelo, self.X, self.y, cv=cv, scoring='accuracy')
            resultados[nombre] = {
                'media': scores.mean(),
                'desviacion': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
        
        # Crear DataFrame con resultados
        df_resultados = pd.DataFrame(resultados).T
        
        # Visualizar comparativa
        plt.figure(figsize=(12, 6))
        sns.barplot(x=df_resultados.index, y='media', data=df_resultados)
        plt.errorbar(x=range(len(df_resultados)), y=df_resultados['media'], 
                    yerr=df_resultados['desviacion'], fmt='o', color='black')
        plt.title(f'Comparativa de modelos - Validación cruzada {cv} folds')
        plt.ylabel('Precisión media')
        plt.ylim(0.5, 1.0)  # Ajustar según resultados
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.carpeta_figuras}/comparativa_modelos_cv{cv}.png')
        plt.close()
        
        return df_resultados
    
    def optimizar_hiperparametros(self, modelo='arbol'):
        """
        Optimiza hiperparámetros para el modelo seleccionado.
        
        Args:
            modelo (str): Modelo a optimizar ('arbol', 'bosque', 'xgboost', 'logistica')
            
        Returns:
            dict: Mejores hiperparámetros encontrados
        """
        print(f"Optimizando hiperparámetros para modelo: {modelo}")
        
        # Verificar que tenemos datos preparados
        if self.X_train is None or self.y_train is None:
            raise ValueError("Primero debe preparar los datos con el método preparar_datos()")
        
        # Definir parámetros de búsqueda según modelo
        if modelo == 'arbol':
            estimador = DecisionTreeClassifier(random_state=42)
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        elif modelo == 'bosque':
            estimador = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }
        elif modelo == 'xgboost':
            estimador = XGBClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif modelo == 'logistica':
            estimador = Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(random_state=42))
            ])
            param_grid = {
                'logreg__C': [0.01, 0.1, 1.0, 10.0],
                'logreg__solver': ['liblinear', 'saga']
            }
        else:
            raise ValueError("Modelo no reconocido. Opciones: 'arbol', 'bosque', 'xgboost', 'logistica'")
        
        # Realizar búsqueda de hiperparámetros
        grid_search = GridSearchCV(
            estimator=estimador,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Mostrar resultados
        print(f"Mejores hiperparámetros: {grid_search.best_params_}")
        print(f"Mejor precisión: {grid_search.best_score_:.4f}")
        
        # Guardar mejor modelo
        if modelo == 'arbol':
            self.modelo_arbol = grid_search.best_estimator_
        elif modelo == 'bosque':
            self.modelo_bosque = grid_search.best_estimator_
        elif modelo == 'xgboost':
            self.modelo_xgboost = grid_search.best_estimator_
        elif modelo == 'logistica':
            self.modelo_logistica = grid_search.best_estimator_
        
        # Evaluar en conjunto de prueba
        y_pred = grid_search.best_estimator_.predict(self.X_test)
        precision = accuracy_score(self.y_test, y_pred)
        
        print(f"Precisión en conjunto de prueba: {precision:.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(self.y_test, y_pred))
        
        return grid_search.best_params_
    
    def interpretar_modelo_shap(self, modelo='bosque', muestra_size=100):
        """
        Interpreta el modelo usando valores SHAP (SHapley Additive exPlanations).
        
        Args:
            modelo (str): Modelo a interpretar ('arbol', 'bosque', 'xgboost')
            muestra_size (int): Tamaño de la muestra para calcular valores SHAP
            
        Returns:
            shap.Explanation: Objeto de explicación SHAP
        """
        print(f"Interpretando modelo {modelo} con SHAP")
        
        # Seleccionar modelo a interpretar
        if modelo == 'arbol' and self.modelo_arbol is not None:
            modelo_a_interpretar = self.modelo_arbol
        elif modelo == 'bosque' and self.modelo_bosque is not None:
            modelo_a_interpretar = self.modelo_bosque
        elif modelo == 'xgboost' and self.modelo_xgboost is not None:
            modelo_a_interpretar = self.modelo_xgboost
        else:
            raise ValueError(f"Modelo {modelo} no disponible o no entrenado aún")
        
        # Tomar una muestra de datos para explicación
        X_muestra = self.X_test.sample(min(muestra_size, len(self.X_test)), random_state=42)
        
        # Inicializar explainer según tipo de modelo
        if modelo == 'xgboost':
            explainer = shap.TreeExplainer(modelo_a_interpretar)
        else:
            explainer = shap.TreeExplainer(modelo_a_interpretar)
        
        # Calcular valores SHAP
        shap_values = explainer.shap_values(X_muestra)
        
        # Visualizaciones SHAP
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_muestra, feature_names=self.X.columns)
        plt.title(f'Impacto de variables en predicciones - {modelo.capitalize()}')
        plt.tight_layout()
        plt.savefig(f'{self.carpeta_figuras}/shap_summary_{modelo}.png')
        plt.close()
        
        # Para clasificación, mostrar impacto en la clase positiva
        if isinstance(shap_values, list) and len(shap_values) > 1:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[1], X_muestra, feature_names=self.X.columns, plot_type='bar')
            plt.title(f'Importancia media de variables (clase positiva) - {modelo.capitalize()}')
            plt.tight_layout()
            plt.savefig(f'{self.carpeta_figuras}/shap_importancia_media_{modelo}.png')
            plt.close()
        
        return shap_values
    
    def analizar_factores_influyentes(self, df_resultados=None):
        """
        Analiza y sintetiza los factores más influyentes en el rendimiento académico
        basándose en todos los modelos entrenados.
        
        Args:
            df_resultados (pandas.DataFrame): DataFrame opcional con resultados para incluir
            
        Returns:
            pandas.DataFrame: Síntesis de factores influyentes
        """
        print("Analizando factores influyentes en el rendimiento académico")
        
        factores = {}
        
        # Recopilar factores de árbol de decisión
        if self.modelo_arbol is not None:
            importancia_arbol = pd.Series(
                self.modelo_arbol.feature_importances_,
                index=self.X.columns
            ).sort_values(ascending=False)
            factores['Árbol de decisión'] = importancia_arbol
        
        # Recopilar factores de Random Forest
        if self.modelo_bosque is not None:
            importancia_bosque = pd.Series(
                self.modelo_bosque.feature_importances_,
                index=self.X.columns
            ).sort_values(ascending=False)
            factores['Random Forest'] = importancia_bosque
        
        # Recopilar factores de XGBoost
        if self.modelo_xgboost is not None:
            importancia_xgboost = pd.Series(
                self.modelo_xgboost.feature_importances_,
                index=self.X.columns
            ).sort_values(ascending=False)
            factores['XGBoost'] = importancia_xgboost
        
        # Recopilar coeficientes de regresión logística (en valor absoluto)
        if self.modelo_logistica is not None:
            if hasattr(self.modelo_logistica, 'named_steps'):
                coef_logistica = pd.Series(
                    np.abs(self.modelo_logistica.named_steps['logistic'].coef_[0]),
                    index=self.X.columns
                ).sort_values(ascending=False)
            else:
                coef_logistica = pd.Series(
                    np.abs(self.modelo_logistica.coef_[0]),
                    index=self.X.columns
                ).sort_values(ascending=False)
            factores['Regresión logística'] = coef_logistica
        
        if not factores:
            print("No hay modelos entrenados para analizar")
            return None
        
        # Crear DataFrame con todos los factores
        df_factores = pd.DataFrame(factores)
        
        # Normalizar valores (para comparación justa)
        for col in df_factores.columns:
            df_factores[col] = df_factores[col] / df_factores[col].sum()
        
        # Calcular importancia promedio
        df_factores['Promedio'] = df_factores.mean(axis=1)
        
        # Ordenar por importancia promedio
        df_factores = df_factores.sort_values('Promedio', ascending=False)
        
        # Visualizar factores más importantes
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_factores.head(10), annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title('Factores más influyentes en el rendimiento académico')
        plt.tight_layout()
        plt.savefig(f'{self.carpeta_figuras}/factores_influyentes_heatmap.png')
        plt.close()
        
        # Gráfico de barras con los 5 factores principales
        plt.figure(figsize=(12, 6))
        df_factores.head(5)['Promedio'].plot(kind='bar')
        plt.title('Top 5 factores que influyen en el rendimiento académico')
        plt.ylabel('Importancia relativa')
        plt.tight_layout()
        plt.savefig(f'{self.carpeta_figuras}/top5_factores_influyentes.png')
        plt.close()
        
        return df_factores

# Cargar y explorar los datos
modelo = ModeloRendimientoAcademico('../ETL/Transformar/alumnosMexico2022Procesados.csv', carpeta_figuras='figuras_analisis')
modelo.info_datos()


# Primero preparar los datos
modelo.preparar_datos(variable_objetivo='matematicas', categorizar=True)

# Análisis exploratorio (no supervisado) de los datos
modelo.analisis_cluster(n_clusters=5)
modelo.analisis_pca(n_componentes=5)

# Preparar datos para modelado supervisado (predecir 'matematicas')
modelo.preparar_datos(variable_objetivo='matematicas', categorizar=True)

# Entrenar diferentes modelos
modelo.entrenar_arbol_decision(max_depth=4)
modelo.entrenar_random_forest(n_estimators=100)
modelo.entrenar_xgboost()
modelo.entrenar_regresion_logistica()

# Comparar modelos
resultados = modelo.comparar_modelos(cv=5)

# Optimizar hiperparámetros del mejor modelo
mejores_params = modelo.optimizar_hiperparametros(modelo='bosque')

# Interpretar modelo con SHAP
modelo.interpretar_modelo_shap(modelo='bosque')

# Analizar factores influyentes
factores = modelo.analizar_factores_influyentes()
print("Top 5 factores que influyen en el rendimiento académico:")
print(factores.head(5))