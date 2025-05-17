import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from sklearn.pipeline import Pipeline

class ModeloPredictivo:
    """
    Clase para la creación y evaluación de modelos predictivos del rendimiento académico.
    """
    
    def __init__(self, ruta_datos):
        """
        Inicializa el modelo predictivo con la ruta del archivo de datos.
        
        Args:
            ruta_datos (str): Ruta al archivo CSV con los datos procesados
        """
        self.ruta_datos = ruta_datos
        self.datos = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modelo = None
        self.resultados = {}
    
    def cargar_datos(self):
        """
        Carga los datos desde el archivo CSV y los prepara para el modelo.
        
        Returns:
            pandas.DataFrame: El dataset cargado
        """
        try:
            self.datos = pd.read_csv(self.ruta_datos)
            print(f"Datos cargados exitosamente. Dimensiones: {self.datos.shape}")
            return self.datos
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return None
    
    def explorar_datos(self):
        """
        Realiza un análisis exploratorio básico de los datos.
        
        Returns:
            dict: Estadísticas descriptivas de los datos
        """
        if self.datos is None:
            print("No hay datos cargados. Llame primero a cargar_datos()")
            return None
        
        # Información básica del dataset
        print("\n===== INFORMACIÓN DEL DATASET =====")
        print(f"Número de filas: {self.datos.shape[0]}")
        print(f"Número de columnas: {self.datos.shape[1]}")
        
        # Variables en el dataset
        print("\n===== VARIABLES DISPONIBLES =====")
        print(self.datos.columns.tolist())
        
        # Estadísticas descriptivas
        print("\n===== ESTADÍSTICAS DESCRIPTIVAS =====")
        estadisticas = self.datos.describe(include='all')
        print(estadisticas)
        
        # Valores faltantes
        print("\n===== VALORES FALTANTES =====")
        faltantes = self.datos.isnull().sum()
        print(faltantes[faltantes > 0])
        
        # Distribución de la variable objetivo (si es categórica)
        if 'rendimiento_general_categoria' in self.datos.columns:
            print("\n===== DISTRIBUCIÓN DE LA VARIABLE OBJETIVO =====")
            distribucion = self.datos['rendimiento_general_categoria'].value_counts(normalize=True) * 100
            print(distribucion)
            
            # Visualizar distribución
            plt.figure(figsize=(10, 6))
            sns.countplot(x='rendimiento_general_categoria', data=self.datos, 
                          order=self.datos['rendimiento_general_categoria'].value_counts().index)
            plt.title('Distribución de Rendimiento Académico')
            plt.xlabel('Categoría de Rendimiento')
            plt.ylabel('Cantidad de Estudiantes')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('distribucion_rendimiento.png')
            plt.close()
        
        return {
            'estadisticas': estadisticas,
            'faltantes': faltantes,
            'distribucion': distribucion if 'rendimiento_general_categoria' in self.datos.columns else None
        }
    
    def detectar_outliers(self):
        """
        Detecta outliers en las variables numéricas del dataset.
        
        Returns:
            dict: Diccionario con los outliers detectados
        """
        if self.datos is None:
            print("No hay datos cargados. Llame primero a cargar_datos()")
            return None
        
        # Seleccionar solo columnas numéricas
        numericas = self.datos.select_dtypes(include=['int64', 'float64'])
        
        # Detectar outliers usando el método IQR
        outliers = {}
        for col in numericas.columns:
            Q1 = numericas[col].quantile(0.25)
            Q3 = numericas[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            outliers[col] = numericas[(numericas[col] < limite_inferior) | (numericas[col] > limite_superior)]
        
        print("Outliers detectados:")
        for col, df_outliers in outliers.items():
            print(f"{col}: {df_outliers.shape[0]} outliers")
        
        return outliers
    
    def analizar_correlaciones(self):
        """
        Analiza las correlaciones entre las variables numéricas.
        
        Returns:
            pandas.DataFrame: Matriz de correlación
        """
        if self.datos is None:
            print("No hay datos cargados. Llame primero a cargar_datos()")
            return None
        
        # Seleccionar solo columnas numéricas
        numericas = self.datos.select_dtypes(include=['int64', 'float64'])
        
        # Calcular matriz de correlación
        correlacion = numericas.corr()
        
        # Visualizar matriz de correlación
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Matriz de Correlación de Variables Numéricas')
        plt.tight_layout()
        plt.savefig('correlacion_variables.png')
        plt.close()
        
        return correlacion
    
    def preparar_datos_clasificacion(self, variable_objetivo='rendimiento_general_categoria', 
                                  test_size=0.30, random_state=0):
        """
        Prepara los datos para un modelo de clasificación.
        
        Args:
            variable_objetivo (str): Nombre de la columna objetivo para la clasificación
            test_size (float): Proporción del conjunto de prueba (entre 0 y 1)
            random_state (int): Semilla para reproducibilidad
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.datos is None:
            print("No hay datos cargados. Llame primero a cargar_datos()")
            return None
        
        if variable_objetivo not in self.datos.columns:
            print(f"La variable objetivo '{variable_objetivo}' no está en el dataset")
            return None
        
        # Excluir variables que no deben ser predictores
        excluir = ['id_alumno', variable_objetivo, 'rendimiento_general', 'matematicas', 'comprension_lectora', 'ciencias', 'matematicasCategoria' , 'comprension_lectoraCategoria', 'cienciasCategoria', 'indice_socioeconomico']
        predictores = [col for col in self.datos.columns if col not in excluir]
        
        # Convertir variables categóricas a dummy si existen
        categoricas = [col for col in predictores if self.datos[col].dtype == 'object']
        if categoricas:
            self.datos = pd.get_dummies(self.datos, columns=categoricas, drop_first=False)
            # Actualizar lista de predictores después de crear variables dummy
            predictores = [col for col in self.datos.columns if col not in excluir]
        
        # Preparar matrices de características y variable objetivo
        self.X = self.datos[predictores]
        self.y = self.datos[variable_objetivo]
        
        # Dividir en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Datos preparados: {self.X_train.shape[0]} muestras de entrenamiento, {self.X_test.shape[0]} muestras de prueba")
        print(f"Variables predictoras: {len(predictores)}")
        print(f"Variables predictoras: {predictores}")
        print(f"Variable objetivo: {variable_objetivo}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preparar_datos_con_transformacion(self, variable_objetivo='rendimiento_general_categoria', 
                                    test_size=0.30, random_state=0,
                                    aplicar_polinomicas=True, 
                                    aplicar_interacciones=True,
                                    aplicar_normalizacion=True):
        """
        Prepara los datos aplicando transformaciones avanzadas sin eliminar variables.
        
        Args:
            variable_objetivo (str): Nombre de la columna objetivo
            test_size (float): Proporción del conjunto de prueba
            random_state (int): Semilla para reproducibilidad
            aplicar_polinomicas (bool): Si se deben crear características polinómicas
            aplicar_interacciones (bool): Si se deben crear interacciones entre variables
            aplicar_normalizacion (bool): Si se deben normalizar variables numéricas
                
        Returns:
            tuple: (X_train_transformed, X_test_transformed, y_train, y_test)
        """
        
        if self.datos is None:
            print("No hay datos cargados. Llame primero a cargar_datos()")
            return None
        
        if variable_objetivo not in self.datos.columns:
            print(f"La variable objetivo '{variable_objetivo}' no está en el dataset")
            return None
        
        # Excluir variables que no deben ser predictores
        excluir = ['id_alumno', variable_objetivo, 'rendimiento_general', 'matematicas', 
                'comprension_lectora', 'ciencias', 'matematicasCategoria', 
                'comprension_lectoraCategoria', 'cienciasCategoria', 'indice_socioeconomico']
        predictores = [col for col in self.datos.columns if col not in excluir]
        
        # Identificar tipos de columnas
        X = self.datos[predictores]
        y = self.datos[variable_objetivo]
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Guardar las columnas originales para el modelo
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Identificar columnas numéricas y categóricas
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Crear pipeline de transformaciones
        transformers = []
        
        # Pipeline para variables numéricas
        num_steps = []
        if aplicar_normalizacion:
            num_steps.append(('scaler', StandardScaler()))
        if aplicar_polinomicas:
            num_steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=aplicar_interacciones,
                                                include_bias=False)))
        
        if num_steps:
            transformers.append(('num', Pipeline(steps=num_steps), num_cols))
        
        # Si hay transformaciones que aplicar, crear el ColumnTransformer
        if transformers:
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'  # Mantener variables no transformadas
            )
            
            # Aplicar transformaciones
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # Guardar el preprocesador para futuras transformaciones
            self.preprocessor = preprocessor
            
            # Si las transformaciones generan una matriz numpy, convertir a DataFrame
            if isinstance(X_train_transformed, np.ndarray):
                # Obtener nombres de características transformadas
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    # Fallback para versiones anteriores de sklearn
                    feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]
                
                X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
                X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
            
            print(f"Datos transformados: {X_train_transformed.shape[1]} características creadas")
            
            return X_train_transformed, X_test_transformed, y_train, y_test
        else:
            print("No se aplicaron transformaciones. Utilizando variables originales.")
            return X_train, X_test, y_train, y_test
    
    def entrenar_modelo_clasificacion(self, modelo='random_forest', hiperparametros=None):
        """
        Entrena un modelo de clasificación con los datos preparados.
        
        Args:
            modelo (str): Tipo de modelo a entrenar ('random_forest', 'decision_tree', 'gradient_boosting')
            hiperparametros (dict): Hiperparámetros para el modelo seleccionado
            
        Returns:
            object: Modelo entrenado
        """
        if self.X_train is None or self.y_train is None:
            print("Datos no preparados. Llame primero a preparar_datos_clasificacion()")
            return None
        
        # Configurar hiperparámetros por defecto si no se proporcionan
        if hiperparametros is None:
            hiperparametros = {}
        
        # Seleccionar modelo
        if modelo == 'random_forest':
            class_weights = {
    'Alto': 1.0,
    'Bajo': 1.0,
    'Medio': 2.0  # Mayor peso para la clase con peor rendimiento
}

            parametros = {
                'n_estimators': 1000,
                'max_depth': None,
                'min_samples_split': 50,
                'min_samples_leaf': 50,
                'max_features': None,
                'bootstrap': False,
                'criterion': 'entropy',
                'class_weight': 'balanced',
                'random_state': 0,
            }
            # Actualizar con hiperparámetros proporcionados
            parametros.update(hiperparametros)
            self.modelo = RandomForestClassifier(**parametros)
            
        elif modelo == 'decision_tree':
            parametros = {
                'max_depth': 8,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 0
            }
            parametros.update(hiperparametros)
            self.modelo = DecisionTreeClassifier(**parametros)
            
        elif modelo == 'gradient_boosting':
            parametros = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 0
            }
            parametros.update(hiperparametros)
            self.modelo = GradientBoostingClassifier(**parametros)
        elif modelo == 'naiveBayes':
            
            self.modelo = GaussianNB()
            
        else:
            print(f"Modelo '{modelo}' no reconocido")
            return None
        
        # Entrenar modelo
        print(f"Entrenando modelo {modelo}...")
        self.modelo.fit(self.X_train, self.y_train)
        print("Modelo entrenado exitosamente")
        
        return self.modelo
    
    def entrenar_modelo_avanzado(self, modelo='xgboost', hiperparametros=None, manejar_categoricas='auto'):
        """
        Entrena un modelo avanzado que maneja naturalmente la importancia de características.
        
        Args:
            modelo (str): Tipo de modelo avanzado ('xgboost', 'lightgbm', 'catboost', 'stacking')
            hiperparametros (dict): Hiperparámetros para el modelo seleccionado
            manejar_categoricas (str): Cómo manejar variables categóricas ('auto', 'convert', 'enable')
                
        Returns:
            object: Modelo entrenado
        """
        if self.X_train is None or self.y_train is None:
            print("Datos no preparados. Llame primero a preparar_datos_clasificacion()")
            return None
        
        # Configurar hiperparámetros por defecto si no se proporcionan
        if hiperparametros is None:
            hiperparametros = {}
        
        # Verificar si hay columnas categóricas
        columnas_categoricas = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if columnas_categoricas:
            print(f"Detectadas {len(columnas_categoricas)} columnas categóricas: {columnas_categoricas}")
            
            # Convertir columnas categóricas según la estrategia elegida
            if manejar_categoricas == 'convert' or (manejar_categoricas == 'auto' and modelo != 'catboost'):
                print("Convirtiendo columnas categóricas a numéricas (one-hot encoding)...")
                
                # Guardar una copia del DataFrame original
                X_train_original = self.X_train.copy()
                X_test_original = self.X_test.copy()
                
                # Aplicar one-hot encoding
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.compose import ColumnTransformer
                
                # Preparar transformer
                transformer = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), columnas_categoricas)
                    ],
                    remainder='passthrough'
                )
                
                # Ajustar y transformar
                self.X_train = transformer.fit_transform(self.X_train)
                self.X_test = transformer.transform(self.X_test)
                
                # Obtener nuevos nombres de columnas
                nombres_columnas = []
                
                # Primero las columnas one-hot encoded
                for i, nombre in enumerate(columnas_categoricas):
                    categorias = transformer.transformers_[0][1].categories_[i]
                    for cat in categorias:
                        nombres_columnas.append(f"{nombre}_{cat}")
                
                # Luego las columnas numéricas originales
                columnas_numericas = X_train_original.select_dtypes(exclude=['object', 'category']).columns.tolist()
                nombres_columnas.extend(columnas_numericas)
                
                # Convertir a DataFrame
                self.X_train = pd.DataFrame(self.X_train, columns=nombres_columnas)
                self.X_test = pd.DataFrame(self.X_test, columns=nombres_columnas)
                
                print(f"Datos transformados: de {len(X_train_original.columns)} a {len(self.X_train.columns)} columnas")
        
        # Convertir etiquetas de clase a numéricas si son strings
        from sklearn.preprocessing import LabelEncoder
        
        # Verificar si las etiquetas son strings
        if self.y_train.dtype == 'object' or pd.api.types.is_categorical_dtype(self.y_train):
            print("Convirtiendo etiquetas de clase a valores numéricos...")
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(self.y_train)
            y_test_encoded = self.label_encoder.transform(self.y_test)
            
            # Guardar mapeo para referencia
            self.class_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
            print(f"Mapeo de clases: {self.class_mapping}")
        else:
            # Ya son numéricas
            y_train_encoded = self.y_train
            y_test_encoded = self.y_test
            self.label_encoder = None
        
        # Verificar clases únicas para configuraciones específicas
        clases_unicas = np.unique(y_train_encoded)
        num_clases = len(clases_unicas)
        
        try:
            if modelo == 'xgboost':
                import xgboost as xgb
                
                # Configuración base para multiclase o binario
                parametros = {
                    'objective': 'multi:softprob' if num_clases > 2 else 'binary:logistic',
                    'num_class': num_clases if num_clases > 2 else None,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'scale_pos_weight': 1,
                    'random_state': 42,
                    'eval_metric': 'mlogloss' if num_clases > 2 else 'logloss'
                }
                
                # Si hay columnas categóricas y elegimos habilitarlas
                if columnas_categoricas and manejar_categoricas == 'enable':
                    parametros['enable_categorical'] = True
                
                # Eliminar parámetros no necesarios para binario
                if num_clases <= 2:
                    parametros.pop('num_class')
                
                # Actualizar con hiperparámetros proporcionados
                parametros.update(hiperparametros)
                
                # Filtrar None values (XGBoost no los acepta)
                parametros = {k: v for k, v in parametros.items() if v is not None}
                    
                self.modelo = xgb.XGBClassifier(**parametros)
            
            elif modelo == 'lightgbm':
                import lightgbm as lgb
                
                parametros = {
                    'objective': 'multiclass' if num_clases > 2 else 'binary',
                    'num_class': num_clases if num_clases > 2 else None,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'max_depth': -1,  # -1 significa sin límite
                    'min_child_samples': 20,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'random_state': 42,
                    'verbose': -1
                }
                
                # Si hay columnas categóricas, agregar sus índices
                if columnas_categoricas and manejar_categoricas != 'convert':
                    parametros['categorical_feature'] = columnas_categoricas
                
                # Eliminar parámetros no necesarios para binario
                if num_clases <= 2:
                    parametros.pop('num_class')
                    
                # Actualizar con hiperparámetros proporcionados
                parametros.update(hiperparametros)
                
                # Filtrar None values
                parametros = {k: v for k, v in parametros.items() if v is not None}
                    
                self.modelo = lgb.LGBMClassifier(**parametros)
            
            # Resto del código igual que antes...
                
            # Entrenar modelo
            if modelo != 'catboost':
                print(f"Entrenando modelo {modelo}...")
                self.modelo.fit(self.X_train, y_train_encoded)
                print("Modelo entrenado exitosamente")
            
            return self.modelo
            
        except ImportError as e:
            print(f"Error: La librería para {modelo} no está instalada.")
            print(f"Instale con: pip install {modelo}")
            print(f"Error específico: {e}")
            return None
    
    def evaluar_modelo(self):
        """
        Evalúa el rendimiento del modelo en el conjunto de prueba.
        
        Returns:
            dict: Métricas de evaluación del modelo
        """
        if self.modelo is None:
            print("No hay modelo entrenado. Llame primero a entrenar_modelo_clasificacion()")
            return None
        
        # Verificar si se usó un codificador de etiquetas
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            y_test_encoded = self.label_encoder.transform(self.y_test)
        else:
            y_test_encoded = self.y_test
        
        # Predecir en conjunto de prueba
        y_pred_encoded = self.modelo.predict(self.X_test)
        
        # Convertir predicciones a etiquetas originales si se usó un codificador
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = y_pred_encoded
        
        # Calcular métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        print(f"Exactitud (Accuracy): {accuracy:.4f}")
        print("\nInforme de Clasificación:")
        print(classification_report(self.y_test, y_pred))
        
        # Guardar resultados
        self.resultados = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
        
        # Visualizar matriz de confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(self.y_test),
                yticklabels=np.unique(self.y_test))
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión')
        plt.tight_layout()
        plt.savefig('matriz_confusion.png')
        plt.close()
        
        return self.resultados
    
    def importancia_variables(self):
        """
        Muestra la importancia de las variables en el modelo.
        
        Returns:
            pandas.DataFrame: Importancia de cada variable
        """
        if self.modelo is None or not hasattr(self.modelo, 'feature_importances_'):
            print("El modelo no tiene atributo feature_importances_")
            return None
        
        # Obtener importancia de variables
        importancias = self.modelo.feature_importances_
        
        # Crear DataFrame con nombres de variables e importancias
        df_importancias = pd.DataFrame({
            'Variable': self.X.columns,
            'Importancia': importancias
        })
        
        # Ordenar por importancia
        df_importancias = df_importancias.sort_values('Importancia', ascending=False)
        
        # Visualizar importancia
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importancia', y='Variable', data=df_importancias.head(15))
        plt.title('Importancia de las Variables')
        plt.xlabel('Importancia')
        plt.ylabel('Variable')
        plt.tight_layout()
        plt.savefig('importancia_variables.png')
        plt.close()
        
        return df_importancias
    
    def visualizar_importancia_avanzada(self, metodo='permutacion'):
        """
        Visualiza la importancia de las variables utilizando diferentes técnicas
        sin eliminar ninguna variable.
        
        Args:
            metodo (str): Método para calcular importancia ('permutacion', 'shap', 'ambos')
                
        Returns:
            pandas.DataFrame: Importancia de cada variable según el método elegido
        """
        if self.modelo is None:
            print("No hay modelo entrenado. Entrene primero un modelo.")
            return None
        
        # Crear figura para múltiples gráficos
        fig, axes = plt.subplots(1, 2 if metodo == 'ambos' else 1, figsize=(20, 10))
        
        # Dataframe para almacenar resultados
        importancias_df = pd.DataFrame({'Variable': self.X.columns})
        
        # 1. Importancia por permutación (funciona con cualquier modelo)
        if metodo in ['permutacion', 'ambos']:
            from sklearn.inspection import permutation_importance
            
            print("Calculando importancia por permutación...")
            # Realizar 10 permutaciones aleatorias para cada característica
            result = permutation_importance(
                self.modelo, self.X_test, self.y_test, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            # Almacenar resultados
            perm_importancia = result.importances_mean
            perm_std = result.importances_std
            
            # Añadir al dataframe
            importancias_df['Importancia_Permutacion'] = perm_importancia
            importancias_df['Std_Permutacion'] = perm_std
            
            # Ordenar por importancia
            df_perm = importancias_df.sort_values('Importancia_Permutacion', ascending=False)
            
            # Visualizar
            ax = axes if metodo != 'ambos' else axes[0]
            sns.barplot(x='Importancia_Permutacion', y='Variable', data=df_perm.head(15), ax=ax)
            ax.set_title('Importancia por Permutación')
            ax.set_xlabel('Reducción en rendimiento al permutar')
            ax.set_ylabel('Variable')
            
            print("Top 15 variables por importancia de permutación:")
            print(df_perm.head(15)[['Variable', 'Importancia_Permutacion']])
        
        # 2. SHAP Values (para modelos basados en árboles)
        if metodo in ['shap', 'ambos'] and hasattr(self.modelo, 'predict_proba'):
            try:
                import shap
                
                print("Calculando valores SHAP...")
                # Crear explicador SHAP
                if hasattr(self.modelo, 'estimators_'):  # Para modelos de ensemble
                    explainer = shap.TreeExplainer(self.modelo)
                else:
                    explainer = shap.Explainer(self.modelo)
                    
                # Calcular valores SHAP para una muestra del conjunto de prueba
                n_samples = min(500, self.X_test.shape[0])  # Limitar a 500 muestras para velocidad
                shap_values = explainer.shap_values(self.X_test.iloc[:n_samples])
                
                # Para modelos de clasificación multiclase
                if isinstance(shap_values, list):
                    # Promedio a través de todas las clases
                    mean_abs_shap = np.mean([np.abs(shap_values[i]).mean(0) for i in range(len(shap_values))], axis=0)
                else:
                    mean_abs_shap = np.abs(shap_values).mean(0)
                
                # Añadir al dataframe
                importancias_df['Importancia_SHAP'] = mean_abs_shap
                
                # Ordenar por importancia SHAP
                df_shap = importancias_df.sort_values('Importancia_SHAP', ascending=False)
                
                # Visualizar
                ax = axes if metodo != 'ambos' else axes[1]
                sns.barplot(x='Importancia_SHAP', y='Variable', data=df_shap.head(15), ax=ax)
                ax.set_title('Importancia por valores SHAP')
                ax.set_xlabel('Impacto promedio en la predicción (|SHAP|)')
                ax.set_ylabel('Variable')
                
                print("\nTop 15 variables por importancia SHAP:")
                print(df_shap.head(15)[['Variable', 'Importancia_SHAP']])
                
                # Guardar gráfico SHAP adicional
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, self.X_test.iloc[:n_samples], plot_type='bar', show=False)
                plt.tight_layout()
                plt.savefig('shap_importancia.png')
                plt.close()
                
            except ImportError:
                print("La librería SHAP no está instalada. Instálela con 'pip install shap'")
                if metodo == 'shap':
                    return df_perm if 'df_perm' in locals() else None
        
        # Ajustar diseño y guardar gráfico
        plt.tight_layout()
        plt.savefig('importancia_avanzada.png')
        plt.close()
        
        # Retornar DataFrame con ambas importancias si están disponibles
        if metodo == 'ambos' and 'Importancia_SHAP' in importancias_df.columns:
            return pd.merge(
                df_perm[['Variable', 'Importancia_Permutacion']], 
                df_shap[['Variable', 'Importancia_SHAP']],
                on='Variable'
            ).sort_values('Importancia_Permutacion', ascending=False)
        elif metodo == 'shap' and 'Importancia_SHAP' in importancias_df.columns:
            return df_shap
        else:
            return df_perm

    def validacion_cruzada(self, cv=5):
        """
        Realiza validación cruzada para evaluar el modelo.
        
        Args:
            cv (int): Número de folds para la validación cruzada
            
        Returns:
            dict: Resultados de la validación cruzada
        """
        if self.modelo is None:
            print("No hay modelo entrenado")
            return None
        
        if self.X is None or self.y is None:
            print("Datos no preparados")
            return None
        
        # Configurar estratificación para mantener proporciones de clases
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
        
        # Realizar validación cruzada
        cv_scores = cross_val_score(self.modelo, self.X, self.y, cv=skf, scoring='accuracy')
        
        print(f"Resultados de validación cruzada ({cv} folds):")
        print(f"Puntuaciones individuales: {cv_scores}")
        print(f"Exactitud media: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Guardar resultados
        cv_resultados = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        return cv_resultados
    
    def guardar_modelo(self, ruta='modelo_rendimiento_academico.pkl'):
        """
        Guarda el modelo entrenado en disco.
        
        Args:
            ruta (str): Ruta donde guardar el modelo
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        if self.modelo is None:
            print("No hay modelo para guardar")
            return False
        
        try:
            # Guardar modelo
            joblib.dump(self.modelo, ruta)
            print(f"Modelo guardado exitosamente en {ruta}")
            return True
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            return False
    
    def cargar_modelo(self, ruta):
        """
        Carga un modelo previamente guardado.
        
        Args:
            ruta (str): Ruta al archivo del modelo
            
        Returns:
            object: Modelo cargado
        """
        try:
            self.modelo = joblib.load(ruta)
            print(f"Modelo cargado exitosamente desde {ruta}")
            return self.modelo
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return None
    
    def predecir(self, nuevos_datos):
        """
        Realiza predicciones con nuevos datos.
        
        Args:
            nuevos_datos (pandas.DataFrame): Datos de entrada para la predicción
            
        Returns:
            array: Categorías de rendimiento académico predichas
        """
        if self.modelo is None:
            print("No hay modelo para realizar predicciones")
            return None
        
        try:
            # Verificar que las columnas coinciden con las del modelo
            columnas_requeridas = self.X.columns
            if not all(col in nuevos_datos.columns for col in columnas_requeridas):
                faltantes = [col for col in columnas_requeridas if col not in nuevos_datos.columns]
                print(f"Faltan columnas requeridas: {faltantes}")
                return None
            
            # Seleccionar solo las columnas utilizadas en el entrenamiento
            X_pred = nuevos_datos[columnas_requeridas]
            
            # Realizar predicción
            predicciones = self.modelo.predict(X_pred)
            return predicciones
        
        except Exception as e:
            print(f"Error al realizar predicciones: {e}")
            return None
        
    


# Ejemplo de uso
if __name__ == "__main__":
    

    modelo = ModeloPredictivo("../ETL/Transformar/alumnos_balanceados_smotetomek.csv")
    
    # Cargar y explorar datos
    modelo.cargar_datos()
    modelo.explorar_datos()
    
    # Analizar correlaciones
    modelo.analizar_correlaciones()
    
    # Preparar datos con transformaciones avanzadas
    modelo.preparar_datos_con_transformacion(aplicar_polinomicas=True, aplicar_interacciones=True)
    
    # Entrenar modelo avanzado
    modelo.entrenar_modelo_avanzado(modelo='xgboost')
    
    # Evaluar modelo
    resultado = modelo.evaluar_modelo()
    
    # Generar informe completo para dashboard
    
    #modelo.validacion_cruzada(cv=5)