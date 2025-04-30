import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

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
                                  test_size=0.25, random_state=42):
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
        excluir = ['id_alumno', variable_objetivo, 'rendimiento_general', 'matematicas', 'comprension_lectora', 'ciencias', 'matematicasCategoria' , 'comprension_lectoraCategoria', 'cienciasCategoria', 'indice_socioeconomicoCategoria']
        predictores = [col for col in self.datos.columns if col not in excluir]
        
        # Convertir variables categóricas a dummy si existen
        categoricas = [col for col in predictores if self.datos[col].dtype == 'object']
        if categoricas:
            self.datos = pd.get_dummies(self.datos, columns=categoricas, drop_first=True)
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
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
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
            parametros = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 0,
                'n_jobs': -1
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
            
        else:
            print(f"Modelo '{modelo}' no reconocido")
            return None
        
        # Entrenar modelo
        print(f"Entrenando modelo {modelo}...")
        self.modelo.fit(self.X_train, self.y_train)
        print("Modelo entrenado exitosamente")
        
        return self.modelo
    
    def evaluar_modelo(self):
        """
        Evalúa el rendimiento del modelo en el conjunto de prueba.
        
        Returns:
            dict: Métricas de evaluación del modelo
        """
        if self.modelo is None:
            print("No hay modelo entrenado. Llame primero a entrenar_modelo_clasificacion()")
            return None
        
        # Predecir en conjunto de prueba
        y_pred = self.modelo.predict(self.X_test)
        
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
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
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
    # Crear instancia del modelo
    modelo = ModeloPredictivo("../ETL/Transformar/alumnosMexico2022ProcesadosCategoricos.csv")
    
    # Cargar y explorar datos
    modelo.cargar_datos()
    modelo.explorar_datos()
    
    # Analizar correlaciones
    modelo.analizar_correlaciones()
    
    # Preparar datos para clasificación
    modelo.preparar_datos_clasificacion()
    
    # Entrenar modelo
    modelo.entrenar_modelo_clasificacion(modelo='gradient_boosting')
    
    # Evaluar modelo
    resultado = modelo.evaluar_modelo()
    
    # Mostrar importancia de variables
    importancia = modelo.importancia_variables()
    print("Top 10 variables más importantes:")
    print(importancia.head(10))
    
    # Realizar validación cruzada
    cv_resultados = modelo.validacion_cruzada(cv=5)
    
    # Guardar modelo
    modelo.guardar_modelo("modelo_rendimiento_academico.pkl")