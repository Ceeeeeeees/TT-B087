import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

pd.set_option('future.no_silent_downcasting', True)  # Desactivar advertencia de downcasting

class ProcesamientoDatos:

    def __init__(self, archivoAlumnos):
        self.archivoAlumnos = archivoAlumnos
        self.df_original = None  # DataFrame original procesado
        
    def procesarArchivoDataFrame(self):
        """
        Permite cargar el archivo csv en un DataFrame de pandas.
        """
        try:
            self.df_original = pd.read_csv(self.archivoAlumnos)
            print(f"Datos cargados correctamente. Dimensiones: {self.df_original.shape}")
            return self.df_original
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None
    
    def eliminarColumnas(self):
        """
        Elimina columnas que no son necesarias.
        """
        if self.df_original is None:
            return None
            
        columnasEliminar = ['desk', 'dishwasher', 'wealth', 'year', 'country', 'school_id', 'stu_wgt', 'computer_n']
        
        columnasDisponibles = [col for col in columnasEliminar if col in self.df_original.columns]
        
        if columnasDisponibles:
            self.df_original.drop(columns=columnasDisponibles, inplace=True)
            print(f"Columnas eliminadas: {columnasDisponibles}")
        else:
            print(f"No se encontraron las columnas {columnasEliminar} para eliminar")
            
        return self.df_original
    
    def renombrarIndiceAlumno(self):
        """
        Renombra el índice de los alumnos del DataFrame.
        """
        if self.df_original is None:
            return None
            
        self.df_original['student_id'] = range(1, len(self.df_original) + 1)
        return self.df_original
    
    def tratarValoresFaltantes(self):
        """
        Maneja todos los valores faltantes en las columnas importantes.
        """
        if self.df_original is None:
            return None
            
        # Tratar valores faltantes en columnas específicas
        # Computer
        if 'computer_n' in self.df_original.columns and 'computer' in self.df_original.columns:
            self.df_original['computer_n'] = pd.to_numeric(self.df_original['computer_n'], errors='coerce') 
            self.df_original.loc[(self.df_original['computer_n'].isna()) | (self.df_original['computer_n'] == 0), 'computer'] = 'no'
            self.df_original.loc[(self.df_original['computer_n'] >= 1) & (self.df_original['computer'].isna()), 'computer'] = 'yes'
        
        # Internet
        if 'computer' in self.df_original.columns and 'internet' in self.df_original.columns:
            self.df_original['computer'] = self.df_original['computer'].astype(str).str.strip()
            self.df_original.loc[(self.df_original['internet'].isna()) & (self.df_original['computer'] == '1'), 'internet'] = 'yes'
            self.df_original.loc[(self.df_original['internet'].isna()) & (self.df_original['computer'] == '0'), 'internet'] = 'no'
        
        # Room
        if 'room' in self.df_original.columns:
            self.df_original['room'] = self.df_original['room'].fillna(self.df_original['room'].mode()[0])
        
        # Television
        if 'television' in self.df_original.columns:
            self.df_original['television'] = self.df_original['television'].fillna(self.df_original['television'].mode()[0])
        
        # Car
        if 'car' in self.df_original.columns:
            self.df_original['car'] = self.df_original['car'].fillna(self.df_original['car'].mode()[0])
        
        # Book
        if 'book' in self.df_original.columns:
            self.df_original['book'] = self.df_original['book'].fillna(self.df_original['book'].mode()[0])

        # Internet
        if 'internet' in self.df_original.columns:
            self.df_original['internet'] = self.df_original['internet'].fillna(self.df_original['internet'].mode()[0])
        
        # ESCS (índice socioeconómico)
        if 'escs' in self.df_original.columns:
            self.df_original['escs'] = self.df_original['escs'].fillna(self.df_original['escs'].median())
        
        # Materias (math, read, science)
        for columna in ['math', 'read', 'science']:
            if columna in self.df_original.columns:
                self.df_original[columna] = self.df_original[columna].fillna(self.df_original[columna].median())
        
        return self.df_original
    
    def convertirVariablesBinarias(self):
        """
        Convierte las variables categóricas a binarias o numéricas.
        """
        if self.df_original is None:
            return None
        
        # Conversión de computer
        if 'computer' in self.df_original.columns:
            equivalenciaComputadora = {'no': 0, 'yes': 1}
            self.df_original['computer'] = self.df_original['computer'].replace(equivalenciaComputadora)
        
        # Conversión de género
        if 'gender' in self.df_original.columns:
            equivalenciaGenero = {'male': 0, 'female': 1}
            self.df_original['gender'] = self.df_original['gender'].replace(equivalenciaGenero)
        
        # Conversión de internet
        if 'internet' in self.df_original.columns:
            equivalenciaInternet = {'no': 0, 'yes': 1}
            self.df_original['internet'] = self.df_original['internet'].replace(equivalenciaInternet)
        
        # Conversión de room
        if 'room' in self.df_original.columns:
            equivalenciaCuarto = {'no': 0, 'yes': 1}
            self.df_original['room'] = self.df_original['room'].replace(equivalenciaCuarto)
        
        # Conversión de television
        if 'television' in self.df_original.columns:
            self.df_original['television'] = self.df_original['television'].replace({'3+': 1}).astype('int')
            self.df_original['television'] = self.df_original['television'].apply(lambda x: 1 if x > 0 else 0)
        
        # Conversión de car
        if 'car' in self.df_original.columns:
            self.df_original['car'] = self.df_original['car'].replace({'3+': 1}).astype('int')
            self.df_original['car'] = self.df_original['car'].apply(lambda x: 1 if x > 0 else 0)
        
        # Conversión de book
        if 'book' in self.df_original.columns:
            equivalenciasLibros = {
                "0-10": 0, "11-25": 1, "26-100": 2,
                "101-200": 3, "201-500": 4, "more than 500": 6
            }
            self.df_original['book'] = self.df_original['book'].replace(equivalenciasLibros).astype('int')
        
        # Conversión de educación de padres
        for columna in ['mother_educ', 'father_educ']:
            if columna in self.df_original.columns:
                equivalenciasISCED = {
                    "less than ISCED1": 1,  # Menor que educación Primaria
                    "ISCED 1": 2,           # Educación Primaria
                    "ISCED 2": 3,           # Educación Secundaria
                    "ISCED 3A": 4,          # Bachillerato
                    "ISCED 3B, C": 5        # Bachillerato Técnico
                }
                self.df_original.loc[(self.df_original[columna].isna()) | 
                                    (self.df_original[columna] == 'NaN') | 
                                    (self.df_original[columna] == 'NA'), columna] = "0"
                self.df_original[columna] = self.df_original[columna].replace(equivalenciasISCED)
        
        return self.df_original
    
    def renombrarEncabezado(self):
        """
        Traduce y renombra las columnas del DataFrame a español.
        """
        if self.df_original is None:
            return None
            
        encabezado = {
            "student_id": "id_alumno",
            "mother_educ": "educacion_madre",
            "father_educ": "educacion_padre",
            "gender": "genero",
            "computer": "computadora",
            "internet": "internet",
            "math": "matematicas",
            "read": "comprension_lectora",
            "science": "ciencias",
            "room": "cuarto_propio",
            "television": "television",
            "car": "auto",   
            "book": "libros",
            "escs": "indice_socioeconomico"  
        }
        
        self.df_original.columns = self.df_original.columns.str.strip()
        
        # Solo renombrar columnas que existen en el DataFrame
        renombrar = {col: encabezado[col] for col in encabezado if col in self.df_original.columns}
        if renombrar:
            self.df_original.rename(columns=renombrar, inplace=True)
            print(f"Columnas renombradas: {list(renombrar.keys())}")
        else:
            print("No se encontraron columnas para renombrar")
            
        return self.df_original
    
    def agregarRendimientoAcademico(self):
        """
        Agrega una columna de rendimiento académico como promedio de las materias.
        """
        if self.df_original is None:
            return None
            
        # Identificar columnas de calificaciones
        columnas_calificaciones = ['matematicas', 'comprension_lectora', 'ciencias']
        
        # Verificar cuáles de estas columnas están en el DataFrame
        columnas_disponibles = [col for col in columnas_calificaciones if col in self.df_original.columns]
        
        if len(columnas_disponibles) == 0:
            print("No se encontraron variables de materias para calcular el rendimiento académico.")
            return self.df_original
            
        # Calcular promedio
        self.df_original['rendimiento_academico'] = self.df_original[columnas_disponibles].mean(axis=1)
        print(f"Rendimiento académico calculado usando: {columnas_disponibles}")
        
        return self.df_original
    
    def normalizarVariables(self):
        """
        Normaliza las variables continuas utilizando StandardScaler.
        """
        if self.df_original is None:
            return None
            
        # Verificar si las columnas existen
        columnas_a_normalizar = []
        
        # Verificar calificaciones
        for col in ['matematicas', 'comprension_lectora', 'ciencias', 'rendimiento_academico']:
            if col in self.df_original.columns:
                columnas_a_normalizar.append(col)
        
        # Verificar índice socioeconómico
        if 'indice_socioeconomico' in self.df_original.columns:
            columnas_socioeconomicas = ['indice_socioeconomico']
        else:
            columnas_socioeconomicas = []
        
        if len(columnas_a_normalizar) == 0 and len(columnas_socioeconomicas) == 0:
            print("No se encontraron columnas para normalizar.")
            return self.df_original
        
        # Normalizar calificaciones
        if columnas_a_normalizar:
            escalador_calificaciones = StandardScaler()
            self.df_original[columnas_a_normalizar] = escalador_calificaciones.fit_transform(
                self.df_original[columnas_a_normalizar]
            )
            print(f"Columnas de calificaciones normalizadas: {columnas_a_normalizar}")
        
        # Normalizar variables socioeconómicas
        if columnas_socioeconomicas:
            escalador_socioeconomico = StandardScaler()
            self.df_original[columnas_socioeconomicas] = escalador_socioeconomico.fit_transform(
                self.df_original[columnas_socioeconomicas]
            )
            print(f"Columnas socioeconómicas normalizadas: {columnas_socioeconomicas}")
        
        return self.df_original
    
    def muestraValoresFaltantes(self):
        """
        Muestra la cantidad de valores faltantes en cada columna.
        """
        if self.df_original is None:
            return None
            
        valores_faltantes = self.df_original.isnull().sum()
        print(f'Valores faltantes en cada columna:\n{valores_faltantes}')
        return valores_faltantes
    
    def guardarDatosProcesados(self, nombre_archivo='alumnosMexico2022_procesados.csv'):
        """
        Guarda el DataFrame procesado en un archivo CSV.
        """
        if self.df_original is None:
            print("No hay datos para guardar.")
            return None
            
        self.df_original.to_csv(nombre_archivo, index=False)
        print(f"Datos procesados guardados como: {nombre_archivo}")
        return nombre_archivo
    
    def pipelineProcesamiento(self, guardar=True, nombre_archivo='alumnosMexico2022_procesados.csv'):
        """
        Ejecuta el pipeline completo de procesamiento de datos.
        """
        
        # Paso 1: Cargar datos
        self.procesarArchivoDataFrame()
        
        # Paso 2: Limpieza básica
        self.renombrarIndiceAlumno()
        self.tratarValoresFaltantes()
        self.convertirVariablesBinarias()
        self.eliminarColumnas()
        
        # Paso 3: Renombrar columnas (IMPORTANTE: hacer esto ANTES de calcular rendimiento)
        self.renombrarEncabezado()
        
        # Paso 4: Calcular rendimiento académico
        self.agregarRendimientoAcademico()
        
        # Paso 5: Normalizar variables
        self.normalizarVariables()
        
        self.muestraValoresFaltantes()
        # Paso 6: Guardar datos procesados
        if guardar and self.df_original is not None:
            self.guardarDatosProcesados(nombre_archivo)
        
        return self.df_original


if __name__ == "__main__":
    rutaArchivoOriginal = "./alumnosMexico2022.csv"
    
    procesador = ProcesamientoDatos(rutaArchivoOriginal)
    df_procesado = procesador.pipelineProcesamiento(guardar=True)
    
    if df_procesado is not None:
        print("\nResumen del DataFrame procesado:")
        print(f"Dimensiones: {df_procesado.shape}")
        print("\nPrimeras filas del DataFrame procesado:")
        print(df_procesado.head())
    else:
        print("El procesamiento no se completó correctamente.")