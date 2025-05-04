import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

pd.set_option('future.no_silent_downcasting', True) # Desactivar advertencia de downcasting

class ProcesamientoDatos:

    def __init__(self, archivoAlumnos):
        self.archivoAlumnos = archivoAlumnos
        self.df_original = None  # DataFrame original procesado
        self.df_continuo = None  # DataFrame con variables continuas normalizadas
        self.df_categorico = None  # DataFrame con variables categorizadas
        
    def procesarArchivoDataFrame(self):
        """
        Permite cargar el archivo csv en un DataFrame de pandas.
        Retorna:
            pandas.DataFrame: El DataFrame cargado con los datos del archivo csv.
        """
        archivoAlumnosSinProcesar = self.archivoAlumnos
        self.df_original = pd.read_csv(archivoAlumnosSinProcesar)
        
        return self.df_original
    
    def eliminarColumnas(self, df=None):
        """
        Manejo de valores faltantes en el dataset.
            Eliminamos columnas que no son necesarias debido a que no contiene información durante la extracción de datos.
            Las columnas eliminadas son: 'desk', 'dishwasher', 'wealth', 'year', 'country', 'school_id', 'stu_wgt', 'computer_n'.
        Retorna:
            pandas.DataFrame: El DataFrame con las columnas eliminadas.
        """
        if df is None:
            df = self.df_original
            
        columnasEliminar = ['desk', 'dishwasher', 'wealth', 'year', 'country', 'school_id', 'stu_wgt', 'computer_n']
        
        columnasDisponibles = [col for col in columnasEliminar if col in df.columns]
        
        if columnasDisponibles:
            df.drop(columns=columnasDisponibles, inplace=True)
        else:
            print(f"No se encontraron las columnas {columnasEliminar} para eliminar")
            
        return df
    
    def eliminarColumnasDuplicadas(self, columnasDuplicadas, df=None):
        """
        Permite eliminar columnas dentro del DataFrame.
        Argumentos:
            columnasDuplicadas (list): Lista de nombres de columnas a eliminar.
            df (pandas.DataFrame): DataFrame a modificar (opcional)
        Retorna:
            pandas.DataFrame: El DataFrame sin columnas duplicadas.
        """
        if df is None:
            df = self.df_original.copy()
        
        columnasEliminar = [columna for columna in columnasDuplicadas if columna in df.columns]
        
        if columnasEliminar:
            df.drop(columns=columnasEliminar, inplace=True)
        else:
            print(f"No se encontraron las columnas {columnasDuplicadas} para eliminar")

        return df
    
    def renombrarIndiceAlumno(self, df=None):
        """
        Permite renombrar el índice de los alumnos del DataFrame.
        Retorna:
            pandas.DataFrame: El DataFrame con el índice renombrado.
        """
        if df is None:
            df = self.df_original
            
        df['student_id'] = range(1, len(df) + 1)
        return df
    
    def valorFaltanteComputadora(self, df=None):
        """
        Manejo de valores faltantes en la columna 'computer'.
            Para ello, revisamos si la columna 'computer_n' tiene valores nulos o cero, y si es así, asignamos 'no' a la columna 'computer'.
            Si la columna 'computer_n' tiene valores mayores a 1, asignamos 'yes' a la columna 'computer'.
        Retorna:    
            pandas.DataFrame: El DataFrame con los valores faltantes en 'computer' rellenados.
        """
        if df is None:
            df = self.df_original
            
        if 'computer_n' in df.columns:
            df['computer_n'] = pd.to_numeric(df['computer_n'], errors='coerce') 
            df.loc[(df['computer_n'].isna()) | (df['computer_n'] == 0), 'computer'] = 'no'
            df.loc[(df['computer_n'] >= 1) & (df['computer'].isna()), 'computer'] = 'yes'

        return df
    
    def valorFaltanteInternet(self, df=None):
        """
        Manejo de valores faltantes en la columna 'internet'.
            Para ello, revisamos si la columna 'internet' tiene valores nulos y si la columna 'computer' tiene valor 1, asignamos 'yes' a la columna 'internet'.
            Si la columna 'computer' tiene valor 0, asignamos 'no' a la columna 'internet'.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'internet' rellenados.
        """
        if df is None:
            df = self.df_original
            
        if 'computer' in df.columns and 'internet' in df.columns:
            df['computer'] = df['computer'].astype(str).str.strip() # Eliminar espacios en blanco
            
            df.loc[(df['internet'].isna()) & (df['computer'] == '1'), 'internet'] = 'yes'
            df.loc[(df['internet'].isna()) & (df['computer'] == '0'), 'internet'] = 'no'
        
        return df
    
    def valorFaltanteCuarto(self, df=None):
        """
        Maneja los valores faltantes en la columna 'room'.
            Para ello, rellanamos los valores NaN en la columna 'room' con el valor más frecuente (moda) de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'room' rellenados.
        """
        if df is None:
            df = self.df_original
            
        if 'room' in df.columns:
            df['room'] = df['room'].fillna(df['room'].mode()[0])     # Rellenar con la moda 
        
        return df
    
    def valorFaltanteTelevision(self, df=None):
        """
        Maneja los valores faltantes en la columna 'television'.
            Para ello, rellanamos los valores NaN en la columna 'television' con el valor más frecuente (moda) de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'television' rellenados.
        """
        if df is None:
            df = self.df_original
            
        if 'television' in df.columns:
            df['television'] = df['television'].fillna(df['television'].mode()[0])     # Rellenar con la moda
        
        return df
    
    def valorFaltanteAutos(self, df=None):
        """"
        Maneja los valores faltantes en la columna 'car'.
            Para ello, rellanamos los valores NaN en la columna 'car' con el valor más frecuente (moda) de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'car' rellenados.
        """
        if df is None:
            df = self.df_original
            
        if 'car' in df.columns:
            df['car'] = df['car'].fillna(df['car'].mode()[0])     # Rellenar con la moda
        
        return df

    def valorFaltanteLibros(self, df=None):
        """"
        Maneja los valores faltantes en la columna 'book'.
            Para ello, rellanamos los valores NaN en la columna 'car' con el valor más frecuente (moda) de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'book' rellenados.
        """
        if df is None:
            df = self.df_original
            
        if 'book' in df.columns:
            df['book'] = df['book'].fillna(df['book'].mode()[0])     # Rellenar con la moda
        
        return df
    
    def valorFaltanteIndiceSocioEconomico(self, df=None):
        """
        Maneja los valores faltantes en la columna 'escs'.
            Para ello, rellanamos los valores NaN en la columna 'escs' con la mediana de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'escs' rellenados.
        """
        if df is None:
            df = self.df_original
            
        if 'escs' in df.columns:
            df['escs'] = df['escs'].fillna(df['escs'].median())   # Rellenar con la mediana
        
        return df
    
    def conversionBinariaComputadora(self, df=None):
        """
        Convierte los valores de la columna 'computer' a valores binarios.
            Valores con 'no' se convierten a 0 y valores con 'yes' se convierten a 1.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'computer' convertida a valores binarios.
        """
        if df is None:
            df = self.df_original
            
        if 'computer' in df.columns:
            equivalenciaComputadora = {
                'no'                : 0,
                'yes'               : 1
            }

            df['computer'] = df['computer'].replace(equivalenciaComputadora).infer_objects(copy=False)
        
        return df

    def conversionBinariaGenero(self, df=None):
        """
        Convierte los valores de la columna 'gender' a valores binarios.
            Valores con 'male' se convierten a 0 y valores con 'female' se convierten a 1.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'gender' convertida a valores binarios.
        """
        if df is None:
            df = self.df_original
            
        if 'gender' in df.columns:
            equivalenciaGenero = {
                'male'              : 0,
                'female'            : 1
            }

            df['gender'] = df['gender'].replace(equivalenciaGenero).infer_objects(copy=False)
        
        return df
    
    def conversionBinariaInternet(self, df=None):
        """
        Convierte los valores de la columna 'internet' a valores binarios.
            Permite definir si el alumno tiene acceso a internet o no.
            Valores con 'no' se convierten a 0 y valores con 'yes' se convierten a 1.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'internet' convertida a valores binarios."""
        if df is None:
            df = self.df_original
            
        equivalenciaInternet = {
            'no'                : 0,
            'yes'               : 1
        }

        df['internet'] = df['internet'].replace(equivalenciaInternet).infer_objects(copy=False)
        
        return df
    
    def conversionBinariaCuarto(self, df=None):
        """
        Convierte los valores de la columna 'room' a valores binarios.
            Permite definir si el alumno tiene cuarto propio o no.
            Valores con 'no' se convierten a 0 y valores con 'yes' se convierten a 1.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'room' convertida a valores binarios.
        """
        if df is None:
            df = self.df_original
            
        equivalenciaCuarto = {
            'no'                : 0,
            'yes'               : 1
        }

        df['room'] = df['room'].replace(equivalenciaCuarto).infer_objects(copy=False)
        
        return df
    
    def conversionBinariaTelevision(self, df=None):
        """
        Convierte los valores de la columna 'television' a valores binarios.
            Permite definir si el alumno tiene televisión o no.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'television' convertida a valores binarios.
        """
        if df is None:
            df = self.df_original
            
        if 'television' in df.columns:
            df['television'] = df['television'].replace({'3+' : 1}).astype('int')         # Reemplaza '3+' por 1 y convierte el tipo de dato a entero.
            df['television'] = df['television'].apply(lambda x: 1 if x > 0 else 0)        # Devuelve 1 si x es mayor que 0, 0 de otra forma.
        
        return df
    
    def conversionBinariaAutos(self, df=None):
        """
        Convierte los valores de la columna 'car' a valores binarios.
            Permite definir si el alumno tiene automóvil o no.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'car' convertida a valores binarios.
        """
        if df is None:
            df = self.df_original
            
        if 'car' in df.columns:
            df['car'] = df['car'].replace({'3+' : 1}).astype('int')        # Reemplaza '3+' por 1 y convierte el tipo de dato a entero.
            df['car'] = df['car'].apply(lambda x: 1 if x > 0 else 0)       # Devuelve 1 si x es mayor que 0, 0 de otra forma.
        
        return df
    
    def conversionLabelEncodingLibros(self, df=None):
        """
        Convierte los valores de la columna 'book' a valores numéricos.
            Permite definir una categoria para la cantidad de libros que tiene el alumno.
        Retornna:
            pandas.DataFrame: El DataFrame con la columna 'book' convertida a valores numéricos.
        """
        if df is None:
            df = self.df_original
            
        equivalenciasLibros = {
            "0-10"          : 0,
            "11-25"         : 1,
            "26-100"        : 2,
            "101-200"       : 3, 
            "201-500"       : 4,     
            "more than 500" : 6
        }
        df['book'] = df['book'].replace(equivalenciasLibros).astype('int')
        
        return df

    def mostrarValoresUnicos(self, df=None):
        """
        Permite visualizar los valores únicos de cada columna del DataFrame.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores únicos de cada columna.
        """
        if df is None:
            df = self.df_original
            
        for columna in df.columns:
            print(f'Valores únicos en {columna}: {df[columna].unique()}')
        
        return df

    def equivalenciaEducacionPadres(self, df=None):
        """
        Convierte la equivalencia de la educación de los padres a su equivalente en español.
            Permite definir la educación de los padres en el DataFrame.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'mother_educ' y 'father_educ' convertida a su equivalente en español.
        """
        if df is None:
            df = self.df_original
            
        equivalenciasISCED = {
            "less than ISCED1"  : "menos que primaria",         # Menor que la educacion Primaria
            "ISCED 1"           : "primaria",                   # Educacion Primaria   
            "ISCED 2"           : "secundaria",                 # Educacion Secundaria
            "ISCED 3A"          : "bachillerato",               # Bachillerato
            "ISCED 3B, C"       : "bachillerato tecnico"        # Bachillerato Técnico
        }

        df.loc[(df['mother_educ'].isna()) | (df['mother_educ'] == 'NaN') | (df['mother_educ'] == 'NA'), 'mother_educ'] = "sin informacion"
        df['mother_educ'] = df['mother_educ'].replace(equivalenciasISCED).infer_objects(copy=True)
        
        df.loc[(df['father_educ'].isna()) | (df['father_educ'] == 'NaN') | (df['father_educ'] == 'NA'), 'father_educ'] = "sin informacion"
        df['father_educ'] = df['father_educ'].replace(equivalenciasISCED).infer_objects(copy=True)

        return df
    
    def equivalenciaEducacionPadresCategorica(self, df=None):
        """
        Convierte la equivalencia de la educación de los padres a valores numéricos categóricos.
            Permite definir la educación de los padres según una categoria en el DataFrame.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'educacion_madre' y 'educacion_padre' convertida a valores numéricos.
        """
        if df is None:
            df = self.df_categorico

        equivalenciasISCED = {
            "sin informacion"      : 0,         # Sin información
            "menos que primaria"   : 1,         # Menor que la educacion Primaria
            "primaria"             : 2,         # Educacion Primaria   
            "secundaria"           : 3,         # Educacion Secundaria
            "bachillerato"         : 4,         # Bachillerato
            "bachillerato tecnico" : 5          # Bachillerato Técnico
        }

        df['educacion_madre'] = df['educacion_madre'].replace(equivalenciasISCED).infer_objects(copy=True)
    
        df['educacion_padre'] = df['educacion_padre'].replace(equivalenciasISCED).infer_objects(copy=True)

        return df

    def renombrarEncabezado(self, df=None):
        """
        Permite traducir y renombrar el nombre de las columnas del DataFrame.
            Se traduce el nombre de las columnas a español y se eliminan los espacios en blanco.
        Retorna:
            pandas.DataFrame: El DataFrame con los nombres de las columnas traducidos y renombrados.
        """
        if df is None:
            df = self.df_original
            
        encabezado = {
            "student_id"    : "id_alumno",
            "mother_educ"   : "educacion_madre",
            "father_educ"   : "educacion_padre",
            "gender"        : "genero",
            "computer"      : "computadora",
            "internet"      : "internet",
            "math"          : "matematicas",
            "read"          : "comprension_lectora",
            "science"       : "ciencias",
            "room"          : "cuarto_propio",
            "television"    : "television",
            "car"           : "auto",   
            "book"          : "libros",
            "escs"          : "indice_socioeconomico"  
        }
        df.columns = df.columns.str.strip()
        
        # Solo renombrar columnas que existen en el DataFrame
        renombrar = {col: encabezado[col] for col in encabezado if col in df.columns}
        if renombrar:
            df.rename(columns=renombrar, inplace=True)
        else:
            print("No se encontraron columnas para renombrar")
            
        return df

    def muestraValoresFaltantes(self, df=None):
        """
        Permite ver la cantidad de valores faltantes en cada columna.
        Retorna:
            pandas.Series: Serie con la cantidad de valores faltantes en cada columna del DataFrame.
        """
        if df is None:
            df = self.df_original
            
        valoresFaltantes = df.isnull().sum()
        print(f'Valores faltantes en cada columna:\n{valoresFaltantes}')
        return valoresFaltantes

    def guardarDatosProcesados(self, df=None, nombreArchivoProcesado='alumnosMexico2022Procesados.csv'):
        """
        Permite guardar el archivo procesado en un nuevo archivo csv.
        Argumentos:
            df (pandas.DataFrame): DataFrame a guardar (opcional)
            nombreArchivoProcesado (str): Nombre del archivo procesado guardado.
        Retorna:
            str: Nombre del archivo procesado guardado.
        """
        if df is None:
            df = self.df_original
            
        df.to_csv(nombreArchivoProcesado, index=False)
        print(f'Procesamiento Completado. Archivo guardado como {nombreArchivoProcesado}')
        return nombreArchivoProcesado
    
    def normalizarVariablesContinuas(self, df=None, escalador='standard'):
        """
        Normaliza las variables continuas del DataFrame.
            Permite normalizar las variables continuas de las columnas:
            'matematicas', 'comprension_lectora', 'ciencias', 'indice_socioeconomico'.
        Argumentos:
            df (pandas.DataFrame): DataFrame a normalizar (opcional)
            escalador (str): Permite elegir entre dos escaladores: 'standard' o 'minmax'.
        Retorna:
            pandas.DataFrame: El DataFrame con las variables continuas normalizadas.
        """
        if df is None:
            df = self.df_continuo
            
        variablesContinuas = [col for col in ['matematicas', 'comprension_lectora', 'ciencias', 'indice_socioeconomico'] 
                             if col in df.columns]
            
        if escalador == 'standard':
            escaladorVariables = StandardScaler()
        elif escalador == 'minmax':
            escaladorVariables = MinMaxScaler()
        else:
            raise ValueError("Escalador no válido. Use 'standard', 'minmax' o 'robust'.")
    
        df[variablesContinuas] = escaladorVariables.fit_transform(df[variablesContinuas])

        return df
    
    def agregarRendimientoAcademicoContinuo(self, df=None, metodo = 'promedioSimple'):
        """
        Agrega una columna de rendimiento académico continuo basada en las materias.
        
        Argumentos:
            df (pandas.DataFrame): DataFrame al que se agregará la columna (opcional)
            metodo (str): Método de cálculo ('promedio_simple', 'promedio_ponderado', 'estandarizado')
            
        Retorna:
            pandas.DataFrame: El DataFrame con la columna de rendimiento académico agregada.
        """
        if df is None:
            df = self.df_continuo
            
        # Comprobamos qué variables existen en el DataFrame
        variablesMaterias = [col for col in ['matematicas', 'comprension_lectora', 'ciencias'] 
                            if col in df.columns]
            
        if len(variablesMaterias) == 0:
            print("No se encontraron variables de materias para calcular el rendimiento académico.")
            return df
        
        if metodo == 'promediosimple':
            df['rendimiento_academico_continuo'] = df[variablesMaterias].mean(axis=1)        
        elif metodo == 'estandarizado':
            for materia in variablesMaterias:
                df[f'{materia}_estandarizado'] = (df[materia] - df[materia].mean()) / df[materia].std()

            columnasEstandarizadas = [ f'{materia}_estandarizado' for materia in variablesMaterias ]
            df['rendimiento_academico_continuo'] = df[columnasEstandarizadas].mean(axis=1)
            df.drop(columns=columnasEstandarizadas, inplace=True)
        else:
            raise ValueError("Método no válido. Puede utilizar 'promediosimple' o 'estandarizado'.")
    
    def categorizarRendimientoAcademicoMaterias(self, df=None, metodo='intervalosIguales'):
        """
        Categoriza las variables de rendimiento académico según diferentes métodos.
        Argumentos:
            df (pandas.DataFrame): DataFrame a categorizar (opcional)
            metodo (str): Método de categorización ('pisa', 'percentiles', 'intervalosIguales')
            
        Retorna:
            pandas.DataFrame: El DataFrame con las variables de rendimiento académico categorizadas.
        """
        if df is None:
            df = self.df_categorico
            
        # Comprobamos qué variables existen en el DataFrame
        variablesMaterias = [col for col in ['matematicas', 'comprension_lectora', 'ciencias'] 
                            if col in df.columns]
            
        #print(f"Categorizando las siguientes materias: {variablesMaterias}")
            
        if metodo == 'pisa':
            puntosCorte = {
                'matematicas' :         [0, 233, 295, 358, 420, 482, 545, 607, 669, float('inf')],
                'comprension_lectora' : [0, 185, 262, 335, 407, 480, 553, 626, 698, float('inf')],
                'ciencias' :            [0, 261, 335, 410, 484, 559, 633, 708, float('inf')]
            }

            etiquetasCorte = ['Nivel <1b', 'Nivel 1b', 'Nivel 1a', 'Nivel 2', 'Nivel 3', 'Nivel 4', 'Nivel 5', 'Nivel 6']

            for variable in variablesMaterias:
                if variable in puntosCorte:
                    df[f'{variable}Categoria'] = pd.cut(
                        df[variable],
                        bins=puntosCorte[variable],
                        labels=etiquetasCorte[:len(puntosCorte[variable])-1],
                        include_lowest=True
                    )
                else:
                    print(f"No hay puntos de corte definidos para '{variable}'")
        
        elif metodo == 'percentiles':
            quintiles = [0, 0.33, 0.66, 1]
            etiquetas = ['Bajo', 'Medio', 'Alto']

            for variable in variablesMaterias:
                puntosCorte = [df[variable].quantile(q) for q in quintiles]
                puntosCorte = sorted(set(puntosCorte))
                
                if len(puntosCorte) < 3:
                    print(f"No hay suficientes valores únicos en '{variable}' para crear quintiles.")
                    df[f'{variable}Categoria'] = 'No categorizable'
                else:
                    df[f'{variable}Categoria'] = pd.cut(
                        df[variable], 
                        bins=puntosCorte,
                        labels=etiquetas[:len(puntosCorte)-1],
                        include_lowest=True
                    )
                    
        elif metodo == 'intervalosIguales':
            etiquetas = ['Bajo', 'Medio', 'Alto']

            for variable in variablesMaterias:
                minValor = df[variable].min()
                maxValor = df[variable].max()

                puntosCorte = np.linspace(minValor, maxValor, 4) #Dividir en 3 intervalos iguales

                df[f'{variable}Categoria'] = pd.cut(
                    df[variable], 
                    bins=puntosCorte,
                    labels=etiquetas,
                    include_lowest=True
                )
                
        else:
            raise ValueError("Método no válido. Puede utilizar 'pisa', 'percentiles' o 'intervalosIguales'.")
            
        return df
    
    def categorizarIndiceSocioEconomico(self, df=None):
        """
        Categoriza el índice socioeconómico en intervalos iguales.
        
        Argumentos:
            df (pandas.DataFrame): DataFrame a categorizar (opcional)
            
        Retorna:
            pandas.DataFrame: El DataFrame con el índice socioeconómico categorizado.
        """
        if df is None:
            df = self.df_categorico
            
        if 'indice_socioeconomico' not in df.columns:
            print("La columna 'indice_socioeconomico' no se encuentra en el DataFrame.")
            return df
            
        columnaCategoria = "indice_socioeconomicoCategoria"
        etiquetasIndice = ['Bajo', 'Medio', 'Alto']

        minValor = df['indice_socioeconomico'].min()
        maxValor = df['indice_socioeconomico'].max()

        puntosCorte = np.linspace(minValor, maxValor, 4)

        df[columnaCategoria] = pd.cut(
            df['indice_socioeconomico'], 
            bins=puntosCorte,
            labels=etiquetasIndice,
            include_lowest=True
        )
        
        return df
    
    def categorizarRendimientoGeneral(self, df=None):
        """
        Categoriza el rendimiento académico general de los estudiantes
        basado en el promedio de matemáticas, comprensión lectora y ciencias.
        
        Argumentos:
            df (pandas.DataFrame): DataFrame a categorizar (opcional)
            
        Retorna:
            pandas.DataFrame: El DataFrame con el rendimiento general categorizado.
        """
        if df is None:
            df = self.df_categorico
            
        variablesMaterias = [col for col in ['matematicas', 'comprension_lectora', 'ciencias'] 
                            if col in df.columns]
                            
        if len(variablesMaterias) == 0:
            print("No se encontraron variables de materias para calcular el rendimiento general.")
            return df
        
        # Creamos la columna de rendimiento general como promedio de las materias
        df['rendimiento_general'] = df[variablesMaterias].mean(axis=1)
        
        etiquetas = ['Bajo', 'Medio', 'Alto']
        
        minValor = df['rendimiento_general'].min()
        maxValor = df['rendimiento_general'].max()
        puntosCorte = np.linspace(minValor, maxValor, 4)
        
        df['rendimiento_general_categoria'] = pd.cut(
            df['rendimiento_general'], 
            bins=puntosCorte,
            labels=etiquetas,
            include_lowest=True
        )
        
        return df
        
    def procesarDatosOriginales(self):
        """
        Realiza el procesamiento inicial de los datos, generando el DataFrame original procesado.
        
        Retorna:
            pandas.DataFrame: DataFrame original procesado.
        """
        self.procesarArchivoDataFrame()
        
        # Procesamos los datos
        self.renombrarIndiceAlumno()
        self.valorFaltanteComputadora()
        self.conversionBinariaComputadora()
        self.valorFaltanteInternet()
        self.conversionBinariaInternet()
        self.valorFaltanteCuarto()
        self.conversionBinariaCuarto()
        self.valorFaltanteTelevision()
        self.conversionBinariaTelevision()
        self.valorFaltanteAutos()
        self.conversionBinariaAutos()
        self.valorFaltanteLibros()
        self.conversionLabelEncodingLibros()
        self.valorFaltanteIndiceSocioEconomico()
        self.eliminarColumnas()
        self.equivalenciaEducacionPadres()
        self.conversionBinariaGenero()
        self.renombrarEncabezado()
        
        print("Procesamiento de datos originales completado.")
        return self.df_original
    
    def procesarDatosContinuos(self):
        """
        Genera el DataFrame con variables continuas normalizadas.
        
        Retorna:
            pandas.DataFrame: DataFrame con variables continuas normalizadas.
        """
        if self.df_original is None:
            self.procesarDatosOriginales()
            
        # Creamos una copia para no modificar el DataFrame original
        self.df_continuo = self.df_original.copy()

        # Agregamos la columna de rendimiento académico continuo
        self.agregarRendimientoAcademicoContinuo(df=self.df_continuo, metodo='promediosimple')
        
        # Normalizamos las variables continuas
        self.normalizarVariablesContinuas(df=self.df_continuo, escalador='standard')
        
        print("Procesamiento de datos continuos completado.")
        return self.df_continuo
    
    def procesarDatosCategoricos(self):
        """
        Genera el DataFrame con variables categorizadas.
        
        Retorna:
            pandas.DataFrame: DataFrame con variables categorizadas.
        """
        if self.df_original is None:
            self.procesarDatosOriginales()
            
        # Creamos una copia para no modificar el DataFrame original
        self.df_categorico = self.df_original.copy()
        
        variablesMaterias = ['matematicas', 'comprension_lectora', 'ciencias']
        
        # Categorizamos las variables de rendimiento académico (sin eliminarlas primero)
        self.categorizarRendimientoAcademicoMaterias(df=self.df_categorico, metodo='intervalosIguales')
        self.categorizarIndiceSocioEconomico(df=self.df_categorico)
        self.categorizarRendimientoGeneral(df=self.df_categorico)
        self.equivalenciaEducacionPadresCategorica(df=self.df_categorico)
        
        # AHORA podemos eliminar las variables originales si es necesario
        # Pero solo después de haber creado sus versiones categóricas
        # variablesMaterias = ['matematicas', 'comprension_lectora', 'ciencias']
        # self.eliminarColumnasDuplicadas(variablesMaterias, df=self.df_categorico)
        
        print("Procesamiento de datos categóricos completado.")
        return self.df_categorico
    
    def guardarTodosProcesados(self, prefijo='alumnosMexico2022'):
        """
        Guarda todos los DataFrames procesados.
        
        Argumentos:
            prefijo (str): Prefijo para los nombres de archivo.
            
        Retorna:
            dict: Diccionario con los nombres de los archivos guardados.
        """
        archivos = {}
        
        # Aseguramos que los DataFrames estén procesados
        if self.df_original is None:
            self.procesarDatosOriginales()
            
        if self.df_continuo is None:
            self.procesarDatosContinuos()
            
        if self.df_categorico is None:
            self.procesarDatosCategoricos()
        
        # Guardamos los DataFrames
        nombreOriginal = f"{prefijo}Procesados.csv"
        nombreContinuo = f"{prefijo}ProcesadosContinuos.csv"
        nombreCategorico = f"{prefijo}ProcesadosCategoricos.csv"
        
        archivos['original'] = self.guardarDatosProcesados(self.df_original, nombreOriginal)
        archivos['continuo'] = self.guardarDatosProcesados(self.df_continuo, nombreContinuo)
        archivos['categorico'] = self.guardarDatosProcesados(self.df_categorico, nombreCategorico)
        
        return archivos
    
    def procesarTodo(self, guardar=True, prefijo='alumnosMexico2022'):
        """
        Ejecuta el procesamiento completo del dataset, generando todas las versiones.
        
        Argumentos:
            guardar (bool): Si se deben guardar los DataFrames en archivos CSV.
            prefijo (str): Prefijo para los nombres de archivo si se guardan.
            
        Retorna:
            tuple: Tupla con los DataFrames procesados (df_original, df_continuo, df_categorico)
        """
        # Procesamos los datos originales
        self.procesarDatosOriginales()
        
        # Procesamos los datos continuos (normalizados)
        self.procesarDatosContinuos()
        
        # Procesamos los datos categóricos
        self.procesarDatosCategoricos()
        
        # Guardamos los datos si es necesario
        if guardar:
            self.guardarTodosProcesados(prefijo)
        
        return self.df_original, self.df_continuo, self.df_categorico

procesamientoMexico = ProcesamientoDatos("../Extraccion/alumnosMexico2022.csv")

print("Iniciando procesamiento de datos...")

# Procesamos todos los datos y los guardamos
df_original, df_continuo, df_categorico = procesamientoMexico.procesarTodo()

print("Procesamiento completo.")
print(f"Dimensiones del DataFrame original: {df_original.shape}")
print(f"Dimensiones del DataFrame continuo: {df_continuo.shape}")
print(f"Dimensiones del DataFrame categórico: {df_categorico.shape}")