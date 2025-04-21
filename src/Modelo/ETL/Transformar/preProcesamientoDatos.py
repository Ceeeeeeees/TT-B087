import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

pd.set_option('future.no_silent_downcasting', True) # Desactivar advertencia de downcasting
class ProcesamientoDatos:

    def __init__(self, archivoAlumnos):
        self.archivoAlumnos = archivoAlumnos
        self.df = None
        
    def procesarArchivoDataFrame(self):
        """
        Permite cargar el archivo csv en un DataFrame de pandas.
        Retorna:
            pandas.DataFrame: El DataFrame cargado con los datos del archivo csv.
        """
        archivoAlumnosSinProcesar = self.archivoAlumnos
        self.df = pd.read_csv(archivoAlumnosSinProcesar)
        return self.df
    
    def eliminarColumnas(self):
        """
        Manejo de valores faltantes en el dataset.
            Eliminamos columnas que no son necesarias debido a que no contiene información durante la extracción de datos.
            Las columnas eliminadas son: 'desk', 'dishwasher', 'wealth', 'year', 'country', 'school_id', 'stu_wgt', 'computer_n'.
        Retorna:
            pandas.DataFrame: El DataFrame con las columnas eliminadas.
        """
        columnasEliminar = ['desk', 'dishwasher', 'wealth', 'year', 'country', 'school_id', 'stu_wgt', 'computer_n']
        self.df = self.df.drop(columns=columnasEliminar)
        return self.df
    
    def renombrarIndiceAlumno(self):
        """
        Permite renombrar el índice de los alumnos del DataFrame.
        Retorna:
            pandas.DataFrame: El DataFrame con el índice renombrado.
        """
        self.df['student_id'] = range(1, len(self.df) + 1)
        return self.df
    
    def valorFaltanteComputadora(self):
        """
        Manejo de valores faltantes en la columna 'computer'.
            Para ello, revisamos si la columna 'computer_n' tiene valores nulos o cero, y si es así, asignamos 'no' a la columna 'computer'.
            Si la columna 'computer_n' tiene valores mayores a 1, asignamos 'yes' a la columna 'computer'.
        Retorna:    
            pandas.DataFrame: El DataFrame con los valores faltantes en 'computer' rellenados.
        """
        self.df['computer_n'] = pd.to_numeric(self.df['computer_n'], errors='coerce') 

        self.df.loc[(self.df['computer_n'].isna()) | (self.df['computer_n'] == 0) , 'computer'] = 'no'
        self.df.loc[(self.df['computer_n'] >= 1) & (self.df['computer'].isna()), 'computer'] = 'yes'

        return self.df
    
    def valorFaltanteInternet(self):
        """
        Manejo de valores faltantes en la columna 'internet'.
            Para ello, revisamos si la columna 'internet' tiene valores nulos y si la columna 'computer' tiene valor 1, asignamos 'yes' a la columna 'internet'.
            Si la columna 'computer' tiene valor 0, asignamos 'no' a la columna 'internet'.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'internet' rellenados.
        """
        self.df['computer'] = self.df['computer'].astype(str).str.strip() # Eliminar espacios en blanco
        
        self.df.loc[(self.df['internet'].isna()) & (self.df['computer'] == '1'), 'internet'] = 'yes'
        self.df.loc[(self.df['internet'].isna()) & (self.df['computer'] == '0'), 'internet'] = 'no'
        return self.df
    
    def valorFaltanteCuarto(self):
        """
        Maneja los valores faltantes en la columna 'room'.
            Para ello, rellanamos los valores NaN en la columna 'room' con el valor más frecuente (moda) de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'room' rellenados.
        """
        self.df['room'] = self.df['room'].fillna(self.df['room'].mode()[0])     # Rellenar con la moda 
        return self.df
    
    def valorFaltanteTelevision(self):
        """
        Maneja los valores faltantes en la columna 'television'.
            Para ello, rellanamos los valores NaN en la columna 'television' con el valor más frecuente (moda) de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'television' rellenados.
        """
        self.df['television'] = self.df['television'].fillna(self.df['television'].mode()[0])     # Rellenar con la moda
        return self.df
    
    def valorFaltanteAutos(self):
        """"
        Maneja los valores faltantes en la columna 'car'.
            Para ello, rellanamos los valores NaN en la columna 'car' con el valor más frecuente (moda) de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'car' rellenados.
        """
        self.df['car'] = self.df['car'].fillna(self.df['car'].mode()[0])     # Rellenar con la moda
        return self.df

    def valorFaltanteLibros(self):
        """"
        Maneja los valores faltantes en la columna 'book'.
            Para ello, rellanamos los valores NaN en la columna 'car' con el valor más frecuente (moda) de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'book' rellenados.
        """
        self.df['book'] = self.df['book'].fillna(self.df['book'].mode()[0])     # Rellenar con la moda
        return self.df
    
    def valorFaltanteIndiceSocioEconomico(self):
        """
        Maneja los valores faltantes en la columna 'escs'.
            Para ello, rellanamos los va""lores NaN en la columna 'escs' con la mediana de dicha columna.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores faltantes en 'escs' rellenados.
        """
        self.df['escs'] = self.df['escs'].fillna(self.df['escs'].median())   # Rellenar con la mediana
        return self.df
    
    def conversionBinariaComputadora(self):
        """
        Convierte los valores de la columna 'computer' a valores binarios.
            Valores con 'no' se convierten a 0 y valores con 'yes' se convierten a 1.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'computer' convertida a valores binarios.
        """
        equivalenciaComputadora = {
            'no'                : 0,
            'yes'               : 1
        }

        self.df['computer'] = self.df['computer'].replace(equivalenciaComputadora).infer_objects(copy=False)
        return self.df

    def conversionBinariaGenero(self):
        """
        Convierte los valores de la columna 'gender' a valores binarios.
            Valores con 'male' se convierten a 0 y valores con 'female' se convierten a 1.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'gender' convertida a valores binarios.
        """
        equivalenciaGenero = {
            'male'              : 0,
            'female'            : 1
        }

        self.df['gender'] = self.df['gender'].replace(equivalenciaGenero).infer_objects(copy=False)
        return self.df
    
    def conversionBinariaInternet(self):
        """
        Convierte los valores de la columna 'internet' a valores binarios.
            Permite definir si el alumno tiene acceso a internet o no.
            Valores con 'no' se convierten a 0 y valores con 'yes' se convierten a 1.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'internet' convertida a valores binarios."""
        equivalenciaInternet = {
            'no'                : 0,
            'yes'               : 1
        }

        self.df['internet'] = self.df['internet'].replace(equivalenciaInternet).infer_objects(copy=False)
        return self.df
    
    def conversionBinariaCuarto(self):
        """
        Convierte los valores de la columna 'room' a valores binarios.
            Permite definir si el alumno tiene cuarto propio o no.
            Valores con 'no' se convierten a 0 y valores con 'yes' se convierten a 1.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'room' convertida a valores binarios.
        """
        equivalenciaCuarto = {
            'no'                : 0,
            'yes'               : 1
        }

        self.df['room'] = self.df['room'].replace(equivalenciaCuarto).infer_objects(copy=False)
        return self.df
    
    def conversionBinariaTelevision(self):
        """
        Convierte los valores de la columna 'television' a valores binarios.
            Permite definir si el alumno tiene televisión o no.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'television' convertida a valores binarios.
        """
        self.df['television'] = self.df['television'].replace({'3+' : 1}).astype('int')         # Reemplaza '3+' por 1 y convierte el tipo de dato a entero.
        self.df['television'] = self.df['television'].apply(lambda x: 1 if x > 0 else 0)        # Devuelve 1 si x es mayor que 0, 0 de otra forma.
        return self.df
    
    def conversionBinariaAutos(self):
        """
        Convierte los valores de la columna 'car' a valores binarios.
            Permite definir si el alumno tiene automóvil o no.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'car' convertida a valores binarios.
        """
        self.df['car'] = self.df['car'].replace({'3+' : 1}).astype('int')        # Reemplaza '3+' por 1 y convierte el tipo de dato a entero.
        self.df['car'] = self.df['car'].apply(lambda x: 1 if x > 0 else 0)       # Devuelve 1 si x es mayor que 0, 0 de otra forma.
        return self.df
    
    def conversionLabelEncodingLibros(self):
        """
        Convierte los valores de la columna 'book' a valores numéricos.
            Permite definir una categoria para la cantidad de libros que tiene el alumno.
        Retornna:
            pandas.DataFrame: El DataFrame con la columna 'book' convertida a valores numéricos.
        """
        equivalenciasLibros = {
            "0-10"          : 0,
            "11-25"         : 1,
            "26-100"        : 2,
            "101-200"       : 3, 
            "201-500"       : 4,     
            "more than 500" : 6
        }
        self.df['book'] = self.df['book'].replace(equivalenciasLibros).astype('int')
        return self.df

    def mostrarValoresUnicos(self):
        """
        Permite visualizar los valores únicos de cada columna del DataFrame.
        Retorna:
            pandas.DataFrame: El DataFrame con los valores únicos de cada columna.
        """
        dataFrame = self.df
        for columna in dataFrame.columns:
            print(f'Valores únicos en {columna}: {dataFrame[columna].unique()}')
        
        return dataFrame

    def equivalenciaEducacionPadres(self):
        """
        Convierte la equivalencia de la educación de los padres a su equivalente en español.
            Permite definir la educación de los padres en el DataFrame.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'mother_educ' y 'father_educ' convertida a su equivalente en español.
        """
        equivalenciasISCED = {
            "less than ISCED1"  : "menos que primaria",         # Menor que la educacion Primaria
            "ISCED 1"           : "primaria",                   # Educacion Primaria   
            "ISCED 2"           : "secundaria",                 # Educacion Secundaria
            "ISCED 3A"          : "bachillerato",               # Bachillerato
            "ISCED 3B, C"       : "bachillerato tecnico"        # Bachillerato Técnico
        }

        self.df.loc[(self.df['mother_educ'].isna()) | (self.df['mother_educ'] == 'NaN') | (self.df['mother_educ'] == 'NA'), 'mother_educ'] = "sin informacion"
        self.df.loc[(self.df['father_educ'].isna()) | (self.df['father_educ'] == 'NaN') | (self.df['father_educ'] == 'NA'), 'father_educ'] = "sin informacion"

        self.df['mother_educ'] = self.df['mother_educ'].replace(equivalenciasISCED).infer_objects(copy=True)
        self.df['father_educ'] = self.df['father_educ'].replace(equivalenciasISCED).infer_objects(copy=True)

        return self.df
    
    def equivalenciaEducacionPadresCategorica(self):
        """
        Convierte la equivalencia de la educación de los padres a su equivalente en español.
            Permite definir la educación de los padres según una categoria en el DataFrame.
        Retorna:
            pandas.DataFrame: El DataFrame con la columna 'mother_educ' y 'father_educ' convertida a su equivalente en español.
        """
        equivalenciasISCED = {
            "less than ISCED1"  : 1,         # Menor que la educacion Primaria
            "ISCED 1"           : 2,         # Educacion Primaria   
            "ISCED 2"           : 3,         # Educacion Secundaria
            "ISCED 3A"          : 4,         # Bachillerato
            "ISCED 3B, C"       : 5          # Bachillerato Técnico
        }

        self.df.loc[(self.df['mother_educ'].isna()) | (self.df['mother_educ'] == 'NaN') | (self.df['mother_educ'] == 'NA'), 'mother_educ'] = 0
        self.df.loc[(self.df['father_educ'].isna()) | (self.df['father_educ'] == 'NaN') | (self.df['father_educ'] == 'NA'), 'father_educ'] = 0

        self.df['mother_educ'] = self.df['mother_educ'].replace(equivalenciasISCED).infer_objects(copy=True)
        self.df['father_educ'] = self.df['father_educ'].replace(equivalenciasISCED).infer_objects(copy=True)

        return self.df

    def renombrarEncabezado(self):
        """
        Permite traducir y renombrar el nombre de las columnas del DataFrame.
            Se traduce el nombre de las columnas a español y se eliminan los espacios en blanco.
        Retorna:
            pandas.DataFrame: El DataFrame con los nombres de las columnas traducidos y renombrados.
        """
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
        self.df.columns = self.df.columns.str.strip()
        self.df.rename(columns=encabezado, inplace=True)
        return self.df

    def muestraValoresFaltantes(self):
        """
        Permite ver la cantidad de valores faltantes en cada columna.
        Retorna:
            pandas.Series: Serie con la cantidad de valores faltantes en cada columna del DataFrame.
        """
        dataFrame = self.df
        valoresFaltantes = dataFrame.isnull().sum()
        print (f'Valores faltantes en cada columna:\n{valoresFaltantes}')
        return valoresFaltantes

    def guardarDatosProcesados(self, nombreArchivoProcesado = 'alumnosMexico2022Procesados.csv'):
        """
        Permite guardar el archivo procesado en un nuevo archivo csv.
        Argumentos:
            str: Nombre del archivo procesado guardado.
        Retorna:
            Archivo csv: El archivo procesado guardado en el directorio actual.
        """
        dataFrame = self.df
        dataFrame.to_csv(nombreArchivoProcesado, index=False)
        print(f'Procesamiento Completado. Archivo guardado como {nombreArchivoProcesado}')
        return nombreArchivoProcesado
    
    def normalizarVariablesContinuas(self, escalador = 'standard'):
        """
        Normaliza las variables continuas del DataFrame.
            Permite normalizar las variables continuas de las columnas:
            'matematicas', 'comprension_lectora', 'ciencias', 'indice_socioeconomico'.
        Argumentos:
            Permite elegir entre dos escaladores: 'standard' o 'minmax'.
        Retorna:
            pandas.DataFrame: El DataFrame con las variables continuas normalizadas.
        """
        variablesContinuas = ['matematicas', 'comprension_lectora', 'ciencias', 'indice_socioeconomico']

        if escalador == 'standard':
            escaladorVariables = StandardScaler()
        elif escalador == 'minmax':
            escaladorVariables = MinMaxScaler()
        else:
            raise ValueError("Escalador no válido. Use 'standard' o 'minmax'.")
    
        self.df[variablesContinuas] = escaladorVariables.fit_transform(self.df[variablesContinuas])

        return self.df
    
    def guardarDatosContinuos(self, nombreArchivoContinuos = 'alumnosMexico2022ProcesadosContinuos.csv'):
        """
        """
        dataFrame = self.df
        dataFrame.to_csv(nombreArchivoContinuos, index=False)
        print(f'Procesamiento Completado. Archivo guardado como {nombreArchivoContinuos}')
        return nombreArchivoContinuos

    def crear_df_categorico(self):
        """
        Crea una copia del DataFrame con variables categorizadas.
        """
        self.df_categorico = self.df.copy()
        return self.df_categorico
    
    def categorizarRendimientoAcademico(self, metodo = 'intervalosIguales'):
        """
        """
        if self.df_categorico is None:
            raise ValueError("El DataFrame categórico no ha sido creado. Use crear_df_categorico() primero.")
        variablesMateriasCategoricas = ['matematicas', 'comprension_lectora', 'ciencias']

        for variable in variablesMateriasCategoricas:
            if variable not in self.df_categorico.columns:
                raise ValueError(f"La variable {variable} no se encuentra en el DataFrame.")
            if metodo == 'pisa':
                puntosCorte = {
                    'matematicas' :         [0, 233, 295, 358, 420, 482, 545, 607, 669, float('inf')],
                    'comprension_lectora' : [0, 185, 262, 335, 407, 480, 553, 626, 698, float('inf')],
                    'ciencias' :            [0, 261, 335, 410, 484, 559, 633, 708, float('inf')]
                }

                etiquetasCorte = ['Nivel <1b', 'Nivel 1b', 'Nivel 1a', 'Nivel 2', 'Nivel 3', 'Nivel 4', 'Nivel 5', 'Nivel 6']

                for var in variable in variablesMateriasCategoricas:
                    self.df_categorico[f'{variable}Categoria'] = pd.cut(
                        self.df_categorico[variable],
                        bins = puntosCorte[variable],
                        labels = etiquetasCorte,
                        include_lowest = True
                    )
            
            elif metodo == 'percentiles':
                quintiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                etiquetas = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']

                for variable in variablesMateriasCategoricas:
                    puntosCorte = [self.df_categorico[variable].quantile(q) for q in quintiles]
                    puntosCorte = sorted(set(puntosCorte))
                    
                    if len(puntosCorte) < 3:
                        print(f"No hay suficientes valores únicos en '{variable} para crear quintiles.")
                        self.df_categorico[f'{variable}Categoria'] = 'No categorizable'
                    else:
                        self.df_categorico[f'{variable}Categoria'] = pd.cut(
                            self.df_categorico[variable], 
                            bins = puntosCorte,
                            labels=etiquetas[:len(puntosCorte)-1],
                            include_lowest = True
                        )
            elif metodo == 'intervalosIguales':
                etiquetas = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']

                for variable in variablesMateriasCategoricas:
                    minValor = self.df_categorico[var].min()
                    maxValor = self.df_categorico[var].max()

                    puntosCorte = np.linspace(minValor, maxValor, 6)

                    self.df_categorico[f'{variable}Categoria'] = pd.cut(
                        self.df_categorico[var], 
                        bins=puntosCorte,
                        labels=etiquetas,
                        include_lowest=True
                    )
            else:
                raise ValueError("Método no válido. Puede utilizar 'pisa', 'persentiles' o 'intervalosIguales'.")
            
        return self.df_categorico
                    

    
    def procesamiento(self):
        self.procesarArchivoDataFrame()
        #self.mostrarValoresUnicos()
        #self.muestraValoresFaltantes()
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
        #self.muestraValoresFaltantes()
        #self.mostrarValoresUnicos()
        self.guardarDatosProcesados()

        #self.mostrarValoresUnicos()
        #self.muestraValoresFaltantes()

procesamientoMexico = ProcesamientoDatos("../Extraccion/alumnosMexico2022.csv")

print(f'Archivo cargado.')
procesamientoMexico.procesamiento()
print(f'Archivo procesado.')


