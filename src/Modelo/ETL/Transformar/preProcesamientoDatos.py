import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
class ProcesamientoDatos:

    def __init__(self, archivoAlumnos):
        self.archivoAlumnos = archivoAlumnos
        self.df = None
        
    def procesarArchivoDataFrame(self):
        """
        Permite cargar el archivo csv en un DataFrame de pandas.
        """
        archivoAlumnosSinProcesar = self.archivoAlumnos
        self.df = pd.read_csv(archivoAlumnosSinProcesar)
        return self.df
    
    def eliminarColumnas(self):
        """
        Manejo de valores faltantes en el dataset.
        """
        columnasEliminar = ['desk', 'dishwasher', 'wealth', 'year', 'country', 'school_id', 'stu_wgt', 'computer_n']
        self.df = self.df.drop(columns=columnasEliminar)
        return self.df
    
    def renombrarIndiceAlumno(self):
        self.df['student_id'] = range(1, len(self.df) + 1)
        return self.df
    
    def valorFaltanteComputadora(self):
        self.df['computer_n'] = pd.to_numeric(self.df['computer_n'], errors='coerce')

        self.df.loc[(self.df['computer_n'].isna()) | (self.df['computer_n'] == 0) , 'computer'] = 'no'
        self.df.loc[(self.df['computer_n'] >= 1) & (self.df['computer'].isna()), 'computer'] = 'yes'

        return self.df
    
    def valorFaltanteInternet(self):
        self.df['computer'] = self.df['computer'].astype(str).str.strip() # Eliminar espacios en blanco
        
        self.df.loc[(self.df['internet'].isna()) & (self.df['computer'] == '1'), 'internet'] = 'yes'
        self.df.loc[(self.df['internet'].isna()) & (self.df['computer'] == '0'), 'internet'] = 'no'
        return self.df
    
    def valorFaltanteCuarto(self):
        self.df['room'] = self.df['room'].fillna(self.df['room'].mode()[0])     # Rellenar con la moda 
        return self.df
    
    def valorFaltanteTelevision(self):
        self.df['television'] = self.df['television'].fillna(self.df['television'].mode()[0])     # Rellenar con la moda
        return self.df
    
    def valorFaltanteAutos(self):
        self.df['car'] = self.df['car'].fillna(self.df['car'].mode()[0])     # Rellenar con la moda
        return self.df

    def valorFaltanteLibros(self):
        self.df['book'] = self.df['book'].fillna(self.df['book'].mode()[0])     # Rellenar con la moda
        return self.df
    
    def valorFaltanteIndiceSocioEconomico(self):
        self.df['escs'] = self.df['escs'].fillna(self.df['escs'].median())   # Rellenar con la mediana
        return self.df
    
    def conversionBinariaComputadora(self):
        equivalenciaComputadora = {
            'no'                : 0,
            'yes'               : 1
        }

        self.df['computer'] = self.df['computer'].replace(equivalenciaComputadora).infer_objects(copy=False)
        return self.df

    def conversionBinariaGenero(self):
        equivalenciaGenero = {
            'male'              : 0,
            'female'            : 1
        }

        self.df['gender'] = self.df['gender'].replace(equivalenciaGenero).infer_objects(copy=False)
        return self.df
    
    def conversionBinariaInternet(self):
        equivalenciaInternet = {
            'no'                : 0,
            'yes'               : 1
        }

        self.df['internet'] = self.df['internet'].replace(equivalenciaInternet).infer_objects(copy=False)
        return self.df
    
    def conversionBinariaCuarto(self):
        equivalenciaCuarto = {
            'no'                : 0,
            'yes'               : 1
        }

        self.df['room'] = self.df['room'].replace(equivalenciaCuarto).infer_objects(copy=False)
        return self.df
    
    def conversionBinariaTelevision(self):
        self.df['television'] = self.df['television'].replace({'3+' : 1}).astype('int')
        self.df['television'] = self.df['television'].apply(lambda x: 1 if x > 0 else 0)        # Devuelve 1 si x es mayor que 0, 0 de otra forma
        return self.df
    
    def conversionBinariaAutos(self):
        self.df['car'] = self.df['car'].replace({'3+' : 1}).astype('int')
        self.df['car'] = self.df['car'].apply(lambda x: 1 if x > 0 else 0)        # Devuelve 1 si x es mayor que 0, 0 de otra forma
        return self.df
    
    def conversionLabelEncodingLibros(self):
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
        Permite ver los valores únicos de cada columna.
        """
        dataFrame = self.df
        for columna in dataFrame.columns:
            print(f'Valores únicos en {columna}: {dataFrame[columna].unique()}')
        
        return dataFrame

    def equivalenciaEducacionPadres(self):
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

    def renombrarEncabezado(self):
        """
        Permite renombrar el encabezado del DataFrame.
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
        """
        dataFrame = self.df
        valoresFaltantes = dataFrame.isnull().sum()
        print (f'Valores faltantes en cada columna:\n{valoresFaltantes}')
        return valoresFaltantes

    def guardarDatosProcesados(self, nombreArchivoProcesado = 'alumnosMexico2022Procesados.csv'):
        """
        Permite guardar el archivo procesado en un nuevo archivo csv.
        """
        dataFrame = self.df
        dataFrame.to_csv(nombreArchivoProcesado, index=False)
        print(f'Procesamiento Completado. Archivo guardado como {nombreArchivoProcesado}')
        return nombreArchivoProcesado
    
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


