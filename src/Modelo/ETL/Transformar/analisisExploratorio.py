import pandas as pd
import numpy as np

class AnalisisExploratorio:

    def __init__(self, archivoAlumnosPreProcesado):
        """
        Inicializa la clase con el archivo CSV de alumnos y carga los datos.
        
        Argumentos:
        
            archivoAlumnosPreProcesado (str): Ruta del archivo CSV procesado.
        """
        self.archivoAlumnosPreProcesado = archivoAlumnosPreProcesado
        self.df = pd.read_csv(archivoAlumnosPreProcesado)
        self.mapeo_binario = {'no': 0, 'si': 1}

    def informacionGeneral(self):
        """
        Muestra la información general del DataFrame, incluyendo el número de filas y columnas,
        los nombres de las columnas y los tipos de datos.
        """
        print("Información general del DataFrame:")
        print(f"Numero de filas: {self.df.shape[0]}")
        print(f"Numero de variables: {self.df.shape[1]}")
        print(f"Nombres de las variables: {self.df.columns.tolist()}")
        print(f"\n\nTipos de datos: {self.df.dtypes}")

        print(f"\nValores nulos por columna: {self.df.isnull().sum()}")

        print(f"Resumen estadístico:\n{self.df.describe(include='all').round(2)}")

    def prepararDatos(self):
        df_temp = self.df.copy()

        for columna in ['computadora', 'internet', 'cuarto_propio', 'television', 'auto']:
            df_temp[f'{columna}_cat'] = df_temp[columna].map(self.mapeo_binario)

        # Categorizar rendimiento académico (promedio de las tres materias)
        df_temp['promedio'] = df_temp[['matematicas', 'comprension_lectora', 'ciencias']].mean(axis=1)
        bins = [0, 400, 500, 600, 700, np.inf]
        labels = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']
        df_temp['rendimiento'] = pd.cut(df_temp['promedio'], bins=bins, labels=labels)
        
        # Categorizar nivel socioeconómico
        bins = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
        labels = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']
        df_temp['nivel_socioeconomico'] = pd.cut(df_temp['indice_socioeconomico'], bins=bins, labels=labels)
        
        return df_temp
    
analisis = AnalisisExploratorio('alumnosMexico2022Procesados.csv')
analisis.informacionGeneral()
#analisis_preparados = analisis.prepararDatos()