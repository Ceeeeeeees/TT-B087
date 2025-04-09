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
        