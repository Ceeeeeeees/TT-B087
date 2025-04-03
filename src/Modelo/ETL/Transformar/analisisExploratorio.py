import pandas as pd
import numpy as np

class AnalisisExploratorio:

    def __init__(self, archivoAlumnosPreProcesado):
        self.archivoAlumnosPreProcesado = archivoAlumnosPreProcesado
        self.df = pd.read_csv(archivoAlumnosPreProcesado)

    