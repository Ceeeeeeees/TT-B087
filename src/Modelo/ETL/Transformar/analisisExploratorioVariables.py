import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuración de visualización
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Evitar notación científica en pandas
pd.set_option('display.float_format', '{:.2f}'.format)

class AnalisisExploratorio:
    """
    Clase para realizar análisis exploratorio de datos relacionados con el rendimiento académico
    de estudiantes de secundaria en México.
    """
    
    def __init__(self, ruta_archivo):
        """
        Inicializa la clase con la ruta del archivo de datos procesados.
        
        Args:
            ruta_archivo (str): Ruta al archivo CSV con los datos procesados.
        """
        self.ruta_archivo = ruta_archivo
        self.df = None
        self.directorio_imagenes = "imagenes_analisis"
        
        # Crear directorio para guardar imágenes si no existe
        if not os.path.exists(self.directorio_imagenes):
            os.makedirs(self.directorio_imagenes)
    
    def cargar_datos(self):
        """
        Carga los datos desde el archivo CSV y muestra información básica.
        
        Returns:
            pandas.DataFrame: DataFrame con los datos cargados.
        """
        try:
            self.df = pd.read_csv(self.ruta_archivo)
            print(f"Datos cargados correctamente. Dimensiones: {self.df.shape}")
            print(f"\nPrimeras 5 filas del conjunto de datos:")
            print(self.df.head())
            return self.df
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return None
    
    def estadisticas_descriptivas(self):
        """
        Calcula y muestra estadísticas descriptivas del conjunto de datos.
        
        Returns:
            pandas.DataFrame: DataFrame con estadísticas descriptivas.
        """
        if self.df is None:
            print("No hay datos cargados.")
            return None
        
        # Estadísticas generales
        print("\n### ESTADÍSTICAS DESCRIPTIVAS ###")
        desc_stats = self.df.describe(include='all').T
        desc_stats['missing'] = self.df.isnull().sum()
        desc_stats['missing_percent'] = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        
        # Calcular asimetría y curtosis para variables numéricas
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            desc_stats.loc[col, 'skewness'] = self.df[col].skew().round(2)
            desc_stats.loc[col, 'kurtosis'] = self.df[col].kurtosis().round(2)
        
        print(desc_stats)
        return desc_stats
    
    def analizar_variable_objetivo(self):
        """
        Analiza la distribución de la variable objetivo (rendimiento académico).
        """
        if self.df is None or 'rendimiento_academico' not in self.df.columns:
            print("No hay datos cargados o no se encuentra la variable objetivo.")
            return
        
        print("\n### ANÁLISIS DE LA VARIABLE OBJETIVO: RENDIMIENTO ACADÉMICO ###")
        
        # Estadísticas básicas
        print(self.df['rendimiento_academico'].describe())
        
        # Prueba de normalidad
        stat, p_value = stats.normaltest(self.df['rendimiento_academico'].dropna())
        print(f"\nPrueba de normalidad (D'Agostino y Pearson):")
        print(f"Estadística de prueba: {stat:.4f}")
        print(f"Valor p: {p_value:.4f}")
        print(f"Los datos {'no ' if p_value < 0.05 else ''}siguen una distribución normal (α=0.05)")
        
        # Visualización
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.histplot(self.df['rendimiento_academico'], kde=True)
        plt.title('Distribución del Rendimiento Académico')
        plt.xlabel('Rendimiento Académico')
        plt.ylabel('Frecuencia')
        
        plt.subplot(2, 2, 2)
        stats.probplot(self.df['rendimiento_academico'].dropna(), plot=plt)
        plt.title('Q-Q Plot Rendimiento Académico')
        
        plt.subplot(2, 2, 3)
        sns.boxplot(y=self.df['rendimiento_academico'])
        plt.title('Boxplot del Rendimiento Académico')
        plt.ylabel('Rendimiento Académico')
        
        plt.subplot(2, 2, 4)
        sns.kdeplot(self.df['rendimiento_academico'], fill=True)
        plt.title('Densidad del Rendimiento Académico')
        plt.xlabel('Rendimiento Académico')
        
        plt.tight_layout()
        plt.savefig(f"{self.directorio_imagenes}/analisis_rendimiento_academico.png")
        plt.close()
        
        print(f"\nGráfico guardado en: {self.directorio_imagenes}/analisis_rendimiento_academico.png")
    
    def histogramas_variables_numericas(self):
        """
        Genera histogramas para todas las variables numéricas.
        """
        if self.df is None:
            print("No hay datos cargados.")
            return
        
        # Identificar variables numéricas
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        # Separar variables continuas y discretas
        continuous_cols = [col for col in numeric_cols if self.df[col].nunique() > 10 
                           and col != 'id_alumno']
        discrete_cols = [col for col in numeric_cols if self.df[col].nunique() <= 10 
                         and col != 'id_alumno']
        
        print("\n### HISTOGRAMAS DE VARIABLES NUMÉRICAS ###")
        
        # Variables continuas
        if continuous_cols:
            print(f"\nVariables continuas: {continuous_cols}")
            
            n_cols = min(2, len(continuous_cols))
            n_rows = (len(continuous_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            if len(continuous_cols) == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            
            for i, col in enumerate(continuous_cols):
                if i < len(axes):
                    sns.histplot(self.df[col], kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribución de {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frecuencia')
            
            # Ocultar ejes vacíos
            for i in range(len(continuous_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{self.directorio_imagenes}/histogramas_continuas.png")
            plt.close()
            
            print(f"Gráficos guardados en: {self.directorio_imagenes}/histogramas_continuas.png")
        
        # Variables discretas
        if discrete_cols:
            print(f"\nVariables discretas: {discrete_cols}")
            
            n_cols = min(3, len(discrete_cols))
            n_rows = (len(discrete_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            
            for i, col in enumerate(discrete_cols):
                if i < len(axes):
                    sns.countplot(x=col, data=self.df, ax=axes[i])
                    axes[i].set_title(f'Frecuencia de {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Conteo')
                    
                    # Mostrar el valor exacto encima de cada barra
                    for p in axes[i].patches:
                        axes[i].annotate(format(p.get_height(), '.0f'),
                                        (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha = 'center', va = 'center',
                                        xytext = (0, 10),
                                        textcoords = 'offset points')
            
            # Ocultar ejes vacíos
            for i in range(len(discrete_cols), len(axes)):
                if i < len(axes):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{self.directorio_imagenes}/histogramas_discretas.png")
            plt.close()
            
            print(f"Gráficos guardados en: {self.directorio_imagenes}/histogramas_discretas.png")
    
    def matriz_correlacion(self):
        """
        Genera y visualiza la matriz de correlación entre variables numéricas.
        """
        if self.df is None:
            print("No hay datos cargados.")
            return
        
        # Seleccionar solo columnas numéricas (excluyendo id_alumno)
        numeric_cols = [col for col in self.df.select_dtypes(include=['int64', 'float64']).columns 
                        if col != 'id_alumno']
        
        if not numeric_cols:
            print("No hay variables numéricas para analizar.")
            return
        
        # Calcular matriz de correlación
        corr_matrix = self.df[numeric_cols].corr()
        
        print("\n### MATRIZ DE CORRELACIÓN ###")
        print(corr_matrix.round(2))
        
        # Visualizar matriz de correlación
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, 
                    mask=mask, 
                    cmap=cmap, 
                    vmax=1, 
                    vmin=-1, 
                    center=0,
                    square=True, 
                    linewidths=.5, 
                    cbar_kws={"shrink": .8}, 
                    annot=True, 
                    fmt=".2f")
        
        plt.title('Matriz de Correlación de Variables Numéricas', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.directorio_imagenes}/matriz_correlacion.png")
        plt.close()
        
        print(f"Matriz de correlación guardada en: {self.directorio_imagenes}/matriz_correlacion.png")
        
        # Identificar correlaciones significativas con rendimiento académico
        if 'rendimiento_academico' in numeric_cols:
            correlaciones_rendimiento = corr_matrix['rendimiento_academico'].sort_values(ascending=False)
            print("\nCorrelaciones con rendimiento académico (ordenadas):")
            print(correlaciones_rendimiento)
            
            # Visualizar las correlaciones más significativas con el rendimiento académico
            plt.figure(figsize=(12, 8))
            correlaciones_sin_rendimiento = correlaciones_rendimiento[correlaciones_rendimiento.index != 'rendimiento_academico']
            sns.barplot(x=correlaciones_sin_rendimiento.values, y=correlaciones_sin_rendimiento.index)
            plt.title('Correlación de Variables con Rendimiento Académico', fontsize=16)
            plt.xlabel('Coeficiente de Correlación')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.savefig(f"{self.directorio_imagenes}/correlaciones_rendimiento.png")
            plt.close()
            
            print(f"Gráfico de correlaciones con rendimiento guardado en: {self.directorio_imagenes}/correlaciones_rendimiento.png")
    
    def analisis_bivariado(self):
        """
        Realiza análisis bivariado entre rendimiento académico y otras variables.
        """
        if self.df is None or 'rendimiento_academico' not in self.df.columns:
            print("No hay datos cargados o no se encuentra la variable objetivo.")
            return
        
        print("\n### ANÁLISIS BIVARIADO CON RENDIMIENTO ACADÉMICO ###")
        
        # Variables categóricas (binarias en este caso)
        binary_cols = [col for col in self.df.columns 
                      if col not in ['id_alumno', 'rendimiento_academico', 'matematicas', 
                                   'comprension_lectora', 'ciencias', 'indice_socioeconomico', 
                                   'educacion_madre', 'educacion_padre', 'libros'] 
                      and col in self.df.select_dtypes(include=['int64']).columns]
        
        if binary_cols:
            print(f"\nAnálisis con variables binarias: {binary_cols}")
            
            n_cols = min(2, len(binary_cols))
            n_rows = (len(binary_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            
            for i, col in enumerate(binary_cols):
                if i < len(axes):
                    sns.boxplot(x=col, y='rendimiento_academico', data=self.df, ax=axes[i])
                    axes[i].set_title(f'Rendimiento Académico por {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Rendimiento Académico')
                    
                    # Realizar prueba t para comparar medias
                    group0 = self.df[self.df[col] == 0]['rendimiento_academico'].dropna()
                    group1 = self.df[self.df[col] == 1]['rendimiento_academico'].dropna()
                    
                    if len(group0) > 0 and len(group1) > 0:
                        t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)
                        significance = "Significativo" if p_val < 0.05 else "No significativo"
                        axes[i].annotate(f"t={t_stat:.2f}, p={p_val:.4f}\n{significance}", 
                                       xy=(0.5, 0.9), xycoords='axes fraction', 
                                       ha='center', va='center',
                                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Ocultar ejes vacíos
            for i in range(len(binary_cols), len(axes)):
                if i < len(axes):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{self.directorio_imagenes}/bivariado_binarias.png")
            plt.close()
            
            print(f"Gráficos guardados en: {self.directorio_imagenes}/bivariado_binarias.png")
        
        # Variables ordinales/discretas con pocos niveles
        ordinal_cols = ['educacion_madre', 'educacion_padre', 'libros']
        ordinal_cols = [col for col in ordinal_cols if col in self.df.columns]
        
        if ordinal_cols:
            print(f"\nAnálisis con variables ordinales: {ordinal_cols}")
            
            n_cols = min(2, len(ordinal_cols))
            n_rows = (len(ordinal_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            
            for i, col in enumerate(ordinal_cols):
                if i < len(axes):
                    sns.boxplot(x=col, y='rendimiento_academico', data=self.df, ax=axes[i])
                    axes[i].set_title(f'Rendimiento Académico por {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Rendimiento Académico')
                    
                    # Realizar ANOVA para comparar medias entre grupos
                    groups = [self.df[self.df[col] == val]['rendimiento_academico'].dropna() 
                            for val in sorted(self.df[col].unique())]
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) > 1:
                        f_stat, p_val = stats.f_oneway(*groups)
                        significance = "Significativo" if p_val < 0.05 else "No significativo"
                        axes[i].annotate(f"F={f_stat:.2f}, p={p_val:.4f}\n{significance}", 
                                       xy=(0.5, 0.9), xycoords='axes fraction', 
                                       ha='center', va='center',
                                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Ocultar ejes vacíos
            for i in range(len(ordinal_cols), len(axes)):
                if i < len(axes):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{self.directorio_imagenes}/bivariado_ordinales.png")
            plt.close()
            
            print(f"Gráficos guardados en: {self.directorio_imagenes}/bivariado_ordinales.png")
        
        # Variables continuas
        continuous_cols = ['matematicas', 'comprension_lectora', 'ciencias', 'indice_socioeconomico']
        continuous_cols = [col for col in continuous_cols if col in self.df.columns]
        
        if continuous_cols:
            print(f"\nAnálisis con variables continuas: {continuous_cols}")
            
            n_cols = min(2, len(continuous_cols))
            n_rows = (len(continuous_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            
            for i, col in enumerate(continuous_cols):
                if i < len(axes) and col != 'rendimiento_academico':
                    sns.scatterplot(x=col, y='rendimiento_academico', data=self.df, ax=axes[i])
                    
                    # Añadir línea de regresión
                    sns.regplot(x=col, y='rendimiento_academico', data=self.df, 
                              scatter=False, ax=axes[i], color='red')
                    
                    axes[i].set_title(f'Rendimiento Académico vs {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Rendimiento Académico')
                    
                    # Calcular y mostrar el coeficiente de correlación
                    corr, p_val = stats.pearsonr(self.df[col].dropna(), 
                                                self.df['rendimiento_academico'].dropna())
                    significance = "Significativo" if p_val < 0.05 else "No significativo"
                    axes[i].annotate(f"r={corr:.2f}, p={p_val:.4f}\n{significance}", 
                                   xy=(0.05, 0.95), xycoords='axes fraction', 
                                   ha='left', va='top',
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Ocultar ejes vacíos
            for i in range(len([c for c in continuous_cols if c != 'rendimiento_academico']), len(axes)):
                if i < len(axes):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{self.directorio_imagenes}/bivariado_continuas.png")
            plt.close()
            
            print(f"Gráficos guardados en: {self.directorio_imagenes}/bivariado_continuas.png")
    
    def pairplot_materias(self):
        """
        Genera un pairplot para visualizar relaciones entre las calificaciones por materia.
        """
        if self.df is None:
            print("No hay datos cargados.")
            return
        
        materias = ['matematicas', 'comprension_lectora', 'ciencias']
        materias = [m for m in materias if m in self.df.columns]
        
        if len(materias) < 2:
            print("No hay suficientes variables de materias para crear un pairplot.")
            return
        
        print("\n### PAIRPLOT DE MATERIAS ###")
        
        # Crear un DataFrame con las columnas de interés y una muestra aleatoria para mejor visualización
        if len(self.df) > 1000:
            muestra = self.df.sample(n=1000, random_state=42)
        else:
            muestra = self.df
            
        # Generar pairplot
        g = sns.pairplot(muestra[materias + ['rendimiento_academico']], 
                        diag_kind='kde', 
                        plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'k'},
                        diag_kws={'fill': True})
        
        g.fig.suptitle('Relaciones entre Calificaciones por Materia', y=1.02, fontsize=16)
        g.fig.tight_layout()
        g.fig.savefig(f"{self.directorio_imagenes}/pairplot_materias.png")
        plt.close()
        
        print(f"Pairplot guardado en: {self.directorio_imagenes}/pairplot_materias.png")
    
    def analisis_factores_socioeconomicos(self):
        """
        Analiza la relación entre factores socioeconómicos y rendimiento académico.
        """
        if self.df is None:
            print("No hay datos cargados.")
            return
        
        print("\n### ANÁLISIS DE FACTORES SOCIOECONÓMICOS ###")
        
        # Variables socioeconómicas
        socioeconomic_vars = ['indice_socioeconomico', 'educacion_madre', 'educacion_padre', 
                              'libros', 'computadora', 'internet', 'cuarto_propio', 
                              'television', 'auto']
        
        socioeconomic_vars = [col for col in socioeconomic_vars if col in self.df.columns]
        
        if 'indice_socioeconomico' in self.df.columns and 'rendimiento_academico' in self.df.columns:
            # Análisis del índice socioeconómico vs rendimiento
            plt.figure(figsize=(10, 6))
            sns.regplot(x='indice_socioeconomico', y='rendimiento_academico', data=self.df,
                       scatter_kws={'alpha': 0.3})
            
            plt.title('Índice Socioeconómico vs Rendimiento Académico')
            plt.xlabel('Índice Socioeconómico')
            plt.ylabel('Rendimiento Académico')
            
            # Calcular estadísticas
            corr, p_val = stats.pearsonr(self.df['indice_socioeconomico'].dropna(), 
                                        self.df['rendimiento_academico'].dropna())
            
            plt.annotate(f"Correlación de Pearson: {corr:.3f}\nValor p: {p_val:.6f}", 
                        xy=(0.05, 0.95), xycoords='axes fraction', 
                        ha='left', va='top', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.savefig(f"{self.directorio_imagenes}/socioeconomico_vs_rendimiento.png")
            plt.close()
            
            print(f"Gráfico guardado en: {self.directorio_imagenes}/socioeconomico_vs_rendimiento.png")
        
        # Análisis de las variables socioeconomicas con múltiples categorías
        multicategory_vars = ['educacion_madre', 'educacion_padre', 'libros']
        multicategory_vars = [col for col in multicategory_vars if col in self.df.columns]
        
        if multicategory_vars and 'rendimiento_academico' in self.df.columns:
            fig, axes = plt.subplots(len(multicategory_vars), 1, figsize=(12, 5*len(multicategory_vars)))
            
            if len(multicategory_vars) == 1:
                axes = np.array([axes])
            
            for i, var in enumerate(multicategory_vars):
                # Calcular medias por grupo
                means = self.df.groupby(var)['rendimiento_academico'].mean().reset_index()
                
                # Ordenar por valor de categoría (asumiendo que son ordinales)
                means = means.sort_values(var)
                
                # Graficar
                sns.barplot(x=var, y='rendimiento_academico', data=means, ax=axes[i])
                
                axes[i].set_title(f'Rendimiento Académico Promedio por {var}')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Rendimiento Académico Promedio')
                
                # Añadir etiquetas de valor
                for p in axes[i].patches:
                    axes[i].annotate(f"{p.get_height():.2f}", 
                                   (p.get_x() + p.get_width() / 2., p.get_height()),
                                   ha = 'center', va = 'bottom',
                                   xytext = (0, 5),
                                   textcoords = 'offset points')
            
            plt.tight_layout()
            plt.savefig(f"{self.directorio_imagenes}/categorias_socioeconomicas.png")
            plt.close()
            
            print(f"Gráfico guardado en: {self.directorio_imagenes}/categorias_socioeconomicas.png")
        
        # Análisis de variables binarias socioeconómicas
        binary_vars = ['computadora', 'internet', 'cuarto_propio', 'television', 'auto']
        binary_vars = [col for col in binary_vars if col in self.df.columns]
        
        if binary_vars and 'rendimiento_academico' in self.df.columns:
            fig, axes = plt.subplots(len(binary_vars), 1, figsize=(12, 5*len(binary_vars)))
            
            if len(binary_vars) == 1:
                axes = np.array([axes])
            
            for i, var in enumerate(binary_vars):
                # Calcular medias por grupo
                means = self.df.groupby(var)['rendimiento_academico'].mean().reset_index()
                counts = self.df.groupby(var)['rendimiento_academico'].count().reset_index()
                
                merged = means.merge(counts, on=var)
                merged.columns = [var, 'mean', 'count']
                merged['percentage'] = merged['count'] / merged['count'].sum() * 100
                
                # Graficar
                sns.barplot(x=var, y='mean', data=merged, ax=axes[i])
                
                axes[i].set_title(f'Rendimiento Académico Promedio por {var}')
                axes[i].set_xlabel(f"{var} (0=No, 1=Sí)")
                axes[i].set_ylabel('Rendimiento Académico Promedio')
                
                # Añadir etiquetas de valor y porcentaje
                for j, p in enumerate(axes[i].patches):
                    perc = merged.iloc[j]['percentage']
                    count = merged.iloc[j]['count']
                    
                    axes[i].annotate(f"{p.get_height():.2f}\n({perc:.1f}%, n={count})", 
                                   (p.get_x() + p.get_width() / 2., p.get_height()),
                                   ha = 'center', va = 'bottom',
                                   xytext = (0, 5),
                                   textcoords = 'offset points')
            
            plt.tight_layout()
            plt.savefig(f"{self.directorio_imagenes}/binarias_socioeconomicas.png")
            plt.close()
            
            print(f"Gráfico guardado en: {self.directorio_imagenes}/binarias_socioeconomicas.png")
    
    def analisis_por_genero(self):
        """
        Analiza diferencias en rendimiento académico por género.
        """
        if self.df is None or 'genero' not in self.df.columns or 'rendimiento_academico' not in self.df.columns:
            print("No hay datos cargados o no se encuentran las variables necesarias.")
            return
        
        print("\n### ANÁLISIS POR GÉNERO ###")
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Boxplot de rendimiento por género
        sns.boxplot(x='genero', y='rendimiento_academico', data=self.df, ax=axes[0, 0])
        axes[0, 0].set_title('Rendimiento Académico por Género')
        axes[0, 0].set_xlabel('Género (0=Masculino, 1=Femenino)')
        axes[0, 0].set_ylabel('Rendimiento Académico')
        
        # Realizar prueba t
        masc = self.df[self.df['genero'] == 0]['rendimiento_academico'].dropna()
        fem = self.df[self.df['genero'] == 1]['rendimiento_academico'].dropna()
        
        if len(masc) > 0 and len(fem) > 0:
            t_stat, p_val = stats.ttest_ind(masc, fem, equal_var=False)
            significance = "Significativo" if p_val < 0.05 else "No significativo"
            axes[0, 0].annotate(f"t={t_stat:.2f}, p={p_val:.4f}\n{significance}", 
                              xy=(0.5, 0.95), xycoords='axes fraction', 
                              ha='center', va='top',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Histograma de rendimiento académico por género
        sns.histplot(data=self.df, x='rendimiento_academico', hue='genero', 
                    multiple="dodge", kde=True, ax=axes[0, 1],
                    palette=['skyblue', 'salmon'],
                    legend=True)
        axes[0, 1].set_title('Distribución del Rendimiento Académico por Género')
        axes[0, 1].set_xlabel('Rendimiento Académico')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].legend(['Masculino', 'Femenino'], title='Género')
        
        # Rendimiento por materia y género
        materias = ['matematicas', 'comprension_lectora', 'ciencias']
        materias_disponibles = [m for m in materias if m in self.df.columns]
        
        if materias_disponibles:
            # Crear DataFrame para visualización
            df_materias = self.df[['genero'] + materias_disponibles].melt(
                id_vars=['genero'], 
                value_vars=materias_disponibles,
                var_name='Materia', 
                value_name='Calificación'
            )
            
            # Boxplot de materias por género
            sns.boxplot(x='Materia', y='Calificación', hue='genero', data=df_materias, ax=axes[1, 0],
                      palette=['skyblue', 'salmon'])
            axes[1, 0].set_title('Calificaciones por Materia y Género')
            axes[1, 0].set_xlabel('Materia')
            axes[1, 0].set_ylabel('Calificación')
            axes[1, 0].legend(['Masculino', 'Femenino'], title='Género')
            
            # Calcular y mostrar promedios
            promedios = self.df.groupby('genero')[materias_disponibles].mean().reset_index()
            promedios_long = promedios.melt(id_vars=['genero'], 
                                         value_vars=materias_disponibles,
                                         var_name='Materia', 
                                         value_name='Promedio')
            
            sns.barplot(x='Materia', y='Promedio', hue='genero', data=promedios_long, ax=axes[1, 1],
                      palette=['skyblue', 'salmon'])
            axes[1, 1].set_title('Calificación Promedio por Materia y Género')
            axes[1, 1].set_xlabel('Materia')
            axes[1, 1].set_ylabel('Calificación Promedio')
            axes[1, 1].legend(['Masculino', 'Femenino'], title='Género')
            
            # Añadir valores a las barras
            for p in axes[1, 1].patches:
                axes[1, 1].annotate(f"{p.get_height():.2f}", 
                                  (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha = 'center', va = 'bottom',
                                  xytext = (0, 5),
                                  textcoords = 'offset points')
        
        plt.tight_layout()
        plt.savefig(f"{self.directorio_imagenes}/analisis_genero.png")
        plt.close()
        
        print(f"Gráfico guardado en: {self.directorio_imagenes}/analisis_genero.png")
        
        # Calcular estadísticas descriptivas por género
        if 'genero' in self.df.columns:
            stats_genero = self.df.groupby('genero')['rendimiento_academico'].agg(
                ['count', 'mean', 'std', 'min', 'max']).reset_index()
            stats_genero.columns = ['Género', 'Conteo', 'Media', 'Desv. Estándar', 'Mínimo', 'Máximo']
            stats_genero['Género'] = stats_genero['Género'].replace({0: 'Masculino', 1: 'Femenino'})
            
            print("\nEstadísticas por género:")
            print(stats_genero)
    
    def identificar_factores_importantes(self):
        """
        Identifica y muestra los factores más importantes que influyen en el rendimiento académico.
        """
        if self.df is None or 'rendimiento_academico' not in self.df.columns:
            print("No hay datos cargados o no se encuentra la variable objetivo.")
            return
        
        print("\n### FACTORES IMPORTANTES QUE INFLUYEN EN EL RENDIMIENTO ACADÉMICO ###")
        
        # Excluir variables no predictivas o ya incluidas en rendimiento
        exclude_cols = ['id_alumno', 'rendimiento_academico', 'matematicas', 'comprension_lectora', 'ciencias']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        if not feature_cols:
            print("No hay variables predictoras disponibles.")
            return
        
        # Correlaciones con rendimiento académico
        correlaciones = self.df[feature_cols + ['rendimiento_academico']].corr()['rendimiento_academico'].sort_values(ascending=False)
        correlaciones = correlaciones[correlaciones.index != 'rendimiento_academico']
        
        print("\nCorrelaciones con rendimiento académico:")
        print(correlaciones)
        
        # Visualizar factores importantes
        plt.figure(figsize=(12, 8))
        sns.barplot(x=correlaciones.values, y=correlaciones.index)
        plt.title('Factores que Influyen en el Rendimiento Académico', fontsize=16)
        plt.xlabel('Correlación con Rendimiento Académico')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Colorear barras según si la correlación es positiva o negativa
        for i, patch in enumerate(plt.gca().patches):
            if correlaciones.values[i] > 0:
                patch.set_facecolor('green')
            else:
                patch.set_facecolor('red')
        
        plt.savefig(f"{self.directorio_imagenes}/factores_importantes.png")
        plt.close()
        
        print(f"Gráfico guardado en: {self.directorio_imagenes}/factores_importantes.png")
        
        # Análisis adicional de las variables más importantes
        top_features = correlaciones.abs().sort_values(ascending=False).index[:5]
        
        if len(top_features) > 0:
            print(f"\nLas 5 variables más influyentes son: {list(top_features)}")
            
            print("\nEstadísticas descriptivas de las variables más influyentes:")
            print(self.df[list(top_features)].describe())
    
    def ejecutar_analisis_completo(self):
        """
        Ejecuta todos los análisis en orden.
        """
        print("INICIANDO ANÁLISIS EXPLORATORIO DE DATOS")
        print("="*50)
        
        # Cargar datos
        self.cargar_datos()
        
        # Ejecutar todos los análisis
        self.estadisticas_descriptivas()
        self.analizar_variable_objetivo()
        self.histogramas_variables_numericas()
        self.matriz_correlacion()
        self.analisis_bivariado()
        self.pairplot_materias()
        self.analisis_factores_socioeconomicos()
        self.analisis_por_genero()
        self.identificar_factores_importantes()
        
        print("\n"+"="*50)
        print(f"ANÁLISIS EXPLORATORIO COMPLETADO. Todos los gráficos han sido guardados en: {self.directorio_imagenes}")
        print("="*50)

# Ejecutar el análisis
if __name__ == "__main__":
    ruta_archivo = "alumnosMexico2022_procesados.csv"
    
    analisis = AnalisisExploratorio(ruta_archivo)
    analisis.ejecutar_analisis_completo()