"""
Plotter - Dev 4
TASK-11: Generación de Gráficas de Rendimiento
"""
import matplotlib
# matplotlib.use('Agg') # Comentado para permitir visualización interactiva si es posible

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import sys
from typing import Optional

class Plotter:
    """
    Clase para generar gráficas de rendimiento a partir de los logs de entrenamiento.
    
    TASK-11:
    - Carga los datos desde un archivo CSV.
    - Genera y guarda las siguientes gráficas:
        - Pasos promedio vs. Episodios (con media móvil).
        - Tasa de éxito vs. Episodios (con media móvil).
    """
    def __init__(self, log_file: str = "data/training_log.csv"):
        """
        Inicializa el Plotter.
        
        Args:
            log_file (str): Ruta al archivo CSV con los logs del entrenamiento.
        """
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"El archivo de log '{log_file}' no fue encontrado. Asegúrate de que el entrenamiento se haya ejecutado primero.")
        
        self.log_file = log_file
        self.output_dir = "data/plots"
        self.df = pd.read_csv(log_file)
        
        # Asegurarse de que el directorio de salida exista
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuración estética de Seaborn
        sns.set_theme(style="darkgrid")

    def plot_steps_vs_episodes(self, window: int = 100, show: bool = False):
        """
        Genera y guarda una gráfica de los pasos promedio por episodio.
        
        Usa una media móvil para suavizar la curva y mostrar la tendencia.
        
        Args:
            window (int): Tamaño de la ventana para la media móvil.
            show (bool): Si es True, muestra la gráfica en lugar de guardarla.
        """
        plt.figure(figsize=(12, 7))
        
        self.df['steps_moving_avg'] = self.df['steps'].rolling(window=window, min_periods=1).mean()
        
        sns.lineplot(
            data=self.df,
            x='episode',
            y='steps_moving_avg',
            label=f'Media móvil (ventana={window})'
        )
        
        plt.title('Rendimiento del Agente: Pasos Promedio por Episodio', fontsize=16)
        plt.xlabel('Episodio', fontsize=12)
        plt.ylabel('Pasos Promedio para Resolver', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            save_path = os.path.join(self.output_dir, "steps_vs_episodes.png")
            plt.savefig(save_path)
            print(f"✓ Gráfica de pasos promedio guardada en: {save_path}")
            if sys.platform == "linux":
                try:
                    subprocess.run(['xdg-open', save_path])
                except FileNotFoundError:
                    print(f"Aviso: No se pudo abrir '{save_path}'. Comando 'xdg-open' no encontrado.")

        plt.close()

    def plot_success_rate_vs_episodes(self, window: int = 100, show: bool = False):
        """
        Genera y guarda una gráfica de la tasa de éxito por episodio.
        
        Usa una media móvil para mostrar la tendencia de la tasa de éxito.
        
        Args:
            window (int): Tamaño de la ventana para la media móvil.
            show (bool): Si es True, muestra la gráfica en lugar de guardarla.
        """
        plt.figure(figsize=(12, 7))
        
        # Convertir 'success' booleano a 1/0 para la media
        self.df['success_rate'] = self.df['success'].astype(int).rolling(window=window, min_periods=1).mean() * 100
        
        sns.lineplot(
            data=self.df,
            x='episode',
            y='success_rate'
        )
        
        plt.title('Rendimiento del Agente: Tasa de Éxito por Episodio', fontsize=16)
        plt.xlabel('Episodio', fontsize=12)
        plt.ylabel(f'Tasa de Éxito (%, media móvil ventana={window})', fontsize=12)
        plt.ylim(0, 105)  # Eje Y de 0% a 105% para mejor visualización
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            save_path = os.path.join(self.output_dir, "success_rate_vs_episodes.png")
            plt.savefig(save_path)
            print(f"✓ Gráfica de tasa de éxito guardada en: {save_path}")
            if sys.platform == "linux":
                try:
                    subprocess.run(['xdg-open', save_path])
                except FileNotFoundError:
                    print(f"Aviso: No se pudo abrir '{save_path}'. Comando 'xdg-open' no encontrado.")
        
        plt.close()

    def generate_all_plots(self, show: bool = False):
        """
        Genera todas las gráficas de rendimiento definidas.
        
        Args:
            show (bool): Si es True, muestra las gráficas en lugar de guardarlas.
        """
        print("\n" + "="*80)
        print("GENERANDO GRÁFICAS DE RENDIMIENTO")
        print("="*80 + "\n")
        
        self.plot_steps_vs_episodes(show=show)
        self.plot_success_rate_vs_episodes(show=show)
        
        print("\n" + "="*80)
        print("GRÁFICAS GENERADAS CORRECTAMENTE")
        print("="*80)

def generate_plots(log_file: str = "data/training_log.csv"):
    """
    Función de utilidad para generar todas las gráficas a partir de un log.
    
    Args:
        log_file (str): Ruta al archivo de log.
    """
    try:
        plotter = Plotter(log_file)
        plotter.generate_all_plots()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Por favor, ejecuta el entrenamiento principal para generar el archivo de log primero.")

# Bloque principal para ejecución directa
if __name__ == '__main__':
    # Esto permite ejecutar `python -m app.plotter` para generar las gráficas
    # de un log ya existente.
    generate_plots()