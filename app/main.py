"""
Punto de entrada principal para ejecutar el entrenamiento y la evaluación.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.environment import EightPuzzle
from app.agent import QLearningAgent, QTable
from app.trainer import Trainer
from app.config import NUM_EPISODES
from app.plotter import generate_plots

def main():
    """
    Función principal que orquesta la creación de componentes,
    el entrenamiento y la generación de gráficas.
    """
    # 1. Configuración del entorno
    env = EightPuzzle(use_reachable_states=False)

    # 2. Configuración del agente
    q_table = QTable()
    agent = QLearningAgent(q_table)

    # 3. Configuración del entrenador
    # Se usan los hiperparámetros definidos en app/config.py
    trainer = Trainer(
        environment=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        verbose=True,
        log_interval=100  # Imprimir estadísticas cada 100 episodios
    )

    # 4. Iniciar el entrenamiento
    training_stats = trainer.train()

    # 5. Generar y guardar las gráficas de rendimiento (TASK-11)
    # El log se guarda automáticamente al final de trainer.train()
    # por la modificación que hicimos.
    generate_plots()

if __name__ == "__main__":
    main()