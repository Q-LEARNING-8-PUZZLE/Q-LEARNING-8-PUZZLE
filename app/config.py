"""
Configuración de hiperparámetros y recompensas
"""

# Hiperparámetros de Q-Learning
ALPHA = 0.1          # Tasa de aprendizaje (learning rate)
GAMMA = 0.9          # Factor de descuento (discount factor)
EPSILON = 0.1        # Exploración vs explotación (epsilon-greedy)
NUM_EPISODES = 10000 # Número de episodios de entrenamiento

# Configuración de recompensas (ahora configurables desde environment.py)
REWARD_GOAL = 1000.0     # Recompensa al alcanzar el estado objetivo
REWARD_STEP = -1.0       # Recompensa por cada paso normal (incentiva rapidez)
REWARD_INVALID = -100.0  # Penalización por movimiento inválido

# Configuración de entrenamiento
MAX_STEPS_PER_EPISODE = 200  # Límite de pasos por episodio
EPSILON_DECAY = 0.995        # Factor de decaimiento de epsilon
EPSILON_MIN = 0.01           # Epsilon mínimo
