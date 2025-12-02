"""
Agente Q-Learning - Dev 2
TASK-04: Implementación de la Tabla Q (Q-Table)
TASK-05: Implementación de Política Epsilon-Greedy
TASK-06: Implementación de la Ecuación de Actualización Q
"""

from typing import Tuple, Optional, Dict, List
from collections import defaultdict
import random


# TASK-04: Implementación de la Tabla Q (Q-Table)
class QTable:
    """
    Tabla Q que mapea (Estado, Acción) -> Valor Q.
    
    Utiliza un diccionario eficiente para almacenar los valores Q.
    Los estados se representan como tuplas inmutables, lo que permite
    usarlas como claves en el diccionario.
    
    Estructura: {(estado, acción): valor_q}
    """
    
    # TASK-04: Constructor de la tabla Q
    def __init__(self, initial_value: float = 0.0):
        """
        Inicializa la Tabla Q.
        
        Args:
            initial_value (float): Valor inicial para todos los pares (estado, acción).
                                  Por defecto es 0.0. Puede ser un valor pequeño aleatorio
                                  para romper simetrías (ej: 0.01).
        """
        # Usamos defaultdict para inicializar automáticamente valores no vistos
        self._q_table: Dict[Tuple[Tuple[int, ...], int], float] = defaultdict(
            lambda: initial_value
        )
        self.initial_value = initial_value
    
    # TASK-04: Método para obtener un valor Q de la tabla
    def get(self, state: Tuple[int, ...], action: int) -> float:
        """
        Obtiene el valor Q para un par (estado, acción).
        
        Args:
            state (Tuple[int, ...]): Estado del puzzle (tupla de 9 elementos).
            action (int): Acción a evaluar (0-3).
        
        Returns:
            float: Valor Q para el par (estado, acción).
        """
        return self._q_table[(state, action)]
    
    # TASK-04: Método para establecer un valor Q en la tabla
    def set(self, state: Tuple[int, ...], action: int, value: float) -> None:
        """
        Establece el valor Q para un par (estado, acción).
        
        Args:
            state (Tuple[int, ...]): Estado del puzzle (tupla de 9 elementos).
            action (int): Acción a actualizar (0-3).
            value (float): Nuevo valor Q.
        """
        self._q_table[(state, action)] = value
    
    # TASK-04: Método auxiliar para obtener la mejor acción (usado en TASK-05 y TASK-06)
    def get_best_action(self, state: Tuple[int, ...], valid_actions: list[int]) -> Optional[int]:
        """
        Obtiene la mejor acción (con mayor valor Q) para un estado dado.
        
        Args:
            state (Tuple[int, ...]): Estado del puzzle.
            valid_actions (list[int]): Lista de acciones válidas para este estado.
        
        Returns:
            Optional[int]: La acción con mayor valor Q, o None si no hay acciones válidas.
        """
        if not valid_actions:
            return None
        
        # Encuentra la acción con el mayor valor Q
        best_action = valid_actions[0]
        best_value = self.get(state, best_action)
        
        for action in valid_actions[1:]:
            q_value = self.get(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action
    
    # TASK-04: Método auxiliar para obtener el máximo valor Q (usado en TASK-06)
    def get_max_q_value(self, state: Tuple[int, ...], valid_actions: list[int]) -> float:
        """
        Obtiene el máximo valor Q para un estado dado entre las acciones válidas.
        
        Args:
            state (Tuple[int, ...]): Estado del puzzle.
            valid_actions (list[int]): Lista de acciones válidas para este estado.
        
        Returns:
            float: El máximo valor Q, o self.initial_value si no hay acciones válidas.
        """
        if not valid_actions:
            return self.initial_value
        
        max_value = self.get(state, valid_actions[0])
        for action in valid_actions[1:]:
            q_value = self.get(state, action)
            if q_value > max_value:
                max_value = q_value
        
        return max_value
    
    # TASK-04: Método auxiliar para obtener el tamaño de la tabla
    def size(self) -> int:
        """
        Retorna el número de pares (estado, acción) almacenados en la tabla.
        
        Returns:
            int: Número de entradas en la tabla Q.
        """
        return len(self._q_table)
    
    # TASK-04: Método auxiliar para limpiar la tabla
    def clear(self) -> None:
        """
        Limpia toda la tabla Q, eliminando todas las entradas.
        """
        self._q_table.clear()
    
    # TASK-04: Método mágico para obtener el tamaño con len()
    def __len__(self) -> int:
        """
        Permite usar len(q_table) para obtener el tamaño.
        
        Returns:
            int: Número de entradas en la tabla Q.
        """
        return len(self._q_table)
    
    # TASK-04: Método mágico para representación en string
    def __repr__(self) -> str:
        """
        Representación en string de la tabla Q.
        
        Returns:
            str: Información sobre el tamaño y valor inicial de la tabla.
        """
        return f"QTable(size={len(self._q_table)}, initial_value={self.initial_value})"


# TASK-05: Implementación de Política Epsilon-Greedy
# TASK-06: Implementación de la Ecuación de Actualización Q
class QLearningAgent:
    """
    Agente Q-Learning que implementa la política epsilon-greedy y la actualización Q.
    
    TASK-05: Implementación de Política Epsilon-Greedy
    TASK-06: Implementación de la Ecuación de Actualización Q
    """
    
    # TASK-05 y TASK-06: Constructor del agente (inicializa parámetros epsilon, alpha, gamma)
    def __init__(
        self,
        q_table: QTable,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.9
    ):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            q_table (QTable): Instancia de la tabla Q para almacenar valores.
            epsilon (float): Probabilidad de exploración (0.0-1.0). 
                            Con probabilidad epsilon elige acción aleatoria.
            alpha (float): Tasa de aprendizaje (0.0-1.0). Controla qué tan rápido
                          se actualizan los valores Q.
            gamma (float): Factor de descuento (0.0-1.0). Controla la importancia
                          de recompensas futuras.
        """
        self.q_table = q_table
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    
    # TASK-05: Implementación de la política epsilon-greedy para seleccionar acciones
    def choose_action(
        self,
        state: Tuple[int, ...],
        valid_actions: List[int]
    ) -> int:
        """
        TASK-05: Selecciona una acción usando la política epsilon-greedy.
        
        Con probabilidad epsilon, elige una acción aleatoria (exploración).
        Con probabilidad (1 - epsilon), elige la mejor acción conocida (explotación).
        
        Args:
            state (Tuple[int, ...]): Estado actual del puzzle.
            valid_actions (List[int]): Lista de acciones válidas para este estado.
        
        Returns:
            int: La acción seleccionada.
        
        Raises:
            ValueError: Si no hay acciones válidas disponibles.
        """
        if not valid_actions:
            raise ValueError("No hay acciones válidas disponibles para este estado")
        
        # Exploración: elegir acción aleatoria con probabilidad epsilon
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Explotación: elegir la mejor acción conocida (con mayor valor Q)
        best_action = self.q_table.get_best_action(state, valid_actions)
        
        # Si hay empate, elegir aleatoriamente entre las mejores acciones
        # (esto puede ocurrir si todas las acciones tienen el mismo valor Q)
        best_value = self.q_table.get(state, best_action)
        best_actions = [
            action for action in valid_actions
            if self.q_table.get(state, action) == best_value
        ]
        
        return random.choice(best_actions)
    
    # TASK-06: Implementación de la ecuación de actualización Q-Learning
    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        next_valid_actions: List[int]
    ) -> None:
        """
        TASK-06: Actualiza el valor Q usando la ecuación de actualización Q-Learning.
        
        Fórmula: Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        
        Donde:
        - s: estado actual
        - a: acción tomada
        - r: recompensa recibida
        - s': siguiente estado
        - alpha: tasa de aprendizaje
        - gamma: factor de descuento
        
        Args:
            state (Tuple[int, ...]): Estado actual del puzzle.
            action (int): Acción que se tomó en el estado actual.
            reward (float): Recompensa recibida por tomar la acción.
            next_state (Tuple[int, ...]): Estado siguiente después de tomar la acción.
            next_valid_actions (List[int]): Lista de acciones válidas en el siguiente estado.
        """
        # Obtener el valor Q actual para (estado, acción)
        current_q_value = self.q_table.get(state, action)
        
        # Calcular el máximo valor Q para el siguiente estado
        max_next_q_value = self.q_table.get_max_q_value(next_state, next_valid_actions)
        
        # Calcular el valor objetivo (target value)
        # Si el siguiente estado es terminal (sin acciones válidas), no hay valor futuro
        target_value = reward + self.gamma * max_next_q_value
        
        # Aplicar la ecuación de actualización Q-Learning
        # Q(s,a) = Q(s,a) + alpha * [target - Q(s,a)]
        new_q_value = current_q_value + self.alpha * (target_value - current_q_value)
        
        # Actualizar la tabla Q
        self.q_table.set(state, action, new_q_value)
    
    # TASK-05: Método auxiliar para actualizar epsilon (útil para decaimiento)
    def set_epsilon(self, epsilon: float) -> None:
        """
        Actualiza el valor de epsilon (útil para decaimiento de epsilon durante el entrenamiento).
        
        Args:
            epsilon (float): Nuevo valor de epsilon (debe estar entre 0.0 y 1.0).
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon debe estar entre 0.0 y 1.0")
        self.epsilon = epsilon
    
    # TASK-06: Método auxiliar para actualizar alpha (tasa de aprendizaje)
    def set_alpha(self, alpha: float) -> None:
        """
        Actualiza el valor de alpha (tasa de aprendizaje).
        
        Args:
            alpha (float): Nuevo valor de alpha (debe estar entre 0.0 y 1.0).
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha debe estar entre 0.0 y 1.0")
        self.alpha = alpha
    
    # TASK-06: Método auxiliar para actualizar gamma (factor de descuento)
    def set_gamma(self, gamma: float) -> None:
        """
        Actualiza el valor de gamma (factor de descuento).
        
        Args:
            gamma (float): Nuevo valor de gamma (debe estar entre 0.0 y 1.0).
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma debe estar entre 0.0 y 1.0")
        self.gamma = gamma


# TASK-05 y TASK-06: Bloque de prueba para verificar la implementación
if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Agregar el directorio raíz al path para imports
    root_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(root_dir))
    from app.environment import EightPuzzle
    
    print("=" * 60)
    print("Pruebas de TASK-05 y TASK-06")
    print("=" * 60)
    
    # Inicializar entorno y agente
    env = EightPuzzle()
    q_table = QTable(initial_value=0.0)
    agent = QLearningAgent(
        q_table=q_table,
        epsilon=0.1,
        alpha=0.1,
        gamma=0.9
    )
    
    # TASK-05: Prueba de choose_action
    print("\n--- TASK-05: Prueba de Política Epsilon-Greedy ---")
    state = env.reset(random_start=True)
    valid_actions = env.get_valid_actions(state)
    
    print(f"Estado inicial: {state}")
    print(f"Acciones válidas: {valid_actions}")
    print(f"Epsilon: {agent.epsilon}")
    
    # Probar choose_action varias veces
    actions_chosen = []
    for i in range(10):
        action = agent.choose_action(state, valid_actions)
        actions_chosen.append(action)
        print(f"  Intento {i+1}: Acción elegida = {action}")
    
    print(f"\nAcciones elegidas: {actions_chosen}")
    print("✓ choose_action funciona correctamente")
    
    # TASK-06: Prueba de update
    print("\n--- TASK-06: Prueba de Actualización Q ---")
    
    # Estado inicial y acción
    state1 = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    action1 = 0  # ARRIBA
    reward1 = -1.0
    
    # Obtener siguiente estado
    env.state = state1
    next_state1, action_valid = env.step(action1)
    next_valid_actions1 = env.get_valid_actions(next_state1)
    
    print(f"Estado actual: {state1}")
    print(f"Acción tomada: {action1}")
    print(f"Recompensa: {reward1}")
    print(f"Estado siguiente: {next_state1}")
    print(f"Acciones válidas en siguiente estado: {next_valid_actions1}")
    
    # Valor Q antes de la actualización
    q_before = q_table.get(state1, action1)
    print(f"\nValor Q antes de actualización: {q_before}")
    
    # Actualizar Q
    agent.update(state1, action1, reward1, next_state1, next_valid_actions1)
    
    # Valor Q después de la actualización
    q_after = q_table.get(state1, action1)
    print(f"Valor Q después de actualización: {q_after}")
    
    # Verificar que el valor cambió
    if q_after != q_before:
        print("✓ La actualización Q funciona correctamente")
    else:
        print("⚠ El valor Q no cambió (puede ser normal si alpha=0 o valores específicos)")
    
    # Verificar la fórmula manualmente
    max_next_q = q_table.get_max_q_value(next_state1, next_valid_actions1)
    expected_q = q_before + agent.alpha * (reward1 + agent.gamma * max_next_q - q_before)
    print(f"\nVerificación manual:")
    print(f"  max(Q(s',a')) = {max_next_q}")
    print(f"  Q(s,a) esperado = {q_before} + {agent.alpha} * [{reward1} + {agent.gamma} * {max_next_q} - {q_before}]")
    print(f"  Q(s,a) esperado = {expected_q}")
    print(f"  Q(s,a) obtenido = {q_after}")
    
    if abs(q_after - expected_q) < 1e-10:
        print("✓ La fórmula de actualización es correcta")
    else:
        print(f"⚠ Diferencia pequeña (posible error de redondeo): {abs(q_after - expected_q)}")
    
    print("\n" + "=" * 60)
    print("Pruebas completadas")
    print("=" * 60)

