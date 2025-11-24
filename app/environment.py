from typing import Tuple, List, Optional, Set
import numpy as np
import os

class EightPuzzle:
    """
    Entorno para el problema del 8-Puzzle.
    El estado se representa como una tupla de 9 elementos (aplanada).
    0 representa el espacio vacío.
    """
    
    # Definimos el estado objetivo como constante de clase
    GOAL_STATE: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def __init__(self, use_reachable_states: bool = False):
        """
        Inicializa el entorno del 8-Puzzle.
        
        Args:
            use_reachable_states (bool): Si es True, carga los estados alcanzables
                                        desde el archivo generado. Útil para validación.
        """
        self.state: Tuple[int, ...] = self.GOAL_STATE
        self.reachable_states: Optional[Set[Tuple[int, ...]]] = None
        
        # Cargar estados alcanzables si se solicita
        if use_reachable_states:
            self._load_reachable_states()
        
        self.reset()

    def reset(self) -> Tuple[int, ...]:
        """
        Reinicia el entorno a un estado inicial.
        Por ahora, iniciamos en el estado objetivo.

        Returns:
            Tuple[int, ...]: El estado inicial del tablero.
        """
        self.state = self.GOAL_STATE
        return self.state

    def step(self, action: int) -> Tuple[int, ...]:
        """
        Ejecuta una acción en el entorno y devuelve el nuevo estado.
        
        Args:
            action (int): La acción a ejecutar.
                0: Arriba
                1: Abajo
                2: Izquierda
                3: Derecha
        
        Returns:
            Tuple[int, ...]: El nuevo estado del tablero.
        """
        # Encontrar la posición del espacio vacío (0)
        empty_pos = self.state.index(0)
        row, col = divmod(empty_pos, 3)
        
        new_pos = empty_pos
        
        # Calcular nueva posición basada en la acción
        # Task 2: Validar límites del tablero para evitar movimientos ilegales
        if action == 0: # Arriba
            if row > 0: # Validar límite superior
                new_pos = empty_pos - 3
        elif action == 1: # Abajo
            if row < 2: # Validar límite inferior (tablero 3x3)
                new_pos = empty_pos + 3
        elif action == 2: # Izquierda
            if col > 0: # Validar límite izquierdo
                new_pos = empty_pos - 1
        elif action == 3: # Derecha
            if col < 2: # Validar límite derecho (tablero 3x3)
                new_pos = empty_pos + 1
        
        # Si la posición cambió, actualizar el estado
        if new_pos != empty_pos:
            state_list = list(self.state)
            # Intercambiar el 0 con el valor en la nueva posición
            state_list[empty_pos], state_list[new_pos] = state_list[new_pos], state_list[empty_pos]
            self.state = tuple(state_list)
        else:
            print(self.state)
            
        return self.state


    def is_goal(self, state: Tuple[int, ...]) -> bool:
        """
        Comprueba si el estado dado es el estado objetivo.

        Args:
            state (Tuple[int, ...]): El estado a comprobar.

        Returns:
            bool: True si es el estado objetivo, False en caso contrario.
        """
        return state == self.GOAL_STATE

    def get_blank_position(self, state: Optional[Tuple[int, ...]] = None) -> int:
        """
        Encuentra la posición del espacio vacío (0) en el estado.
        
        Args:
            state (Optional[Tuple[int, ...]]): Estado a analizar. Si es None, usa self.state.
        
        Returns:
            int: Índice (0-8) donde se encuentra el espacio vacío.
        """
        if state is None:
            state = self.state
        return state.index(0)
    
    def get_valid_actions(self, state: Optional[Tuple[int, ...]] = None) -> List[int]:
        """
        Obtiene las acciones válidas desde un estado dado.
        
        Las acciones se codifican como:
        - 0: Mover espacio vacío ARRIBA
        - 1: Mover espacio vacío ABAJO
        - 2: Mover espacio vacío IZQUIERDA
        - 3: Mover espacio vacío DERECHA
        
        Args:
            state (Optional[Tuple[int, ...]]): Estado a analizar. Si es None, usa self.state.
        
        Returns:
            List[int]: Lista de acciones válidas (0-3).
        """
        if state is None:
            state = self.state
        
        blank_pos = self.get_blank_position(state)
        row = blank_pos // 3
        col = blank_pos % 3
        
        valid_actions = []
        
        # Acción 0: ARRIBA (mover pieza de arriba hacia abajo)
        if row > 0:
            valid_actions.append(0)
        
        # Acción 1: ABAJO (mover pieza de abajo hacia arriba)
        if row < 2:
            valid_actions.append(1)
        
        # Acción 2: IZQUIERDA (mover pieza de la izquierda hacia la derecha)
        if col > 0:
            valid_actions.append(2)
        
        # Acción 3: DERECHA (mover pieza de la derecha hacia la izquierda)
        if col < 2:
            valid_actions.append(3)
        
        return valid_actions
    
    def step(self, action: int) -> Tuple[Tuple[int, ...], bool]:
        """
        Ejecuta una acción y devuelve el nuevo estado.
        
        Args:
            action (int): Acción a ejecutar (0=ARRIBA, 1=ABAJO, 2=IZQUIERDA, 3=DERECHA).
        
        Returns:
            Tuple[Tuple[int, ...], bool]: (nuevo_estado, acción_válida)
                - nuevo_estado: El estado resultante después de la acción
                - acción_válida: True si la acción era válida, False si no
        """
        # Verificar si la acción es válida
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            # Acción inválida, devolver el mismo estado
            return self.state, False
        
        # Obtener posición del espacio vacío
        blank_pos = self.get_blank_position()
        row = blank_pos // 3
        col = blank_pos % 3
        
        # Calcular nueva posición según la acción
        # Mapeo de acciones a movimientos (delta_row, delta_col)
        action_to_move = {
            0: (-1, 0),  # ARRIBA
            1: (1, 0),   # ABAJO
            2: (0, -1),  # IZQUIERDA
            3: (0, 1)    # DERECHA
        }
        
        delta_row, delta_col = action_to_move[action]
        new_row = row + delta_row
        new_col = col + delta_col
        new_blank_pos = new_row * 3 + new_col
        
        # Crear el nuevo estado intercambiando las posiciones
        state_list = list(self.state)
        state_list[blank_pos], state_list[new_blank_pos] = \
            state_list[new_blank_pos], state_list[blank_pos]
        
        # Actualizar el estado interno
        self.state = tuple(state_list)
        
        return self.state, True
    
    def _load_reachable_states(self) -> None:
        """
        Carga los estados alcanzables desde el archivo pickle.
        Este método es privado y se llama automáticamente si se solicita.
        """
        filepath = "data/reachable_states.pkl"
        if os.path.exists(filepath):
            import pickle
            with open(filepath, 'rb') as f:
                self.reachable_states = pickle.load(f)
            print(f"Estados alcanzables cargados: {len(self.reachable_states)}")
        else:
            print(f"Advertencia: No se encontró {filepath}")
            print("Ejecuta test_state_generation.py para generar los estados.")
    
    def is_reachable(self, state: Optional[Tuple[int, ...]] = None) -> bool:
        """
        Verifica si un estado es alcanzable (solo si se cargaron los estados).
        
        Args:
            state (Optional[Tuple[int, ...]]): Estado a verificar. Si es None, usa self.state.
        
        Returns:
            bool: True si el estado es alcanzable, False si no o si no se cargaron estados.
        """
        if self.reachable_states is None:
            raise ValueError("Estados alcanzables no cargados. Inicializa con use_reachable_states=True")
        
        if state is None:
            state = self.state
        
        return state in self.reachable_states
    
    def render(self) -> None:
        """
        Imprime el estado actual del tablero en formato 3x3.
        """
        arr = np.array(self.state).reshape(3, 3)
        print(arr)

# Bloque de prueba simple (Solo Tarea 1)
if __name__ == "__main__":
    env = EightPuzzle()
    print("Estado inicial (Objetivo):")
    env.render()
    
    print(f"\n¿Es estado objetivo? {env.is_goal(env.state)}")
    
    # Prueba con un estado falso
    fake_state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    print(f"¿Es {fake_state} objetivo? {env.is_goal(fake_state)}")
    
