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
    GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def __init__(self, use_reachable_states: bool = False):
        """
        Inicializa el entorno del 8-Puzzle.
        
        Args:
            use_reachable_states (bool): Si es True, carga los estados alcanzables
                                        desde el archivo generado. Útil para validación.
        """
        self.state = self.GOAL_STATE
        self.reachable_states = None
        
        # Cargar estados alcanzables si se solicita
        if use_reachable_states:
            self._load_reachable_states()
        
        self.reset()

    def generate_reachable_states_and_graph(self, verbose: bool = True) -> Tuple[Set[Tuple[int, ...]], dict]:
        """
        Genera todos los estados alcanzables desde el estado objetivo usando BFS.
        Incluye optimización para evitar explorar estados inalcanzables usando invariante de paridad.
        Args:
            verbose (bool): Si True, imprime el progreso y resultados por consola.
        Returns:
            Tuple[Set[Tuple[int, ...]], dict]: (estados alcanzables, grafo de transiciones)
        """
        from collections import deque, defaultdict
        initial_state = self.GOAL_STATE
        visited = set()
        graph = defaultdict(list)
        queue = deque([initial_state])
        steps = 0
        skipped_states = 0
        
        if verbose:
            print("[BFS] Generando estados alcanzables desde el estado objetivo...")
            print("[BFS] Aplicando optimización de invariante de paridad...")
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            # Optimización: verificar si el estado es potencialmente alcanzable usando invariante de paridad
            if not self._is_solvable(current):
                skipped_states += 1
                continue
                
            visited.add(current)
            valid_actions = self.get_valid_actions(current)
            for action in valid_actions:
                next_state = self._simulate_action(current, action)
                graph[current].append((action, next_state))
                if next_state not in visited:
                    queue.append(next_state)
            steps += 1
            if verbose and steps % 1000 == 0:
                print(f"[BFS] Estados explorados: {steps} | Cola: {len(queue)} | Visitados: {len(visited)} | Saltados: {skipped_states}")
        
        if verbose:
            print(f"[BFS] Total de estados alcanzables: {len(visited)}")
            print(f"[BFS] Estados saltados por optimización: {skipped_states}")
            print(f"[BFS] Ejemplo de estado alcanzable: {next(iter(visited))}")
        return visited, graph

    def _simulate_action(self, state: Tuple[int, ...], action: int) -> Tuple[int, ...]:
        """
        Simula una acción sobre un estado dado (sin modificar self.state).
        Args:
            state (Tuple[int, ...]): Estado sobre el que aplicar la acción.
            action (int): Acción a ejecutar.
        Returns:
            Tuple[int, ...]: Nuevo estado tras la acción.
        """
        blank_pos = state.index(0)
        row = blank_pos // 3
        col = blank_pos % 3
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
        state_list = list(state)
        state_list[blank_pos], state_list[new_blank_pos] = state_list[new_blank_pos], state_list[blank_pos]
        return tuple(state_list)
    
    def _is_solvable(self, state: Tuple[int, ...]) -> bool:
        """
        Verifica si un estado es resoluble usando el invariante de paridad.
        Para el 8-puzzle, un estado es resoluble si tiene un número par de inversiones.
        Args:
            state (Tuple[int, ...]): Estado a verificar.
        Returns:
            bool: True si el estado es resoluble, False si no.
        """
        # Contar inversiones (pares de elementos fuera de orden, excluyendo el 0)
        inversions = 0
        state_without_blank = [x for x in state if x != 0]
        
        for i in range(len(state_without_blank)):
            for j in range(i + 1, len(state_without_blank)):
                if state_without_blank[i] > state_without_blank[j]:
                    inversions += 1
        
        # El estado es resoluble si tiene un número par de inversiones
        return inversions % 2 == 0

    def reset(self, random_start: bool = False, shuffles: int = 100) -> Tuple[int, ...]:
        """
        Reinicia el entorno a un estado inicial.
        Args:
            random_start (bool): Si True, genera un estado inicial aleatorio válido.
            shuffles (int): Número de movimientos aleatorios para generar estado inicial.
        Returns:
            Tuple[int, ...]: El estado inicial del tablero.
        """
        if random_start:
            self.state = self._generate_random_solvable_state(shuffles)
        else:
            self.state = self.GOAL_STATE
        return self.state
    
    def _generate_random_solvable_state(self, shuffles: int = 100) -> Tuple[int, ...]:
        """
        Genera un estado inicial aleatorio pero resoluble mediante shuffling.
        Args:
            shuffles (int): Número de movimientos aleatorios a realizar.
        Returns:
            Tuple[int, ...]: Estado aleatorio pero resoluble.
        """
        import random
        
        current_state = self.GOAL_STATE
        for _ in range(shuffles):
            valid_actions = self.get_valid_actions(current_state)
            if valid_actions:
                action = random.choice(valid_actions)
                current_state = self._simulate_action(current_state, action)
        
        return current_state

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
    
    def get_reward(self, state: Optional[Tuple[int, ...]] = None, action_valid: bool = True) -> float:
        """
        Calcula la recompensa para un estado dado.
        Args:
            state (Optional[Tuple[int, ...]]): Estado para calcular recompensa. Si es None, usa self.state.
            action_valid (bool): Si la acción que llevó a este estado era válida.
        Returns:
            float: Valor de recompensa.
        """
        if state is None:
            state = self.state
        
        # Gran recompensa positiva por llegar al estado objetivo
        if self.is_goal(state):
            return 100.0
        
        # Penalización fuerte por movimientos inválidos
        if not action_valid:
            return -10.0
        
        # Recompensa negativa pequeña por cada paso (incentiva rapidez)
        return -1.0
    
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
    # TASK-03: Generación de estados alcanzables y grafo de transiciones
    print("\n--- Generando estados alcanzables y grafo de transiciones (TASK-03) ---")
    reachable_states, transitions_graph = env.generate_reachable_states_and_graph(verbose=True)
    print(f"\nTotal de estados alcanzables generados: {len(reachable_states)}")
    print(f"Ejemplo de transiciones para el estado objetivo:")
    for action, next_state in transitions_graph[env.GOAL_STATE]:
        print(f"Acción: {action} -> Estado: {next_state}")
    
