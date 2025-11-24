from typing import Tuple
import numpy as np

class EightPuzzle:
    """
    Entorno para el problema del 8-Puzzle.
    El estado se representa como una tupla de 9 elementos (aplanada).
    0 representa el espacio vacío.
    """
    
    # Definimos el estado objetivo como constante de clase
    GOAL_STATE: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def __init__(self):
        self.state: Tuple[int, ...] = self.GOAL_STATE
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
    
