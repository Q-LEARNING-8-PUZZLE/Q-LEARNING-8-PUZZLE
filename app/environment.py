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
