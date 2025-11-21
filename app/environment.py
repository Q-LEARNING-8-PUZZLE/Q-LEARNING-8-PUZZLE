import numpy as np
import random

class EightPuzzle:
    """
    Entorno para el problema del 8-Puzzle.
    El estado se representa como una tupla de 9 elementos (aplanada).
    0 representa el espacio vacío.
    """

    def __init__(self):
        # Definimos el estado objetivo: 1, 2, 3, 4, 5, 6, 7, 8, 0
        self.goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.reset()

    def reset(self):
        """
        Reinicia el entorno a un estado inicial.
        Por ahora, iniciamos en el estado objetivo.
        """
        self.state = self.goal_state
        return self.state

    def is_goal(self, state):
        """
        Comprueba si el estado dado es el estado objetivo.
        """
        return state == self.goal_state

    def render(self):
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
    print(f"¿Es (1, 2, 3, 4, 5, 6, 7, 0, 8) objetivo? {env.is_goal(fake_state)}")
