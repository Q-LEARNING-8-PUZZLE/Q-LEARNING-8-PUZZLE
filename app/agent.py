"""
Agente Q-Learning - Dev 2
TASK-04: Implementación de la Tabla Q (Q-Table)
"""

from typing import Tuple, Optional, Dict
from collections import defaultdict
import random


class QTable:
    """
    Tabla Q que mapea (Estado, Acción) -> Valor Q.
    
    Utiliza un diccionario eficiente para almacenar los valores Q.
    Los estados se representan como tuplas inmutables, lo que permite
    usarlas como claves en el diccionario.
    
    Estructura: {(estado, acción): valor_q}
    """
    
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
    
    def set(self, state: Tuple[int, ...], action: int, value: float) -> None:
        """
        Establece el valor Q para un par (estado, acción).
        
        Args:
            state (Tuple[int, ...]): Estado del puzzle (tupla de 9 elementos).
            action (int): Acción a actualizar (0-3).
            value (float): Nuevo valor Q.
        """
        self._q_table[(state, action)] = value
    
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
    
    def size(self) -> int:
        """
        Retorna el número de pares (estado, acción) almacenados en la tabla.
        
        Returns:
            int: Número de entradas en la tabla Q.
        """
        return len(self._q_table)
    
    def clear(self) -> None:
        """
        Limpia toda la tabla Q, eliminando todas las entradas.
        """
        self._q_table.clear()
    
    def __len__(self) -> int:
        """
        Permite usar len(q_table) para obtener el tamaño.
        
        Returns:
            int: Número de entradas en la tabla Q.
        """
        return len(self._q_table)
    
    def __repr__(self) -> str:
        """
        Representación en string de la tabla Q.
        
        Returns:
            str: Información sobre el tamaño y valor inicial de la tabla.
        """
        return f"QTable(size={len(self._q_table)}, initial_value={self.initial_value})"

