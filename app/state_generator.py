"""
Generador de Estados Alcanzables para el 8-Puzzle.

Este módulo implementa un algoritmo BFS (Búsqueda en Anchura) para generar
todos los estados alcanzables del 8-puzzle desde el estado objetivo.
"""

from typing import Tuple, Set, List
from collections import deque
import pickle
import os
from pathlib import Path


def get_blank_position(state: Tuple[int, ...]) -> int:
    """
    Encuentra la posición del espacio vacío (0) en el estado.
    
    Args:
        state (Tuple[int, ...]): Estado del tablero como tupla de 9 elementos.
    
    Returns:
        int: Índice (0-8) donde se encuentra el espacio vacío.
    
    Example:
        >>> get_blank_position((1, 2, 3, 4, 5, 6, 7, 8, 0))
        8
    """
    return state.index(0)


def get_neighbors(state: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Genera todos los estados vecinos válidos desde el estado actual.
    
    Un vecino es un estado que se puede alcanzar moviendo el espacio vacío
    en una de las cuatro direcciones: arriba, abajo, izquierda, derecha.
    
    Args:
        state (Tuple[int, ...]): Estado actual del tablero.
    
    Returns:
        List[Tuple[int, ...]]: Lista de estados vecinos válidos.
    
    Example:
        >>> state = (1, 2, 3, 4, 0, 5, 6, 7, 8)
        >>> neighbors = get_neighbors(state)
        >>> len(neighbors)  # Puede moverse arriba, abajo, izquierda, derecha
        4
    """
    neighbors = []
    blank_pos = get_blank_position(state)
    
    # Convertir el índice lineal a coordenadas de fila y columna
    row = blank_pos // 3
    col = blank_pos % 3
    
    # Definir los movimientos posibles: (delta_fila, delta_columna)
    # Arriba, Abajo, Izquierda, Derecha
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for delta_row, delta_col in moves:
        new_row = row + delta_row
        new_col = col + delta_col
        
        # Verificar que el movimiento esté dentro de los límites del tablero
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            # Calcular la nueva posición lineal del espacio vacío
            new_blank_pos = new_row * 3 + new_col
            
            # Crear el nuevo estado intercambiando el espacio vacío con la pieza
            state_list = list(state)
            state_list[blank_pos], state_list[new_blank_pos] = \
                state_list[new_blank_pos], state_list[blank_pos]
            
            # Agregar el nuevo estado a la lista de vecinos
            neighbors.append(tuple(state_list))
    
    return neighbors


def generate_reachable_states(goal_state: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)) -> Set[Tuple[int, ...]]:
    """
    Genera todos los estados alcanzables del 8-puzzle usando BFS.
    
    Comienza desde el estado objetivo y explora todos los estados que se pueden
    alcanzar mediante movimientos válidos. Debido a la paridad de las permutaciones,
    solo la mitad de todos los estados posibles (9!/2 = 181,440) son alcanzables.
    
    Args:
        goal_state (Tuple[int, ...]): Estado objetivo desde el cual comenzar la búsqueda.
                                       Por defecto es (1, 2, 3, 4, 5, 6, 7, 8, 0).
    
    Returns:
        Set[Tuple[int, ...]]: Conjunto de todos los estados alcanzables.
    
    Example:
        >>> states = generate_reachable_states()
        >>> len(states)
        181440
        >>> (1, 2, 3, 4, 5, 6, 7, 8, 0) in states
        True
    """
    # Conjunto para almacenar todos los estados visitados (alcanzables)
    visited: Set[Tuple[int, ...]] = set()
    
    # Cola para el algoritmo BFS
    queue: deque = deque()
    
    # Inicializar con el estado objetivo
    queue.append(goal_state)
    visited.add(goal_state)
    
    # Contador para mostrar progreso
    count = 0
    
    print("=" * 70)
    print("INICIANDO GENERACIÓN DE ESTADOS ALCANZABLES")
    print("=" * 70)
    print(f"Estado objetivo inicial: {goal_state}")
    print(f"Representación 3x3:")
    import numpy as np
    print(np.array(goal_state).reshape(3, 3))
    print("\nComenzando exploración BFS...")
    print("-" * 70)
    
    import time
    start_time = time.time()
    
    # BFS: explorar todos los estados alcanzables
    while queue:
        current_state = queue.popleft()
        count += 1
        
        # Mostrar progreso cada 5,000 estados
        if count % 5000 == 0:
            elapsed = time.time() - start_time
            states_per_sec = count / elapsed if elapsed > 0 else 0
            print(f"[Progreso] Estados procesados: {count:,} | "
                  f"En cola: {len(queue):,} | "
                  f"Visitados: {len(visited):,} | "
                  f"Velocidad: {states_per_sec:.0f} estados/seg")
        
        # Generar todos los vecinos del estado actual
        for neighbor in get_neighbors(current_state):
            # Si el vecino no ha sido visitado, agregarlo
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    elapsed_time = time.time() - start_time
    
    print("-" * 70)
    print("¡GENERACIÓN COMPLETA!")
    print("=" * 70)
    print(f"Total de estados alcanzables: {len(visited):,}")
    print(f"Tiempo total: {elapsed_time:.2f} segundos")
    print(f"Velocidad promedio: {len(visited) / elapsed_time:.0f} estados/seg")
    print("=" * 70)
    
    return visited


def save_reachable_states(states: Set[Tuple[int, ...]], filepath: str = "data/reachable_states.pkl") -> None:
    """
    Guarda el conjunto de estados alcanzables en un archivo pickle.
    
    Args:
        states (Set[Tuple[int, ...]]): Conjunto de estados a guardar.
        filepath (str): Ruta del archivo donde guardar los estados.
    
    Example:
        >>> states = generate_reachable_states()
        >>> save_reachable_states(states)
    """
    # Crear el directorio si no existe
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar los estados en formato pickle
    with open(filepath, 'wb') as f:
        pickle.dump(states, f)
    
    # Obtener el tamaño del archivo
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\nEstados guardados en: {filepath}")
    print(f"Tamaño del archivo: {file_size_mb:.2f} MB")


def load_reachable_states(filepath: str = "data/reachable_states.pkl") -> Set[Tuple[int, ...]]:
    """
    Carga el conjunto de estados alcanzables desde un archivo pickle.
    
    Args:
        filepath (str): Ruta del archivo desde donde cargar los estados.
    
    Returns:
        Set[Tuple[int, ...]]: Conjunto de estados alcanzables.
    
    Raises:
        FileNotFoundError: Si el archivo no existe.
    
    Example:
        >>> states = load_reachable_states()
        >>> len(states)
        181440
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Archivo no encontrado: {filepath}\n"
            f"Ejecuta primero el generador de estados."
        )
    
    with open(filepath, 'rb') as f:
        states = pickle.load(f)
    
    print(f"Estados cargados desde: {filepath}")
    print(f"Total de estados: {len(states)}")
    
    return states


# Bloque de prueba
if __name__ == "__main__":
    # Generar todos los estados alcanzables
    reachable_states = generate_reachable_states()
    
    # Guardar los estados en un archivo
    save_reachable_states(reachable_states)
    
    # Verificar que el estado objetivo está incluido
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    print(f"\n¿Estado objetivo en el conjunto? {goal in reachable_states}")
    
    # Verificar que un estado inválido NO está incluido
    # (intercambiar solo dos piezas crea un estado inalcanzable)
    invalid = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    print(f"¿Estado inválido en el conjunto? {invalid in reachable_states}")
