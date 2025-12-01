# Implementación de la Tabla Q (Q-Table)

## Descripción

Estructura de datos que mapea `(Estado, Acción) → Valor Q` para almacenar la calidad esperada de cada acción en cada estado del 8-Puzzle.

## Estructura

```python
{
    (estado, acción): valor_q
}
```

- **Estado**: Tupla de 9 enteros `(1, 2, 3, 4, 5, 6, 7, 8, 0)`
- **Acción**: Entero 0-3 (ARRIBA, ABAJO, IZQUIERDA, DERECHA)
- **Valor Q**: Float que representa calidad esperada

## Métodos Principales

```python
class QTable:
    def __init__(self, initial_value: float = 0.0)
    def get(self, state, action) -> float
    def set(self, state, action, value: float) -> None
    def get_best_action(self, state, valid_actions) -> Optional[int]
    def get_max_q_value(self, state, valid_actions) -> float
    def size(self) -> int
```

## Características

- **Lazy initialization**: Solo crea entradas cuando se acceden
- **defaultdict**: Inicialización automática con valor por defecto
- **O(1)**: Acceso y actualización eficientes
- **Memoria**: ~28 MB máximo para ~181,440 estados

## Uso Básico

```python
from app.agent import QTable

q_table = QTable(initial_value=0.0)
q_table.set(state, action, 0.5)
valor = q_table.get(state, action)
mejor_accion = q_table.get_best_action(state, valid_actions)
```
