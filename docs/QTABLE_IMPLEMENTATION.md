# Implementación de la Tabla Q (Q-Table)

## 📋 Descripción General

La **Tabla Q** es la estructura de datos fundamental del algoritmo Q-Learning. Almacena los valores Q que representan la calidad esperada de tomar una acción específica en un estado dado. Esta implementación está diseñada para manejar eficientemente la gran cantidad de estados posibles del 8-Puzzle (aproximadamente 181,440 estados solucionables).

## 🎯 Objetivo

Crear una estructura de datos eficiente que mapee `(Estado, Acción) → Valor Q`, permitiendo:
- Almacenar valores Q de forma eficiente
- Inicializar valores automáticamente cuando se accede a nuevos pares (estado, acción)
- Obtener y actualizar valores Q fácilmente
- Encontrar la mejor acción para un estado dado

## 🏗️ Arquitectura de la Implementación

### Estructura de Datos

La implementación utiliza un **diccionario de Python** (`defaultdict`) con la siguiente estructura:

```python
{
    (estado, acción): valor_q
}
```

Donde:
- **Estado**: Tupla inmutable de 9 enteros que representa la configuración del tablero
  - Ejemplo: `(1, 2, 3, 4, 5, 6, 7, 8, 0)`
- **Acción**: Entero que representa la acción (0-3)
  - `0`: Mover espacio vacío ARRIBA
  - `1`: Mover espacio vacío ABAJO
  - `2`: Mover espacio vacío IZQUIERDA
  - `3`: Mover espacio vacío DERECHA
- **Valor Q**: Número flotante que representa la calidad esperada de la acción

### Ejemplo Visual

```
Tabla Q:
┌─────────────────────────────────────┬─────────┐
│ (Estado, Acción)                   │ Valor Q │
├─────────────────────────────────────┼─────────┤
│ ((1,2,3,4,5,6,7,8,0), 0)          │  0.5    │
│ ((1,2,3,4,5,6,7,8,0), 1)          │  0.3    │
│ ((2,1,3,4,5,6,7,8,0), 0)          │  0.1    │
│ ((2,1,3,4,5,6,7,8,0), 2)          │  0.7    │
│ ...                                 │  ...    │
└─────────────────────────────────────┴─────────┘
```

## 📝 Componentes Principales

### 1. Inicialización (`__init__`)

```python
def __init__(self, initial_value: float = 0.0):
    self._q_table = defaultdict(lambda: initial_value)
    self.initial_value = initial_value
```

**Características:**
- Usa `defaultdict` para inicialización automática (lazy initialization)
- No necesita pre-inicializar todos los estados posibles
- Valor inicial por defecto: `0.0` (puede cambiarse)
- Eficiente en memoria: solo almacena estados visitados

**Ventajas:**
- ✅ No requiere memoria para estados nunca visitados
- ✅ Inicialización automática al acceder a nuevos pares
- ✅ Flexible: permite cambiar el valor inicial

### 2. Obtener Valor Q (`get`)

```python
def get(self, state: Tuple[int, ...], action: int) -> float:
    return self._q_table[(state, action)]
```

**Funcionalidad:**
- Retorna el valor Q para un par (estado, acción)
- Si el par no existe, `defaultdict` retorna automáticamente `initial_value`
- Operación O(1) en promedio (hash map)

**Ejemplo:**
```python
valor = q_table.get((1, 2, 3, 4, 5, 6, 7, 8, 0), 0)
# Si no existe: retorna 0.0 (valor inicial)
# Si existe: retorna el valor almacenado
```

### 3. Establecer Valor Q (`set`)

```python
def set(self, state: Tuple[int, ...], action: int, value: float) -> None:
    self._q_table[(state, action)] = value
```

**Funcionalidad:**
- Establece o actualiza el valor Q para un par (estado, acción)
- Operación O(1) en promedio

**Ejemplo:**
```python
q_table.set((1, 2, 3, 4, 5, 6, 7, 8, 0), 0, 0.5)
# Ahora Q((1,2,3,4,5,6,7,8,0), 0) = 0.5
```

### 4. Obtener Mejor Acción (`get_best_action`)

```python
def get_best_action(self, state: Tuple[int, ...], valid_actions: list[int]) -> Optional[int]:
    # Encuentra la acción con mayor valor Q entre las acciones válidas
    ...
```

**Funcionalidad:**
- Encuentra la acción con el mayor valor Q para un estado dado
- Solo considera acciones válidas (que no violan límites del tablero)
- Retorna `None` si no hay acciones válidas

**Uso en Q-Learning:**
- Utilizado durante la **explotación** (cuando epsilon-greedy elige la mejor acción conocida)
- Parte esencial de la política epsilon-greedy

**Ejemplo:**
```python
state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
valid_actions = [1, 3]  # Solo puede mover abajo o derecha
mejor_accion = q_table.get_best_action(state, valid_actions)
# Retorna la acción con mayor Q entre [1, 3]
```

### 5. Obtener Máximo Valor Q (`get_max_q_value`)

```python
def get_max_q_value(self, state: Tuple[int, ...], valid_actions: list[int]) -> float:
    # Retorna el máximo valor Q entre las acciones válidas
    ...
```

**Funcionalidad:**
- Retorna el máximo valor Q para un estado dado
- Esencial para la ecuación de actualización Q-Learning

**Uso en Q-Learning:**
- Necesario para calcular `max Q(s', a')` en la ecuación:
  ```
  Q(s,a) ← Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
  ```

**Ejemplo:**
```python
next_state = (2, 1, 3, 4, 5, 6, 7, 8, 0)
next_valid_actions = [0, 1, 2, 3]
max_q = q_table.get_max_q_value(next_state, next_valid_actions)
# Usado en la actualización Q-Learning
```

### 6. Métodos Auxiliares

#### `size()` / `__len__()`
Retorna el número de pares (estado, acción) almacenados en la tabla.

```python
tamaño = q_table.size()  # o len(q_table)
```

#### `clear()`
Limpia toda la tabla Q, eliminando todas las entradas.

```python
q_table.clear()
```

#### `__repr__()`
Representación legible de la tabla Q.

```python
print(q_table)  # QTable(size=1234, initial_value=0.0)
```

## 🔄 Flujo de Uso en Q-Learning

### 1. Inicialización
```python
from app.agent import QTable, QLearningAgent

q_table = QTable(initial_value=0.0)
agent = QLearningAgent(q_table, epsilon=0.1, alpha=0.1, gamma=0.9)
```

### 2. Durante el Entrenamiento (Usando QLearningAgent)

```python
# Paso 1: Obtener estado actual
state = env.state

# Paso 2: Obtener acciones válidas
valid_actions = env.get_valid_actions(state)

# Paso 3: Elegir acción usando epsilon-greedy (TASK-05)
action = agent.choose_action(state, valid_actions)

# Paso 4: Ejecutar acción y obtener recompensa
next_state, action_valid = env.step(action)
reward = env.get_reward(next_state, action_valid)

# Paso 5: Obtener acciones válidas del siguiente estado
next_valid_actions = env.get_valid_actions(next_state)

# Paso 6: Actualizar Q usando la ecuación Q-Learning (TASK-06)
agent.update(state, action, reward, next_state, next_valid_actions)
```

**Nota**: Para uso manual sin `QLearningAgent`, ver la sección anterior. Sin embargo, se recomienda usar `QLearningAgent` para código más limpio y mantenible.

## ⚡ Optimizaciones y Consideraciones

### 1. Lazy Initialization
- **No pre-inicializa** todos los estados posibles
- Solo crea entradas cuando se accede a ellas
- Ahorra memoria significativamente

### 2. Uso de Tuplas Inmutables
- Los estados son tuplas inmutables → pueden usarse como claves de diccionario
- Hash eficiente y seguro
- Comparación rápida

### 3. defaultdict vs dict normal
- **defaultdict**: Inicialización automática
- **dict normal**: Requeriría verificar existencia antes de acceder
- `defaultdict` simplifica el código y mejora rendimiento

### 4. Escalabilidad
- Maneja eficientemente ~181,440 estados solucionables
- Cada entrada ocupa aproximadamente:
  - Clave (tupla + int): ~100 bytes
  - Valor (float): 8 bytes
  - Overhead de dict: ~50 bytes
  - **Total por entrada**: ~158 bytes
- **Memoria estimada máxima**: ~28 MB (muy manejable)

## 📊 Comparación con Alternativas

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **Diccionario (implementación actual)** | ✅ Eficiente O(1) acceso<br>✅ Lazy initialization<br>✅ Flexible | ⚠️ Overhead de hash |
| **Array/Matriz** | ✅ Acceso rápido | ❌ Requiere indexación compleja<br>❌ Memoria fija grande<br>❌ Difícil de implementar |
| **Base de datos** | ✅ Persistencia | ❌ Muy lento<br>❌ Overhead innecesario |

## ✅ Criterios de Aceptación Cumplidos

### ✅ Estructura Eficiente
- Utiliza diccionario/hash map con tuplas como claves
- Acceso O(1) en promedio
- Lazy initialization para eficiencia de memoria

### ✅ Inicialización Correcta
- Valores inicializados automáticamente con `defaultdict`
- Valor inicial configurable
- Manejo correcto de estados no visitados

## 🧪 Ejemplo de Uso Completo

```python
from app.agent import QTable
from app.environment import EightPuzzle

# Crear tabla Q
q_table = QTable(initial_value=0.0)

# Crear entorno
env = EightPuzzle()
env.reset()

# Estado inicial
state = env.state
print(f"Estado inicial: {state}")

# Obtener acciones válidas
valid_actions = env.get_valid_actions(state)
print(f"Acciones válidas: {valid_actions}")

# Obtener valores Q iniciales (todos 0.0)
for action in valid_actions:
    q_value = q_table.get(state, action)
    print(f"Q({state}, {action}) = {q_value}")

# Establecer algunos valores Q
q_table.set(state, 0, 0.5)
q_table.set(state, 1, 0.3)
q_table.set(state, 2, 0.7)
q_table.set(state, 3, 0.2)

# Obtener mejor acción
best_action = q_table.get_best_action(state, valid_actions)
print(f"Mejor acción: {best_action}")

# Obtener máximo valor Q
max_q = q_table.get_max_q_value(state, valid_actions)
print(f"Máximo Q: {max_q}")

# Ver tamaño de la tabla
print(f"Tamaño de tabla: {q_table.size()}")
print(f"Representación: {q_table}")
```

## 📚 Referencias

- **Q-Learning Algorithm**: Algoritmo de aprendizaje por refuerzo que aprende la función Q
- **8-Puzzle Problem**: Problema de búsqueda clásico con ~181,440 estados solucionables
- **Hash Maps**: Estructura de datos eficiente para búsqueda O(1)
- **Lazy Initialization**: Patrón de diseño que inicializa valores solo cuando se necesitan

## 🔗 Relación con Otras Tareas

- **TASK-05**: ✅ Implementación de Política Epsilon-Greedy (usa `get_best_action`)
  - Ver documentación completa en `docs/AGENT_IMPLEMENTATION.md`
- **TASK-06**: ✅ Implementación de la Ecuación de Actualización Q (usa `get`, `set`, `get_max_q_value`)
  - Ver documentación completa en `docs/AGENT_IMPLEMENTATION.md`
- **TASK-03**: Generación de Tabla de Transiciones (provee estados alcanzables)

---

**Autor**: Dev 2  
**Tarea**: TASK-04 - Implementación de la Tabla Q (Q-Table)  
**Fecha**: 2024

