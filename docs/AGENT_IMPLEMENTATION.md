# Implementación del Agente Q-Learning

## 📋 Descripción General

Este documento describe la implementación de las tareas **TASK-05** (Política Epsilon-Greedy) y **TASK-06** (Ecuación de Actualización Q) del agente Q-Learning para el problema del 8-Puzzle.

## 🎯 Objetivos

### TASK-05: Política Epsilon-Greedy
Implementar la lógica de selección de acciones que equilibre la exploración y explotación:
- Elegir una acción aleatoria con probabilidad `epsilon` (exploración)
- Elegir la mejor acción conocida con probabilidad `1 - epsilon` (explotación)
- `epsilon` debe ser parametrizable

### TASK-06: Ecuación de Actualización Q
Implementar la fórmula de actualización Q-Learning:
```
Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
```
- Función `update()` correcta
- Uso correcto de `alpha` (tasa de aprendizaje) y `gamma` (factor de descuento)

## 🏗️ Arquitectura de la Implementación

### Clase QLearningAgent

La clase `QLearningAgent` encapsula toda la lógica del agente Q-Learning, incluyendo:
- Política de selección de acciones (epsilon-greedy)
- Actualización de valores Q
- Gestión de hiperparámetros (epsilon, alpha, gamma)

```python
class QLearningAgent:
    def __init__(self, q_table, epsilon=0.1, alpha=0.1, gamma=0.9)
    def choose_action(self, state, valid_actions) -> int
    def update(self, state, action, reward, next_state, next_valid_actions) -> None
    def set_epsilon(self, epsilon: float) -> None
    def set_alpha(self, alpha: float) -> None
    def set_gamma(self, gamma: float) -> None
```

## 📝 Componentes Principales

### 1. Inicialización (`__init__`)

```python
def __init__(
    self,
    q_table: QTable,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.9
):
    self.q_table = q_table
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
```

**Parámetros:**
- `q_table`: Instancia de `QTable` para almacenar valores Q
- `epsilon`: Probabilidad de exploración (0.0-1.0)
- `alpha`: Tasa de aprendizaje (0.0-1.0)
- `gamma`: Factor de descuento (0.0-1.0)

**Características:**
- ✅ Todos los hiperparámetros son configurables
- ✅ Valores por defecto razonables para comenzar el entrenamiento
- ✅ Integración con la tabla Q existente

### 2. TASK-05: Selección de Acción (`choose_action`)

#### Implementación

```python
def choose_action(
    self,
    state: Tuple[int, ...],
    valid_actions: List[int]
) -> int:
    if not valid_actions:
        raise ValueError("No hay acciones válidas disponibles")
    
    # Exploración: elegir acción aleatoria con probabilidad epsilon
    if random.random() < self.epsilon:
        return random.choice(valid_actions)
    
    # Explotación: elegir la mejor acción conocida
    best_action = self.q_table.get_best_action(state, valid_actions)
    best_value = self.q_table.get(state, best_action)
    
    # Manejar empates eligiendo aleatoriamente entre las mejores
    best_actions = [
        action for action in valid_actions
        if self.q_table.get(state, action) == best_value
    ]
    
    return random.choice(best_actions)
```

#### Algoritmo Paso a Paso

1. **Validación de Entrada**
   - Verificar que hay acciones válidas disponibles
   - Lanzar excepción si no hay acciones válidas

2. **Decisión de Exploración/Explotación**
   - Generar número aleatorio entre 0 y 1
   - Si `random() < epsilon` → **Exploración**
   - Si `random() >= epsilon` → **Explotación**

3. **Exploración (Acción Aleatoria)**
   - Elegir aleatoriamente una acción de `valid_actions`
   - Permite descubrir nuevas estrategias
   - Importante al inicio del entrenamiento

4. **Explotación (Mejor Acción Conocida)**
   - Obtener la mejor acción usando `q_table.get_best_action()`
   - Obtener el valor Q de la mejor acción
   - Identificar todas las acciones con el mismo valor Q (empates)
   - Elegir aleatoriamente entre las acciones empatadas

#### Ejemplo Visual

```
Estado: (1, 2, 3, 4, 5, 6, 7, 0, 8)
Acciones válidas: [0, 1, 2, 3]
Epsilon: 0.1

Valores Q actuales:
  Q(state, 0) = 0.5
  Q(state, 1) = 0.3
  Q(state, 2) = 0.7  ← Mejor
  Q(state, 3) = 0.2

Decisión:
  random() = 0.15
  
  Como 0.15 >= 0.1 (epsilon):
    → Explotación
    → Elegir acción 2 (mejor valor Q = 0.7)
```

#### Manejo de Empates

Cuando múltiples acciones tienen el mismo valor Q máximo, el algoritmo elige aleatoriamente entre ellas. Esto evita sesgos y mejora la exploración incluso durante la explotación.

**Ejemplo de Empate:**
```python
Q(state, 0) = 0.5
Q(state, 1) = 0.5  ← Empate
Q(state, 2) = 0.3
Q(state, 3) = 0.5  ← Empate

# Elige aleatoriamente entre [0, 1, 3]
```

### 3. TASK-06: Actualización Q (`update`)

#### Implementación

```python
def update(
    self,
    state: Tuple[int, ...],
    action: int,
    reward: float,
    next_state: Tuple[int, ...],
    next_valid_actions: List[int]
) -> None:
    # Obtener el valor Q actual
    current_q_value = self.q_table.get(state, action)
    
    # Calcular el máximo valor Q para el siguiente estado
    max_next_q_value = self.q_table.get_max_q_value(
        next_state, 
        next_valid_actions
    )
    
    # Calcular el valor objetivo (target value)
    target_value = reward + self.gamma * max_next_q_value
    
    # Aplicar la ecuación de actualización Q-Learning
    new_q_value = current_q_value + self.alpha * (
        target_value - current_q_value
    )
    
    # Actualizar la tabla Q
    self.q_table.set(state, action, new_q_value)
```

#### Ecuación de Actualización Q-Learning

La fórmula implementada es:

```
Q(s,a) ← Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
```

Donde:
- `s`: Estado actual
- `a`: Acción tomada
- `r`: Recompensa recibida
- `s'`: Estado siguiente
- `α` (alpha): Tasa de aprendizaje
- `γ` (gamma): Factor de descuento
- `max Q(s',a')`: Máximo valor Q del siguiente estado

#### Algoritmo Paso a Paso

1. **Obtener Valor Q Actual**
   ```python
   current_q_value = self.q_table.get(state, action)
   ```
   - Recupera el valor Q almacenado para el par (estado, acción)
   - Si no existe, retorna el valor inicial (0.0 por defecto)

2. **Calcular Máximo Q del Siguiente Estado**
   ```python
   max_next_q_value = self.q_table.get_max_q_value(
       next_state, 
       next_valid_actions
   )
   ```
   - Encuentra el máximo valor Q entre todas las acciones válidas del siguiente estado
   - Representa la mejor recompensa futura esperada

3. **Calcular Valor Objetivo (Target)**
   ```python
   target_value = reward + self.gamma * max_next_q_value
   ```
   - Combina la recompensa inmediata con la recompensa futura descontada
   - `gamma` controla qué tan importante es el futuro (0 = solo presente, 1 = futuro igual de importante)

4. **Aplicar Actualización**
   ```python
   new_q_value = current_q_value + self.alpha * (
       target_value - current_q_value
   )
   ```
   - Calcula la diferencia entre el valor objetivo y el valor actual
   - `alpha` controla qué tan rápido se actualiza el valor Q
   - Actualización incremental (no reemplaza completamente el valor anterior)

5. **Guardar Nuevo Valor**
   ```python
   self.q_table.set(state, action, new_q_value)
   ```
   - Almacena el nuevo valor Q en la tabla

#### Ejemplo Numérico

```python
# Estado inicial
state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
action = 0  # ARRIBA
reward = -1.0
next_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
next_valid_actions = [0, 1, 2, 3]

# Hiperparámetros
alpha = 0.1
gamma = 0.9

# Valores Q iniciales (todos 0.0)
current_q = 0.0
max_next_q = 0.0  # Todas las acciones tienen Q = 0.0

# Cálculo
target_value = -1.0 + 0.9 * 0.0 = -1.0
new_q = 0.0 + 0.1 * (-1.0 - 0.0) = -0.1

# Resultado: Q(state, action) = -0.1
```

#### Interpretación de Hiperparámetros

**Alpha (α) - Tasa de Aprendizaje:**
- `alpha = 0.0`: No aprende (mantiene valores iniciales)
- `alpha = 0.1`: Aprendizaje conservador (cambios graduales)
- `alpha = 1.0`: Aprendizaje agresivo (reemplaza completamente el valor anterior)
- **Recomendado**: 0.1 - 0.3 para problemas estocásticos

**Gamma (γ) - Factor de Descuento:**
- `gamma = 0.0`: Solo considera recompensas inmediatas
- `gamma = 0.9`: Considera recompensas futuras con descuento moderado
- `gamma = 1.0`: Recompensas futuras igual de importantes que las inmediatas
- **Recomendado**: 0.9 - 0.99 para problemas secuenciales

### 4. Métodos de Configuración de Hiperparámetros

#### `set_epsilon(epsilon: float)`

```python
def set_epsilon(self, epsilon: float) -> None:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon debe estar entre 0.0 y 1.0")
    self.epsilon = epsilon
```

**Uso:**
- Útil para implementar decaimiento de epsilon durante el entrenamiento
- Permite comenzar con alta exploración y reducir gradualmente

**Ejemplo de Decaimiento:**
```python
# Inicio: alta exploración
agent.set_epsilon(1.0)

# Durante entrenamiento: reducir gradualmente
for episode in range(num_episodes):
    epsilon = max(0.01, 1.0 - episode / num_episodes)
    agent.set_epsilon(epsilon)
```

#### `set_alpha(alpha: float)`

```python
def set_alpha(self, alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe estar entre 0.0 y 1.0")
    self.alpha = alpha
```

**Uso:**
- Permite ajustar la tasa de aprendizaje dinámicamente
- Útil para técnicas como annealing de tasa de aprendizaje

#### `set_gamma(gamma: float)`

```python
def set_gamma(self, gamma: float) -> None:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma debe estar entre 0.0 y 1.0")
    self.gamma = gamma
```

**Uso:**
- Generalmente se mantiene constante durante el entrenamiento
- Puede ajustarse según el horizonte del problema

## 🔄 Flujo Completo de Uso

### Ejemplo de Uso Básico

```python
from app.agent import QTable, QLearningAgent
from app.environment import EightPuzzle

# 1. Inicializar componentes
q_table = QTable(initial_value=0.0)
agent = QLearningAgent(
    q_table=q_table,
    epsilon=0.1,
    alpha=0.1,
    gamma=0.9
)
env = EightPuzzle()

# 2. Resetear entorno
state = env.reset(random_start=True)
done = False
steps = 0

# 3. Bucle de entrenamiento (un episodio)
while not done and steps < 1000:
    # Obtener acciones válidas
    valid_actions = env.get_valid_actions(state)
    
    # Elegir acción usando epsilon-greedy
    action = agent.choose_action(state, valid_actions)
    
    # Ejecutar acción
    next_state, action_valid = env.step(action)
    reward = env.get_reward(next_state, action_valid)
    
    # Obtener acciones válidas del siguiente estado
    next_valid_actions = env.get_valid_actions(next_state)
    
    # Actualizar Q
    agent.update(
        state, 
        action, 
        reward, 
        next_state, 
        next_valid_actions
    )
    
    # Verificar si terminó
    done = env.is_goal(next_state)
    
    # Actualizar estado
    state = next_state
    steps += 1

print(f"Episodio completado en {steps} pasos")
```

### Flujo Detallado Paso a Paso

```
┌─────────────────────────────────────────────────────────┐
│ 1. INICIALIZACIÓN                                       │
│    - Crear QTable                                       │
│    - Crear QLearningAgent con hiperparámetros         │
│    - Crear Environment                                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. RESET DEL ENTORNO                                    │
│    - Generar estado inicial aleatorio                   │
│    - done = False                                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. BUCLE DE ENTRENAMIENTO (por cada paso)               │
│                                                         │
│    a) Obtener acciones válidas                         │
│       valid_actions = env.get_valid_actions(state)     │
│                                                         │
│    b) Elegir acción (TASK-05)                          │
│       action = agent.choose_action(state, valid_actions)│
│       ├─ Con prob. epsilon: acción aleatoria           │
│       └─ Con prob. (1-epsilon): mejor acción conocida  │
│                                                         │
│    c) Ejecutar acción                                   │
│       next_state, valid = env.step(action)              │
│       reward = env.get_reward(next_state, valid)       │
│                                                         │
│    d) Actualizar Q (TASK-06)                           │
│       agent.update(state, action, reward,              │
│                    next_state, next_valid_actions)     │
│       ├─ Obtener Q(s,a) actual                         │
│       ├─ Calcular max Q(s',a')                         │
│       ├─ Calcular target = r + γ*max Q(s',a')          │
│       ├─ Actualizar: Q(s,a) += α*(target - Q(s,a))    │
│       └─ Guardar nuevo Q(s,a)                          │
│                                                         │
│    e) Verificar terminación                             │
│       done = env.is_goal(next_state)                    │
│                                                         │
│    f) Actualizar estado                                │
│       state = next_state                                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. FIN DEL EPISODIO                                     │
│    - Guardar métricas (pasos, recompensa total)        │
│    - Repetir desde paso 2 para nuevo episodio          │
└─────────────────────────────────────────────────────────┘
```

## 🧪 Pruebas y Validación

### Script de Prueba Incluido

El archivo `app/agent.py` incluye un bloque de prueba que valida ambas funcionalidades:

```python
if __name__ == "__main__":
    # Prueba TASK-05: choose_action
    # Prueba TASK-06: update
    # Verificación manual de la fórmula
```

### Ejecutar Pruebas

```bash
python app/agent.py
```

### Resultados Esperados

```
============================================================
Pruebas de TASK-05 y TASK-06
============================================================

--- TASK-05: Prueba de Política Epsilon-Greedy ---
Estado inicial: (7, 1, 6, 2, 0, 3, 5, 8, 4)
Acciones válidas: [0, 1, 2, 3]
Epsilon: 0.1
✓ choose_action funciona correctamente

--- TASK-06: Prueba de Actualización Q ---
Estado actual: (1, 2, 3, 4, 5, 6, 7, 0, 8)
Acción tomada: 0
Recompensa: -1.0
Estado siguiente: (1, 2, 3, 4, 0, 6, 7, 5, 8)
✓ La actualización Q funciona correctamente
✓ La fórmula de actualización es correcta

============================================================
Pruebas completadas
============================================================
```

## ⚡ Optimizaciones y Consideraciones

### 1. Manejo de Empates en choose_action

- Cuando múltiples acciones tienen el mismo valor Q máximo, se elige aleatoriamente entre ellas
- Evita sesgos hacia acciones específicas
- Mejora la exploración incluso durante la explotación

### 2. Validación de Parámetros

- Todos los métodos de configuración validan que los valores estén en el rango [0.0, 1.0]
- Previene errores por valores inválidos
- Mensajes de error claros

### 3. Integración con QTable

- Usa los métodos existentes de `QTable` (`get`, `set`, `get_best_action`, `get_max_q_value`)
- No duplica lógica
- Mantiene la separación de responsabilidades

### 4. Eficiencia

- `choose_action`: O(n) donde n es el número de acciones válidas (máximo 4)
- `update`: O(n) donde n es el número de acciones válidas del siguiente estado
- Operaciones muy rápidas para el problema del 8-Puzzle

## 📊 Comparación con Alternativas

| Aspecto | Implementación Actual | Alternativa Manual |
|---------|---------------------|-------------------|
| **Código** | Encapsulado en clase | Disperso en múltiples lugares |
| **Reutilización** | ✅ Fácil de reutilizar | ❌ Difícil de mantener |
| **Testing** | ✅ Fácil de probar | ❌ Difícil de aislar |
| **Configuración** | ✅ Métodos dedicados | ❌ Variables globales |
| **Mantenibilidad** | ✅ Código organizado | ❌ Código acoplado |

## ✅ Criterios de Aceptación Cumplidos

### TASK-05: Política Epsilon-Greedy

✅ **Función `choose_action(state)` funcional**
- Implementada en `QLearningAgent.choose_action()`
- Maneja correctamente exploración y explotación
- Maneja empates correctamente

✅ **`epsilon` debe ser parametrizable**
- Configurable en el constructor
- Método `set_epsilon()` para cambios dinámicos
- Validación de rango [0.0, 1.0]

### TASK-06: Ecuación de Actualización Q

✅ **Función `update(state, action, reward, next_state)` correcta**
- Implementada en `QLearningAgent.update()`
- Implementa la fórmula completa de Q-Learning
- Maneja correctamente estados terminales

✅ **Uso correcto de `alpha` y `gamma`**
- `alpha` controla la tasa de actualización
- `gamma` controla el descuento de recompensas futuras
- Ambos son configurables y validados

## 🔗 Relación con Otras Tareas

- **TASK-04**: Usa `QTable` para almacenar y recuperar valores Q
- **TASK-07**: Usa el sistema de recompensas del entorno
- **TASK-08**: Será usado en el bucle principal de entrenamiento
- **TASK-09**: Los hiperparámetros pueden ajustarse dinámicamente

## 📚 Referencias

- **Q-Learning Algorithm**: Algoritmo de aprendizaje por refuerzo sin modelo
- **Epsilon-Greedy Policy**: Estrategia de balance exploración/explotación
- **Temporal Difference Learning**: Método de actualización incremental
- **8-Puzzle Problem**: Problema de búsqueda clásico

## 🎓 Conceptos Clave

### Exploración vs Explotación

- **Exploración**: Probar acciones nuevas para descubrir mejores estrategias
- **Explotación**: Usar el conocimiento actual para maximizar recompensas
- **Balance**: Epsilon-greedy equilibra ambos mediante probabilidad

### Aprendizaje Incremental

- Los valores Q se actualizan gradualmente, no se reemplazan completamente
- `alpha` controla qué tan rápido se incorpora nueva información
- Permite adaptación continua durante el entrenamiento

### Descuento Temporal

- `gamma` determina qué tan importante es el futuro
- Valores altos de gamma → planificación a largo plazo
- Valores bajos de gamma → enfoque en recompensas inmediatas

---

**Autor**: Dev 2  
**Tareas**: TASK-05 - Política Epsilon-Greedy | TASK-06 - Ecuación de Actualización Q  
**Fecha**: 2024

