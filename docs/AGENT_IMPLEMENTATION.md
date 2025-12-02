# Implementación del Agente Q-Learning

## Descripción

Implementación de **TASK-05** (Política Epsilon-Greedy) y **TASK-06** (Ecuación de Actualización Q) para el agente Q-Learning.

## Clase QLearningAgent

```python
class QLearningAgent:
    def __init__(self, q_table, epsilon=0.1, alpha=0.1, gamma=0.9)
    def choose_action(self, state, valid_actions) -> int
    def update(self, state, action, reward, next_state, next_valid_actions) -> None
    def set_epsilon(self, epsilon: float) -> None
    def set_alpha(self, alpha: float) -> None
    def set_gamma(self, gamma: float) -> None
```

## TASK-05: Política Epsilon-Greedy

**choose_action()** equilibra exploración y explotación:

```python
if random.random() < self.epsilon:
    return random.choice(valid_actions)  # Exploración
else:
    return self.q_table.get_best_action(state, valid_actions)  # Explotación
```

- **epsilon**: Probabilidad de exploración (0.0-1.0)
- Con probabilidad `epsilon`: acción aleatoria
- Con probabilidad `1-epsilon`: mejor acción conocida

## TASK-06: Ecuación de Actualización Q

**update()** implementa la fórmula Q-Learning:

```
Q(s,a) ← Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
```

```python
current_q = self.q_table.get(state, action)
max_next_q = self.q_table.get_max_q_value(next_state, next_valid_actions)
target = reward + self.gamma * max_next_q
new_q = current_q + self.alpha * (target - current_q)
self.q_table.set(state, action, new_q)
```

- **alpha**: Tasa de aprendizaje (0.0-1.0)
- **gamma**: Factor de descuento (0.0-1.0)

## Uso Básico

```python
from app.agent import QTable, QLearningAgent

q_table = QTable()
agent = QLearningAgent(q_table, epsilon=0.1, alpha=0.1, gamma=0.9)

# Elegir acción
action = agent.choose_action(state, valid_actions)

# Actualizar Q
agent.update(state, action, reward, next_state, next_valid_actions)
```
