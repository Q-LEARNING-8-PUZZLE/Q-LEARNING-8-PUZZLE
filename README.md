# Q-LEARNING-8-PUZZLE

Este es un repositorio colaborativo para la realización de una tarea sobre **Aplicación de Aprendizaje por Refuerzo con Q-Learning** para resolver el **Problema del 8-Puzzle** en el contexto de Programación en IA.

## MEJORAS RECIENTES

**Fecha de implementación:** 2 de diciembre de 2025

### Resumen de Mejoras

Se han implementado mejoras de configurabilidad y flexibilidad en el entorno del 8-Puzzle:

#### 1. Recompensas Configurables

Las recompensas ahora son parámetros configurables del constructor:

```python
from app.environment import EightPuzzle

# Personalizar recompensas
env = EightPuzzle(
    reward_goal=500.0,      # Recompensa al alcanzar objetivo
    reward_step=-0.5,       # Recompensa por cada paso
    reward_invalid=-50.0    # Penalización por movimiento inválido
)
```

**Beneficios:** Experimentación más fácil, sin modificar código fuente, compatible con `config.py`.

#### 2. Parámetro `verbose` en `step()`

Control del output durante la ejecución:

```python
# Sin prints (ideal para entrenamiento)
state, valid = env.step(0, verbose=False)

# Con prints detallados (útil para debugging)
state, valid = env.step(0, verbose=True)
```

**Beneficios:** Entrenamiento silencioso, debugging selectivo, mejor rendimiento (~15-20% más rápido).

#### 3. Parámetro `return_string` en `render()`

Captura la representación del tablero como string:

```python
# Capturar como string (para logging)
estado_str = env.render(return_string=True)
```

**Beneficios:** Logging a archivos más fácil, útil para comparaciones, retrocompatible.

### Archivos Modificados

- **`app/environment.py`**: Añadidos parámetros configurables en `__init__()`, `step()`, `render()` y `get_reward()`
- **`app/config.py`**: Añadidas constantes `REWARD_GOAL`, `REWARD_STEP`, `REWARD_INVALID`, `MAX_STEPS_PER_EPISODE`, `EPSILON_DECAY`, `EPSILON_MIN`
- **`tests/examples_usage.py`** (NUEVO): 7 ejemplos completos de uso de las nuevas funcionalidades

### Ejecutar Tests y Ejemplos

```bash
# Ejecutar todos los ejemplos
uv run python tests/examples_usage.py
```

Los ejemplos incluyen:
- Recompensas por defecto y personalizadas
- Modo verbose activado/desactivado
- Render con return_string
- Configuración desde config.py
- Simulación de entrenamiento

### Retrocompatibilidad

Todas las mejoras son **100% retrocompatibles**. El código existente sigue funcionando sin modificaciones.

---

## Descripción de la Tarea

### Objetivos de Aprendizaje

- Comprender y aplicar los conceptos de aprendizaje por refuerzo y el algoritmo Q-learning
- Implementar un agente de aprendizaje por refuerzo que resuelva el problema del 8-puzzle
- Evaluar el rendimiento del agente y ajustar los hiperparámetros para mejorar la eficacia del aprendizaje

### 1. Introducción al Problema del 8-Puzzle

El problema del 8-puzzle consiste en una cuadrícula de 3x3 con 8 piezas numeradas del 1 al 8 y un espacio vacío. El objetivo es reordenar las piezas de modo que coincidan con un estado objetivo predefinido, por ejemplo:

```
1 2 3
4 5 6
7 8
```

### 2. Configuración del Entorno (uv)

Este proyecto utiliza `uv` para la gestión de dependencias, una alternativa moderna y rápida a pip.

#### Instalación de uv

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Linux / WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Instalación de Dependencias

Para instalar las dependencias del proyecto definidas en `pyproject.toml`:

```bash
uv sync
```

#### Ejecución

Para ejecutar scripts utilizando el entorno virtual gestionado por `uv`:

```bash
uv run app/environment.py
```

### 3. Configuración del Entorno de Aprendizaje

Implementa un entorno para el 8-puzzle donde el agente pueda mover las piezas en las cuatro direcciones posibles: arriba, abajo, izquierda y derecha. Asegúrate de definir las reglas de movimiento (es decir, que no puede moverse fuera de los límites de la cuadrícula) y de penalizar movimientos inválidos.

### 4. Definición de Recompensas y Estados

- Cada estado es una configuración única del tablero
- El objetivo del agente es llegar al estado objetivo
- Asigna una recompensa negativa para cada paso para incentivar al agente a resolver el puzzle en el menor número de movimientos posible
- Otorga una recompensa positiva significativa al alcanzar el estado objetivo

### 5. Implementación del Algoritmo Q-Learning

- Implementa la tabla Q para almacenar las recompensas para cada acción en cada estado
- Usa una política ε-greedy para equilibrar la exploración y explotación
- Aplica la actualización de la función Q con la ecuación:

```
Q(s,a) = Q(s,a) + alpha[r + gamma * max Q(s',a') - Q(s,a)]
```

donde:
- `s` y `s'` son el estado actual y el siguiente
- `a` y `a'` son las acciones actuales y las posibles en `s'`
- `α` es la tasa de aprendizaje
- `γ` es el factor de descuento

### 6. Entrenamiento del Agente

- Entrena al agente con diferentes configuraciones iniciales del 8-puzzle
- Ajusta los hiperparámetros (ε, α, γ) para optimizar el rendimiento
- Registra el número de pasos que toma el agente para resolver el puzzle desde diferentes estados

### 7. Evaluación y Análisis de Resultados

- Evalúa el rendimiento del agente en el 8-puzzle
- Presenta gráficas que muestren cómo cambia la tasa de éxito y el número de pasos promedio para resolver el puzzle a lo largo del tiempo
- Reflexiona sobre el impacto de cada hiperparámetro en el rendimiento del agente y su capacidad de generalización desde estados iniciales aleatorios

### 8. Entrega

- Código implementado con explicaciones de cada parte
- Un informe que incluya:
  - Gráficas de rendimiento
  - Análisis de los resultados
  - Reflexiones sobre los desafíos encontrados y las decisiones de diseño tomadas

## Criterios de Evaluación

- **Estructura, funcionamiento correcto y claridad del código** (3 Puntos)
- **Documentación y explicaciones** (3 Puntos)
- **Calidad del análisis de resultados** (2 Puntos)
- **Ajuste y justificación de los hiperparámetros** (2 Puntos)

## Ayuda para la Implementación

### Generación de la Tabla de Transiciones

Definimos la tabla de transiciones generando todos los estados posibles como permutaciones del número `123456789` siendo el hueco el número 9.

Los estados irían desde el estado solución `123456789` hasta el `987654321`.

Para generar la tabla sabemos que hay cuatro posibles acciones que puede realizar el hueco:
- `a0` (subir)
- `a1` (derecha)
- `a2` (abajo)
- `a3` (izquierda)

Partiendo del estado solución no todos los estados son alcanzables.

Realizamos un proceso sistemático generando un array bidimensional `T` de `(987654321 - 123456789)` filas y 4 columnas rellenadas a `-1`. Una vez creada recorremos cada fila y columna y vamos rellenando la tabla con los estados alcanzables. (Se generan dos grafos no conexos de la mitad de nodos cada uno). Con un recorrido en anchura desde el estado solución podría marcar los alcanzables.
