# Backlog del Proyecto: Agente Q-Learning para 8-Puzzle

Este documento define las tareas necesarias para completar el proyecto, organizadas como "Issues" de GitHub. Las tareas están distribuidas para un equipo de 4 personas, cubriendo desde la lógica del juego hasta el análisis final.

## 👥 Estructura del Equipo (Sugerida)
*   **Dev 1 - Environment:** Encargado de la lógica del puzzle, generación de estados y validación de movimientos.
*   **Dev 2 - Agent:** Encargado de la implementación del algoritmo Q-Learning y la política de decisiones.
*   **Dev 3 - Trainer:** Encargado del bucle principal de entrenamiento y ajuste de hiperparámetros.
*   **Dev 4 - Analyst:** Encargado de la recolección de métricas, visualización de datos y redacción del informe.

---

## 📋 Listado de Tareas (Issues)

### Epic 1: Configuración del Entorno (Environment)

#### [TASK-01] Implementación de la Estructura del Tablero y Estados
*   **Asignado a:** Dev 1
*   **Nivel:** ⭐⭐ (Medio)
*   **Descripción:** Definir la estructura de datos para representar el tablero de 3x3. Implementar la función para generar el estado objetivo (1,2,3,4,5,6,7,8,0).
*   **Criterios de Aceptación:**
    *   Representación clara del estado (ej. matriz o array).
    *   Función para detectar si un estado es el objetivo.

#### [TASK-02] Lógica de Movimientos y Transiciones
*   **Asignado a:** Dev 1
*   **Nivel:** ⭐⭐ (Medio)
*   **Descripción:** Implementar las 4 acciones posibles (Arriba, Abajo, Izquierda, Derecha) para el espacio vacío. Validar límites del tablero para evitar movimientos ilegales.
*   **Criterios de Aceptación:**
    *   Función `step(action)` que devuelve el nuevo estado.
    *   Manejo correcto de bordes (no salir de la cuadrícula).

#### [TASK-03] Generación de Tabla de Transiciones y Alcanzabilidad (Opcional/Avanzado)
*   **Asignado a:** Dev 1 (Apoyo de Dev 2)
*   **Nivel:** ⭐⭐⭐ (Alto)
*   **Descripción:** Siguiendo la "Ayuda" del documento, generar todos los estados posibles (permutaciones) y validar cuáles son alcanzables desde la solución usando BFS.
*   **Criterios de Aceptación:**
    *   Mapa/Grafo de estados válidos vs inválidos.
    *   Optimización para evitar explorar estados inalcanzables.

---

### Epic 2: Implementación del Agente (Q-Learning)

#### [TASK-04] Implementación de la Tabla Q (Q-Table)
*   **Asignado a:** Dev 2
*   **Nivel:** ⭐ (Bajo)
*   **Descripción:** Crear la estructura de datos para la Tabla Q que mapee `(Estado, Acción) -> Valor`. Debe manejar la gran cantidad de estados posibles (362,880).
*   **Criterios de Aceptación:**
    *   Estructura eficiente (ej. Diccionario/Hash Map con el estado como key).
    *   Inicialización correcta de valores.

#### [TASK-05] Implementación de Política Epsilon-Greedy
*   **Asignado a:** Dev 2
*   **Nivel:** ⭐⭐ (Medio)
*   **Descripción:** Implementar la lógica de selección de acciones. Debe elegir una acción aleatoria con probabilidad `epsilon` (exploración) y la mejor acción conocida con probabilidad `1 - epsilon` (explotación).
*   **Criterios de Aceptación:**
    *   Función `choose_action(state)` funcional.
    *   `epsilon` debe ser parametrizable.

#### [TASK-06] Implementación de la Ecuación de Actualización Q
*   **Asignado a:** Dev 2
*   **Nivel:** ⭐⭐⭐ (Alto)
*   **Descripción:** Implementar la fórmula de actualización:
    `Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]`
*   **Criterios de Aceptación:**
    *   Función `update(state, action, reward, next_state)` correcta.
    *   Uso correcto de `alpha` (tasa de aprendizaje) y `gamma` (factor de descuento).

#### [TASK-07] Definición del Sistema de Recompensas
*   **Asignado a:** Dev 2 & Dev 3
*   **Nivel:** ⭐ (Bajo)
*   **Descripción:** Definir los valores de recompensa.
    *   Recompensa negativa pequeña por cada paso (para incentivar rapidez).
    *   Gran recompensa positiva al llegar al estado objetivo.
    *   Penalización fuerte por movimientos inválidos (si aplica).
*   **Criterios de Aceptación:**
    *   Función de recompensa configurada y probada.

---

### Epic 3: Entrenamiento y Optimización (Training)

#### [TASK-08] Bucle Principal de Entrenamiento
*   **Asignado a:** Dev 3
*   **Nivel:** ⭐⭐ (Medio)
*   **Descripción:** Crear el script principal que orqueste los episodios de entrenamiento. Reiniciar el entorno, ejecutar pasos hasta terminar o límite de pasos, y actualizar la Q-Table.
*   **Criterios de Aceptación:**
    *   Script ejecutable que corre N episodios.
    *   El agente aprende (los pasos para resolver disminuyen).

#### [TASK-09] Ajuste de Hiperparámetros (Tuning)
*   **Asignado a:** Dev 3
*   **Nivel:** ⭐⭐⭐ (Alto)
*   **Descripción:** Experimentar con diferentes valores de `alpha`, `gamma` y estrategias de decaimiento de `epsilon`.
*   **Criterios de Aceptación:**
    *   Identificación de la mejor combinación de parámetros.
    *   Documentación de los experimentos realizados.

---

### Epic 4: Evaluación y Reporte (Analytics)

#### [TASK-10] Sistema de Logging y Métricas
*   **Asignado a:** Dev 4
*   **Nivel:** ⭐ (Bajo)
*   **Descripción:** Implementar el registro de datos por episodio: número de pasos para resolver, recompensa total acumulada, éxito/fallo.
*   **Criterios de Aceptación:**
    *   Generación de logs (CSV o listas) durante el entrenamiento.

#### [TASK-11] Generación de Gráficas de Rendimiento
*   **Asignado a:** Dev 4
*   **Nivel:** ⭐⭐ (Medio)
*   **Descripción:** Crear scripts para visualizar la evolución del aprendizaje.
    *   Gráfica: Pasos promedio vs Episodios.
    *   Gráfica: Tasa de éxito vs Tiempo.
*   **Criterios de Aceptación:**
    *   Gráficas claras generadas con Matplotlib o similar.

#### [TASK-12] Redacción del Informe y Documentación Final
*   **Asignado a:** Todo el Equipo (Liderado por Dev 4)
*   **Nivel:** ⭐⭐ (Medio)
*   **Descripción:** Compilar el código, las gráficas y el análisis en el informe final. Explicar decisiones de diseño y desafíos (como el espacio de estados).
*   **Criterios de Aceptación:**
    *   Informe completo según los requisitos de entrega del PDF.
    *   Código comentado y limpio.
