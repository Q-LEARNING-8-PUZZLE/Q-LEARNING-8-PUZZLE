# Q-LEARNING-8-PUZZLE

Este es un repositorio para resolver el **Problema del 8-Puzzle** utilizando **Aprendizaje por Refuerzo con Q-Learning**.

## Cómo Ejecutar el Proyecto

Sigue estos pasos para poner en marcha el entrenamiento y la evaluación del agente.

### 1. Prerrequisitos

- **Python 3.10+**
- **`uv`**: Una herramienta rápida para la gestión de entornos y paquetes de Python.

Si no tienes `uv`, instálalo con uno de los siguientes comandos:

- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Linux, macOS o WSL:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Instalación

Clona el repositorio y usa `uv` para instalar las dependencias.

```bash
# Clona este repositorio
git clone <URL_DEL_REPOSITORIO>
cd Q-LEARNING-8-PUZZLE

# Crea un entorno virtual e instala las dependencias del proyecto
uv venv
uv sync
```

### 3. Ejecución del Entrenamiento

Para entrenar al agente, ejecuta el script principal `main.py`. Este script se encargará de todo el proceso: entrenamiento, evaluación e informe de resultados en la consola.

```bash
# Activa el entorno virtual
# En Linux/macOS/WSL:
source .venv/bin/activate
# En Windows (PowerShell):
# .venv\Scripts\activate

# Ejecuta el script principal
python main.py
```

El script mostrará el progreso del entrenamiento, incluyendo la tasa de éxito, los pasos promedio y la recompensa. Al final, ejecutará una fase de evaluación donde podrás ver al agente resolver el puzzle paso a paso.

### 4. Ver los Resultados

Los resultados se pueden analizar de dos maneras:

1.  **Consola**: Al finalizar la ejecución, se imprime un resumen completo del rendimiento del agente, incluyendo la tasa de éxito final y los hiperparámetros utilizados.
2.  **Gráficos (si se generan)**: Si el proyecto está configurado para guardar gráficos, estos aparecerán en el directorio `data/plots/`.
    - `steps_vs_episodes.png`: Muestra la evolución de los pasos necesarios por episodio.
    - `success_rate_vs_episodes.png`: Ilustra cómo mejora la tasa de éxito.

---

## Descripción de la Tarea

### Objetivos de Aprendizaje

- Comprender y aplicar los conceptos de aprendizaje por refuerzo y el algoritmo Q-learning.
- Implementar un agente que resuelva el 8-puzzle.
- Evaluar el rendimiento del agente y ajustar hiperparámetros.

### El Problema del 8-Puzzle

El 8-puzzle es una cuadrícula de 3x3 con 8 piezas numeradas (1-8) y un espacio vacío. El objetivo es ordenar las piezas hasta alcanzar una configuración final predefinida.

```
Estado Objetivo:
1 2 3
4 5 6
7 8 .
```

### Implementación del Algoritmo Q-Learning

- **Tabla Q**: Almacena los valores de las acciones para cada estado.
- **Política ε-greedy**: Equilibra la exploración (probar acciones nuevas) y la explotación (usar el conocimiento actual).
- **Ecuación de Bellman (actualización Q)**:
  ```
  Q(s,a) = Q(s,a) + α [r + γ * max Q(s',a') - Q(s,a)]
  ```
  - `α` (alpha): Tasa de aprendizaje.
  - `γ` (gamma): Factor de descuento.
