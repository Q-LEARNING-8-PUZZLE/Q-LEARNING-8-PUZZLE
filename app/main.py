"""
Punto de entrada principal para ejecutar el entrenamiento y la evaluación.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.environment import EightPuzzle
from app.agent import QLearningAgent, QTable
from app.trainer import Trainer
from app.config import NUM_EPISODES
from app.plotter import generate_plots
from app.visualizer import PuzzleVisualizer
import time
import random

def run_visual_demo(env: EightPuzzle, agent: QLearningAgent, visualizer: PuzzleVisualizer):
    """
    Ejecuta una demostración visual FINAL del agente resolviendo el puzzle.
    """
    print("\n" + "="*60)
    print("  DEMOSTRACIÓN FINAL: AGENTE ENTRENADO")
    print("="*60)
    
    # 1. Desordenar (Shuffling)
    print("Desordenando el puzzle para la prueba final...")
    state = env.reset(random_start=False)
    visualizer.update(state, "Prueba Final: Desordenando...")
    time.sleep(1)
    
    shuffles = 20
    current_state = state
    for i in range(shuffles):
        valid_actions = env.get_valid_actions(current_state)
        action = random.choice(valid_actions)
        current_state, _ = env.step(action)
        visualizer.update(current_state, f"Desordenando... {i+1}/{shuffles}")
    
    print("¡Desordenado! El agente intentará resolverlo...")
    time.sleep(1)
    
    # 2. Resolver (Solving)
    state = current_state
    done = False
    steps = 0
    max_steps = 100
    
    # Aseguramos que el agente use política greedy
    original_epsilon = agent.epsilon
    agent.set_epsilon(0.0)
    
    while not done and steps < max_steps:
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            break
            
        action = agent.choose_action(state, valid_actions)
        next_state, action_valid = env.step(action)
        
        # Mapping de acciones a texto
        action_names = {0: "ARRIBA", 1: "ABAJO", 2: "IZQUIERDA", 3: "DERECHA"}
        action_name = action_names.get(action, str(action))
        
        step_msg = f"Resolviendo... Paso {steps+1}: {action_name}"
        print(f"{step_msg} -> Estado: {next_state}")
        
        visualizer.update(next_state, step_msg)
        
        state = next_state
        done = env.is_goal(state)
        steps += 1
    
    agent.set_epsilon(original_epsilon)
    
    if done:
        msg = f"¡RESUELTO en {steps} pasos!"
        print(f"\n{msg}")
        visualizer.update(state, msg)
    else:
        msg = "No se encontró solución en el límite de pasos."
        print(f"\n{msg}")
        visualizer.update(state, msg)
        
    print("\nLa ventana se cerrará en 5 segundos...")
    time.sleep(5)
    visualizer.close()

def main():
    """
    Función principal que orquesta la creación de componentes,
    el entrenamiento y la generación de gráficas.
    """
    # 0. Inicializar Visualizador PRIMERO (según requerimiento)
    print("Inicializando sistema visual...")
    try:
        visualizer = PuzzleVisualizer(title="8-Puzzle Training Monitor")
        # Mostrar imagen inicial
        visualizer.update((1,2,3,4,5,6,7,8,0), "Inicializando: Imagen Original")
        time.sleep(2)
        
        # Mostrar desorden inicial (Simulación de "romper" la imagen antes de empezar)
        print("Desordenando imagen inicial...")
        visualizer.update((1,2,3,4,5,6,7,8,0), "Desordenando imagen inicial...")
        # Simular un desorden rápido visual
        dummy_env = EightPuzzle()
        curr = dummy_env.reset(random_start=False)
        for i in range(15):
            acts = dummy_env.get_valid_actions(curr)
            act = random.choice(acts)
            curr, _ = dummy_env.step(act)
            visualizer.update(curr, f"Preparando entrenamiento... {i+1}/15")
        
        time.sleep(1)
        
    except Exception as e:
        print(f"Advertencia: No se pudo iniciar el visualizador: {e}")
        visualizer = None

    # 1. Configuración del entorno
    env = EightPuzzle(use_reachable_states=False)

    # 2. Configuración del agente
    q_table = QTable()
    agent = QLearningAgent(q_table)

    # 3. Configuración del entrenador
    # Pasamos el visualizer al trainer
    trainer = Trainer(
        environment=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        verbose=True,
        log_interval=100,  # Imprimir y actualizar visualizador cada 100 episodios
        visualizer=visualizer
    )

    # 4. Iniciar el entrenamiento (Con visualización integrada)
    training_stats = trainer.train()

    # 5. Generar y guardar las gráficas de rendimiento
    generate_plots()
    
    # 6. Demostración Visual Final
    if visualizer:
        run_visual_demo(env, agent, visualizer)

if __name__ == "__main__":
    main()
