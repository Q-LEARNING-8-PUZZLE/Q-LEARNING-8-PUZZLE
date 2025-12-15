"""
Script principal para entrenar el agente Q-Learning en el 8-Puzzle.
TASK-08: Bucle Principal de Entrenamiento

Este script orquesta los episodios de entrenamiento:
- Reinicia el entorno al inicio de cada episodio
e- Ejecuta pasos hasta terminar o alcanzar el límite de pasos
- Actualiza la Q-Table después de cada acción
- Muestra cada paso del entorno durante el entrenamiento
"""

from app.environment import EightPuzzle
from app.agent import QLearningAgent, QTable
from app.config import (
    ALPHA, GAMMA, EPSILON,
    MAX_STEPS_PER_EPISODE,
    REWARD_GOAL, REWARD_STEP, REWARD_INVALID
)


def main():
    """
    Función principal que ejecuta el entrenamiento del agente Q-Learning.
    Muestra cada paso del entorno durante el entrenamiento.
    """
    print("=" * 80)
    print("ENTRENAMIENTO DEL AGENTE Q-LEARNING - 8-PUZZLE")
    print("=" * 80)
    print()
    
    # Configuración del entrenamiento
    NUM_EPISODES = 5  # Número de episodios para demostración
    SHOW_STEPS = True  # Mostrar cada paso del entorno
    
    print("Configuracion:")
    print(f"  - Episodios: {NUM_EPISODES}")
    print(f"  - Pasos maximos por episodio: {MAX_STEPS_PER_EPISODE}")
    print(f"  - Alpha (tasa de aprendizaje): {ALPHA}")
    print(f"  - Gamma (factor de descuento): {GAMMA}")
    print(f"  - Epsilon (exploracion): {EPSILON}")
    print(f"  - Visualizacion de pasos: {'SI' if SHOW_STEPS else 'NO'}")
    print()
    
    # Crear entorno del 8-Puzzle
    print("Inicializando entorno...")
    env = EightPuzzle(
        use_reachable_states=False,
        reward_goal=REWARD_GOAL,
        reward_step=REWARD_STEP,
        reward_invalid=REWARD_INVALID
    )
    
    # Crear Q-Table
    print("Creando Q-Table...")
    q_table = QTable(initial_value=0.0)
    
    # Crear agente Q-Learning
    print("Creando agente Q-Learning...")
    agent = QLearningAgent(
        q_table=q_table,
        epsilon=EPSILON,
        alpha=ALPHA,
        gamma=GAMMA
    )
    
    print("\n" + "=" * 80)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 80)
    print()
    
    # Métricas de entrenamiento
    total_steps_per_episode = []
    total_rewards_per_episode = []
    success_count = 0
    
    # BUCLE PRINCIPAL DE ENTRENAMIENTO
    for episode in range(1, NUM_EPISODES + 1):
        print("\n" + "=" * 80)
        print(f"EPISODIO {episode}/{NUM_EPISODES}")
        print("=" * 80)
        
        # 1. REINICIAR EL ENTORNO
        state = env.reset(random_start=True, shuffles=20)
        
        print("\nEstado inicial:")
        env.render()
        print()
        
        episode_reward = 0.0
        steps = 0
        done = False
        
        # 2. EJECUTAR PASOS HASTA TERMINAR O LÍMITE DE PASOS
        while not done and steps < MAX_STEPS_PER_EPISODE:
            # Obtener acciones válidas
            valid_actions = env.get_valid_actions(state)
            
            if not valid_actions:
                print("No hay acciones validas disponibles.")
                break
            
            # Seleccionar acción usando política epsilon-greedy
            action = agent.choose_action(state, valid_actions)
            
            # Ejecutar acción en el entorno (con visualización)
            next_state, action_valid = env.step(action, verbose=SHOW_STEPS)
            
            # Obtener recompensa
            reward = env.get_reward(next_state, action_valid)
            episode_reward += reward
            
            # Verificar si llegamos al objetivo
            done = env.is_goal(next_state)
            
            # 3. ACTUALIZAR LA Q-TABLE
            next_valid_actions = env.get_valid_actions(next_state)
            agent.update(state, action, reward, next_state, next_valid_actions)
            
            # Actualizar estado
            state = next_state
            steps += 1
            
            # Mostrar información del paso si no está en modo verbose
            if not SHOW_STEPS:
                action_names = {0: "ARRIBA", 1: "ABAJO", 2: "IZQUIERDA", 3: "DERECHA"}
                print(f"Paso {steps}: {action_names[action]} | Recompensa: {reward:.1f}")
        
        # Registrar métricas del episodio
        total_steps_per_episode.append(steps)
        total_rewards_per_episode.append(episode_reward)
        
        if done:
            success_count += 1
            print("\n*** PUZZLE RESUELTO! ***")
        else:
            print(f"\nNo se resolvio en {MAX_STEPS_PER_EPISODE} pasos.")
        
        print(f"\nResumen del episodio:")
        print(f"  - Pasos totales: {steps}")
        print(f"  - Recompensa total: {episode_reward:.1f}")
        print(f"  - Estado final: {'OBJETIVO' if done else 'NO RESUELTO'}")
        print(f"  - Tamano Q-Table: {len(agent.q_table)} entradas")
    
    # RESUMEN FINAL
    print("\n" + "=" * 80)
    print("RESUMEN FINAL DEL ENTRENAMIENTO")
    print("=" * 80)
    print(f"\nEpisodios completados: {NUM_EPISODES}")
    print(f"Episodios exitosos: {success_count}")
    print(f"Tasa de exito: {(success_count / NUM_EPISODES * 100):.1f}%")
    print(f"\nPasos promedio: {sum(total_steps_per_episode) / NUM_EPISODES:.1f}")
    print(f"Recompensa promedio: {sum(total_rewards_per_episode) / NUM_EPISODES:.1f}")
    print(f"\nTamano final de Q-Table: {len(agent.q_table)} entradas")
    print(f"Epsilon final: {agent.epsilon:.4f}")
    
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
