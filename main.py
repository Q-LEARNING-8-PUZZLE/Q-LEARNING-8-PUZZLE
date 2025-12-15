"""
Script principal para entrenar el agente Q-Learning en el 8-Puzzle.
TASK-08: Bucle Principal de Entrenamiento
TASK-09: Ajuste de Hiperparámetros (Tuning)

Este script orquesta los episodios de entrenamiento:
- Reinicia el entorno al inicio de cada episodio
- Ejecuta pasos hasta terminar o alcanzar el límite de pasos
- Actualiza la Q-Table después de cada acción
- Implementa decaimiento de epsilon
- Detecta ciclos para evitar bucles infinitos
- Evalúa el agente después del entrenamiento
"""

from app.environment import EightPuzzle
from app.agent import QLearningAgent, QTable
from app.config import (
    ALPHA, GAMMA, EPSILON,
    NUM_EPISODES, MAX_STEPS_PER_EPISODE,
    EPSILON_DECAY, EPSILON_MIN,
    REWARD_GOAL, REWARD_STEP, REWARD_INVALID
)


def main():
    """
    Función principal que ejecuta el entrenamiento del agente Q-Learning.
    TASK-09: Ajuste de Hiperparámetros (Tuning)
    """
    print("=" * 80)
    print("ENTRENAMIENTO DEL AGENTE Q-LEARNING - 8-PUZZLE")
    print("TASK-09: Ajuste de Hiperparámetros")
    print("=" * 80)
    print()
    
    print("Configuracion de Hiperparametros:")
    print(f"  - Episodios de entrenamiento: {NUM_EPISODES}")
    print(f"  - Pasos maximos por episodio: {MAX_STEPS_PER_EPISODE}")
    print(f"  - Alpha (tasa de aprendizaje): {ALPHA}")
    print(f"  - Gamma (factor de descuento): {GAMMA}")
    print(f"  - Epsilon inicial: {EPSILON}")
    print(f"  - Epsilon decay: {EPSILON_DECAY}")
    print(f"  - Epsilon minimo: {EPSILON_MIN}")
    print(f"  - Recompensas: Goal={REWARD_GOAL}, Step={REWARD_STEP}, Invalid={REWARD_INVALID}")
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
    
    # Crear agente Q-Learning con epsilon inicial
    print("Creando agente Q-Learning...")
    current_epsilon = EPSILON
    agent = QLearningAgent(
        q_table=q_table,
        epsilon=current_epsilon,
        alpha=ALPHA,
        gamma=GAMMA
    )
    
    print("\n" + "=" * 80)
    print("FASE 1: ENTRENAMIENTO")
    print("=" * 80)
    print()
    
    # Métricas de entrenamiento
    total_steps_per_episode = []
    total_rewards_per_episode = []
    success_count = 0
    success_per_100 = []
    
    # BUCLE PRINCIPAL DE ENTRENAMIENTO
    for episode in range(1, NUM_EPISODES + 1):
        # 1. REINICIAR EL ENTORNO (puzzles más fáciles: 10 shuffles)
        state = env.reset(random_start=True, shuffles=10)
        
        episode_reward = 0.0
        steps = 0
        done = False
        visited_states = {}  # Detección de ciclos
        
        # 2. EJECUTAR PASOS HASTA TERMINAR O LÍMITE DE PASOS
        while not done and steps < MAX_STEPS_PER_EPISODE:
            # Detección de ciclos: si visitamos el mismo estado 3 veces, terminar
            if state in visited_states:
                visited_states[state] += 1
                if visited_states[state] >= 3:
                    break
            else:
                visited_states[state] = 1
            
            # Obtener acciones válidas
            valid_actions = env.get_valid_actions(state)
            
            if not valid_actions:
                break
            
            # Seleccionar acción usando política epsilon-greedy
            action = agent.choose_action(state, valid_actions)
            
            # Ejecutar acción en el entorno (SIN visualización durante entrenamiento)
            next_state, action_valid = env.step(action, verbose=False)
            
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
        
        # Registrar métricas del episodio
        total_steps_per_episode.append(steps)
        total_rewards_per_episode.append(episode_reward)
        
        if done:
            success_count += 1
        
        # 4. DECAIMIENTO DE EPSILON (después de cada episodio)
        current_epsilon = max(EPSILON_MIN, current_epsilon * EPSILON_DECAY)
        agent.set_epsilon(current_epsilon)
        
        # Mostrar progreso cada 100 episodios
        if episode % 100 == 0:
            recent_success = sum(1 for i in range(max(0, episode-100), episode) 
                               if i < len(total_steps_per_episode) and 
                               total_steps_per_episode[i] < MAX_STEPS_PER_EPISODE and
                               env.is_goal(env.GOAL_STATE))
            success_rate = (success_count / episode) * 100
            success_per_100.append(success_rate)
            avg_steps = sum(total_steps_per_episode[-100:]) / min(100, episode)
            avg_reward = sum(total_rewards_per_episode[-100:]) / min(100, episode)
            
            print(f"Episodio {episode:5d}/{NUM_EPISODES} | "
                  f"Exito: {success_rate:5.1f}% | "
                  f"Pasos: {avg_steps:6.1f} | "
                  f"Recompensa: {avg_reward:8.1f} | "
                  f"Epsilon: {current_epsilon:.4f} | "
                  f"Q-Table: {len(agent.q_table):6d}")
    
    # RESUMEN DE ENTRENAMIENTO
    print("\n" + "=" * 80)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("=" * 80)
    print(f"\nEpisodios completados: {NUM_EPISODES}")
    print(f"Episodios exitosos: {success_count}")
    print(f"Tasa de exito final: {(success_count / NUM_EPISODES * 100):.2f}%")
    print(f"\nPasos promedio: {sum(total_steps_per_episode) / NUM_EPISODES:.1f}")
    print(f"Recompensa promedio: {sum(total_rewards_per_episode) / NUM_EPISODES:.1f}")
    print(f"\nTamano final de Q-Table: {len(agent.q_table)} entradas")
    print(f"Epsilon final: {current_epsilon:.6f}")
    
    # FASE 2: EVALUACIÓN (con epsilon=0, solo explotación)
    print("\n" + "=" * 80)
    print("FASE 2: EVALUACION (Epsilon=0, Solo Explotacion)")
    print("=" * 80)
    print()
    
    # Guardar epsilon actual y establecer a 0 para evaluación
    agent.set_epsilon(0.0)
    
    eval_episodes = 5
    eval_success = 0
    
    for eval_ep in range(1, eval_episodes + 1):
        print("\n" + "=" * 80)
        print(f"EVALUACION {eval_ep}/{eval_episodes}")
        print("=" * 80)
        
        # Puzzle más fácil para evaluación
        state = env.reset(random_start=True, shuffles=10)
        
        print("\nEstado inicial:")
        env.render()
        print()
        
        steps = 0
        done = False
        eval_reward = 0.0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            valid_actions = env.get_valid_actions(state)
            
            if not valid_actions:
                break
            
            # Elegir MEJOR acción (epsilon=0)
            action = agent.choose_action(state, valid_actions)
            
            # Ejecutar con visualización
            next_state, action_valid = env.step(action, verbose=True)
            
            reward = env.get_reward(next_state, action_valid)
            eval_reward += reward
            
            done = env.is_goal(next_state)
            
            state = next_state
            steps += 1
        
        if done:
            eval_success += 1
            print("\n*** PUZZLE RESUELTO! ***")
        else:
            print(f"\nNo se resolvio en {MAX_STEPS_PER_EPISODE} pasos.")
        
        print(f"\nResumen:")
        print(f"  - Pasos: {steps}")
        print(f"  - Recompensa: {eval_reward:.1f}")
        print(f"  - Estado final: {'OBJETIVO' if done else 'NO RESUELTO'}")
    
    # RESUMEN FINAL
    print("\n" + "=" * 80)
    print("RESUMEN FINAL - TASK-09")
    print("=" * 80)
    print(f"\n[ENTRENAMIENTO]")
    print(f"  Episodios: {NUM_EPISODES}")
    print(f"  Tasa de exito: {(success_count / NUM_EPISODES * 100):.2f}%")
    print(f"  Q-Table size: {len(agent.q_table)} entradas")
    print(f"\n[EVALUACION]")
    print(f"  Episodios: {eval_episodes}")
    print(f"  Tasa de exito: {(eval_success / eval_episodes * 100):.1f}%")
    print(f"\n[HIPERPARAMETROS USADOS]")
    print(f"  Alpha: {ALPHA}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Epsilon: {EPSILON} -> {current_epsilon:.6f}")
    print(f"  Epsilon Decay: {EPSILON_DECAY}")
    
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
