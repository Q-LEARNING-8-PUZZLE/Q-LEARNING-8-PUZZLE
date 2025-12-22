"""
Trainer - Dev 3
TASK-08: Bucle Principal de Entrenamiento
TASK-09: Ajuste de Hiperparámetros (Tuning)
TASK-10: Registro de datos por episodio
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Tuple, List, Dict, Optional
import time
import csv
import itertools
from app.environment import EightPuzzle
from app.agent import QLearningAgent, QTable
from app.config import (
    ALPHA, GAMMA, EPSILON, NUM_EPISODES,
    MAX_STEPS_PER_EPISODE, EPSILON_DECAY, EPSILON_MIN,
    REWARD_GOAL, REWARD_STEP, REWARD_INVALID
)
from app.analytics import Analytics


class Trainer:
    """
    Clase que orquesta el entrenamiento del agente Q-Learning.
    
    TASK-08: Implementa el bucle principal de entrenamiento que:
    - Reinicia el entorno al inicio de cada episodio
    - Ejecuta pasos hasta terminar o alcanzar el límite de pasos
    - Actualiza la Q-Table después de cada acción
    - Registra métricas de rendimiento
    
    TASK-09: Permite ajustar hiperparámetros y experimentar con diferentes configuraciones.
    """
    
    def __init__(
        self,
        environment: EightPuzzle,
        agent: QLearningAgent,
        num_episodes: int = NUM_EPISODES,
        max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
        epsilon_decay: float = EPSILON_DECAY,
        epsilon_min: float = EPSILON_MIN,
        verbose: bool = True,
        log_interval: int = 100,
        show_steps: bool = False,
        visualizer = None
    ):
        """
        Inicializa el entrenador.
        
        Args:
            environment (EightPuzzle): Instancia del entorno del 8-Puzzle.
            agent (QLearningAgent): Instancia del agente Q-Learning.
            num_episodes (int): Número total de episodios de entrenamiento.
            max_steps_per_episode (int): Límite de pasos por episodio.
            epsilon_decay (float): Factor de decaimiento de epsilon.
            epsilon_min (float): Valor mínimo de epsilon.
            verbose (bool): Si True, imprime información durante el entrenamiento.
            log_interval (int): Intervalo de episodios para mostrar estadísticas.
            show_steps (bool): Si True, muestra cada paso individual del entorno.
            visualizer: Objeto PuzzleVisualizer opcional para actualizaciones visuales.
        """
        self.env = environment
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.verbose = verbose
        self.log_interval = log_interval
        self.show_steps = show_steps
        self.visualizer = visualizer
        
        # Métricas de entrenamiento
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.episode_success: List[bool] = []
        self.training_time: float = 0.0

        # Analytics para logging de episodios
        self.analytics = Analytics()
        
    def train(self) -> Dict:
        """
        TASK-08: Bucle principal de entrenamiento.
        """
        if self.verbose:
            print("=" * 80)
            print("INICIANDO ENTRENAMIENTO DEL AGENTE Q-LEARNING")
            print("=" * 80)
            print(f"Episodios: {self.num_episodes}")
            print(f"Máximo de pasos por episodio: {self.max_steps_per_episode}")
            print(f"Alpha (tasa de aprendizaje): {self.agent.alpha}")
            print(f"Gamma (factor de descuento): {self.agent.gamma}")
            print(f"Epsilon inicial: {self.agent.epsilon}")
            print(f"Epsilon decay: {self.epsilon_decay}")
            print(f"Epsilon mínimo: {self.epsilon_min}")
            if self.visualizer:
                print("Modo Visual: Activado (Actualización periódica)")
            print("=" * 80)
            print()
        
        start_time = time.time()
        
        # Variables para estadísticas
        total_successes = 0
        recent_rewards = []
        recent_steps = []
        recent_successes = []
        
        for episode in range(1, self.num_episodes + 1):
            # 1. Reiniciar el entorno a un estado inicial aleatorio
            state = self.env.reset(random_start=True, shuffles=50)

            episode_reward = 0.0
            steps = 0
            done = False

            # 2. Ejecutar pasos hasta terminar o alcanzar el límite
            while not done and steps < self.max_steps_per_episode:
                # Obtener acciones válidas
                valid_actions = self.env.get_valid_actions(state)

                if not valid_actions:
                    # No hay acciones válidas (caso extremo)
                    break

                # Seleccionar acción usando política epsilon-greedy
                action = self.agent.choose_action(state, valid_actions)

                # Ejecutar acción en el entorno
                next_state, action_valid = self.env.step(action, verbose=self.show_steps)

                # Obtener recompensa
                reward = self.env.get_reward(next_state, action_valid)
                episode_reward += reward

                # Verificar si llegamos al objetivo
                done = self.env.is_goal(next_state)

                # 3. Actualizar la Q-Table
                next_valid_actions = self.env.get_valid_actions(next_state)
                self.agent.update(state, action, reward, next_state, next_valid_actions)

                # Actualizar estado
                state = next_state
                steps += 1

            # Registrar métricas del episodio
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)
            self.episode_success.append(done)

            # Registrar en Analytics (logging)
            self.analytics.log_episode(episode=episode, steps=steps, total_reward=episode_reward, success=done)

            if done:
                total_successes += 1

            # Mantener ventana de episodios recientes para estadísticas
            recent_rewards.append(episode_reward)
            recent_steps.append(steps)
            recent_successes.append(done)

            if len(recent_rewards) > self.log_interval:
                recent_rewards.pop(0)
                recent_steps.pop(0)
                recent_successes.pop(0)

            # 5. Aplicar decaimiento de epsilon
            if self.agent.epsilon > self.epsilon_min:
                self.agent.set_epsilon(max(self.epsilon_min, self.agent.epsilon * self.epsilon_decay))

            # Mostrar progreso y VISUALIZAR
            if episode % self.log_interval == 0:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                avg_steps = sum(recent_steps) / len(recent_steps)
                success_rate = sum(recent_successes) / len(recent_successes) * 100
                
                if self.verbose:
                    print(f"Episodio {episode}/{self.num_episodes} | "
                          f"Epsilon: {self.agent.epsilon:.4f} | "
                          f"Éxitos totales: {total_successes} | "
                          f"Tasa de éxito (últimos {self.log_interval}): {success_rate:.1f}% | "
                          f"Pasos promedio: {avg_steps:.1f} | "
                          f"Recompensa promedio: {avg_reward:.1f} | "
                          f"Tamaño Q-Table: {len(self.agent.q_table)}")
                
                # ACTUALIZACIÓN VISUAL: Demostrar lo aprendido hasta ahora
                if self.visualizer:
                    self._run_visual_validation(episode)

        
        self.training_time = time.time() - start_time
        
        # Estadísticas finales
        if self.verbose:
            print()
            print("=" * 80)
            print("ENTRENAMIENTO COMPLETADO")
            print("=" * 80)
            self._print_training_summary()
        # Guardar logs de entrenamiento
        log_filename = "data/training_log.csv"
        self.analytics.to_csv(log_filename)
        if self.verbose:
            print(f"\n✓ Log de entrenamiento exportado a {log_filename}")

        return self._get_training_stats()
    
    def _run_visual_validation(self, episode_num: int):
        """
        Ejecuta un breve episodio de validación visual para mostrar el progreso.
        """
        # Guardar estado actual del entorno para no romper el flujo principal
        # (Aunque el entorno se resetea al inicio del bucle, es buena práctica)
        
        # Usamos política greedy para mostrar "lo que sabe"
        original_epsilon = self.agent.epsilon
        self.agent.set_epsilon(0.0)
        
        # Generar un estado de prueba (no muy difícil para que sea rápido de ver)
        val_state = self.env.reset(random_start=True, shuffles=15)
        self.visualizer.update(val_state, f"Entrenando... Ep {episode_num} (Validación)")
        
        # Ejecutar unos pocos pasos para ver si sabe resolverlo
        val_steps = 0
        val_done = False
        max_val_steps = 20 # Limitado para no ralentizar mucho el entrenamiento
        
        while not val_done and val_steps < max_val_steps:
            valid_actions = self.env.get_valid_actions(val_state)
            if not valid_actions:
                break
            
            action = self.agent.choose_action(val_state, valid_actions)
            next_state, _ = self.env.step(action)
            
            self.visualizer.update(next_state, f"Entrenando... Ep {episode_num} | Paso {val_steps+1}")
            
            val_state = next_state
            val_done = self.env.is_goal(val_state)
            val_steps += 1
            
            # Pequeña pausa ya incluida en visualizer.update, pero podemos ajustarla si es lento
            
        # Restaurar epsilon
        self.agent.set_epsilon(original_epsilon)

    def train_single_episode(
        self,
        random_start: bool = True,
        shuffles: int = 50,
        verbose: bool = False
    ) -> Tuple[float, int, bool]:
        """
        Ejecuta un solo episodio de entrenamiento.
        
        Útil para debugging o entrenamiento paso a paso.
        
        Args:
            random_start (bool): Si True, inicia desde un estado aleatorio.
            shuffles (int): Número de movimientos para generar estado inicial.
            verbose (bool): Si True, imprime información detallada del episodio.
        
        Returns:
            Tuple[float, int, bool]: (recompensa_total, pasos, éxito)
        """
        state = self.env.reset(random_start=random_start, shuffles=shuffles)
        
        if verbose:
            print(f"\nEstado inicial:")
            self.env.render()
        
        episode_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < self.max_steps_per_episode:
            valid_actions = self.env.get_valid_actions(state)
            
            if not valid_actions:
                break
            
            action = self.agent.choose_action(state, valid_actions)
            next_state, action_valid = self.env.step(action, verbose=self.show_steps)
            reward = self.env.get_reward(next_state, action_valid)
            episode_reward += reward
            done = self.env.is_goal(next_state)
            
            next_valid_actions = self.env.get_valid_actions(next_state)
            self.agent.update(state, action, reward, next_state, next_valid_actions)
            
            state = next_state
            steps += 1
            
            if verbose:
                print(f"\nPaso {steps} | Acción: {action} | Recompensa: {reward:.1f}")
                self.env.render()
        
        if verbose:
            status = "¡RESUELTO!" if done else "No resuelto"
            print(f"\n{status} | Pasos: {steps} | Recompensa total: {episode_reward:.1f}")
        
        return episode_reward, steps, done
    
    def evaluate(
        self,
        num_episodes: int = 100,
        shuffles: int = 50,
        use_greedy: bool = True
    ) -> Dict:
        """
        Evalúa el rendimiento del agente entrenado.
        
        Args:
            num_episodes (int): Número de episodios de evaluación.
            shuffles (int): Complejidad del estado inicial.
            use_greedy (bool): Si True, usa política greedy (epsilon=0).
        
        Returns:
            Dict: Estadísticas de evaluación.
        """
        original_epsilon = self.agent.epsilon
        
        if use_greedy:
            self.agent.set_epsilon(0.0)  # Política completamente greedy
        
        eval_rewards = []
        eval_steps = []
        eval_successes = []
        
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"EVALUANDO AGENTE ({num_episodes} episodios)")
            print("=" * 80)
        
        for episode in range(1, num_episodes + 1):
            state = self.env.reset(random_start=True, shuffles=shuffles)
            episode_reward = 0.0
            steps = 0
            done = False
            
            while not done and steps < self.max_steps_per_episode:
                valid_actions = self.env.get_valid_actions(state)
                if not valid_actions:
                    break
                
                action = self.agent.choose_action(state, valid_actions)
                next_state, action_valid = self.env.step(action, verbose=self.show_steps)
                reward = self.env.get_reward(next_state, action_valid)
                episode_reward += reward
                done = self.env.is_goal(next_state)
                
                state = next_state
                steps += 1
            
            eval_rewards.append(episode_reward)
            eval_steps.append(steps)
            eval_successes.append(done)
        
        # Restaurar epsilon original
        self.agent.set_epsilon(original_epsilon)
        
        # Calcular estadísticas
        success_rate = sum(eval_successes) / num_episodes * 100
        avg_steps = sum(eval_steps) / num_episodes
        avg_reward = sum(eval_rewards) / num_episodes
        
        successful_episodes = [s for s, success in zip(eval_steps, eval_successes) if success]
        avg_steps_success = sum(successful_episodes) / len(successful_episodes) if successful_episodes else 0
        
        if self.verbose:
            print(f"\nTasa de éxito: {success_rate:.1f}%")
            print(f"Pasos promedio (todos): {avg_steps:.1f}")
            print(f"Pasos promedio (éxitos): {avg_steps_success:.1f}")
            print(f"Recompensa promedio: {avg_reward:.1f}")
            print("=" * 80)
        
        return {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_steps_success': avg_steps_success,
            'avg_reward': avg_reward,
            'rewards': eval_rewards,
            'steps': eval_steps,
            'successes': eval_successes
        }
    
    def _print_training_summary(self):
        """Imprime un resumen detallado del entrenamiento."""
        total_episodes = len(self.episode_rewards)
        total_successes = sum(self.episode_success)
        success_rate = (total_successes / total_episodes * 100) if total_episodes > 0 else 0
        
        avg_reward = sum(self.episode_rewards) / total_episodes if total_episodes > 0 else 0
        avg_steps = sum(self.episode_steps) / total_episodes if total_episodes > 0 else 0
        
        successful_episodes = [s for s, success in zip(self.episode_steps, self.episode_success) if success]
        avg_steps_success = sum(successful_episodes) / len(successful_episodes) if successful_episodes else 0
        
        print(f"Tiempo total: {self.training_time:.2f} segundos")
        print(f"Episodios totales: {total_episodes}")
        print(f"Episodios exitosos: {total_successes}")
        print(f"Tasa de éxito global: {success_rate:.2f}%")
        print(f"Pasos promedio (todos): {avg_steps:.2f}")
        print(f"Pasos promedio (éxitos): {avg_steps_success:.2f}")
        print(f"Recompensa promedio: {avg_reward:.2f}")
        print(f"Tamaño final de Q-Table: {len(self.agent.q_table)} entradas")
        print(f"Epsilon final: {self.agent.epsilon:.6f}")
        print("=" * 80)
    
    def _get_training_stats(self) -> Dict:
        """
        Retorna un diccionario con todas las estadísticas del entrenamiento.
        
        Returns:
            Dict: Estadísticas completas del entrenamiento.
        """
        total_episodes = len(self.episode_rewards)
        total_successes = sum(self.episode_success)
        success_rate = (total_successes / total_episodes * 100) if total_episodes > 0 else 0
        
        avg_reward = sum(self.episode_rewards) / total_episodes if total_episodes > 0 else 0
        avg_steps = sum(self.episode_steps) / total_episodes if total_episodes > 0 else 0
        
        successful_episodes = [s for s, success in zip(self.episode_steps, self.episode_success) if success]
        avg_steps_success = sum(successful_episodes) / len(successful_episodes) if successful_episodes else 0
        
        return {
            'training_time': self.training_time,
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'avg_steps_success': avg_steps_success,
            'q_table_size': len(self.agent.q_table),
            'final_epsilon': self.agent.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'episode_success': self.episode_success
        }
    
    def get_metrics(self) -> Dict:
        """
        Retorna las métricas actuales del entrenamiento.
        
        Returns:
            Dict: Métricas del entrenamiento.
        """
        return self._get_training_stats()
    
    def reset_metrics(self):
        """Reinicia todas las métricas de entrenamiento."""
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_success = []
        self.training_time = 0.0


# TASK-09: Funciones de utilidad para ajuste de hiperparámetros

def create_trainer_with_params(
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    epsilon: float = EPSILON,
    num_episodes: int = NUM_EPISODES,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    epsilon_decay: float = EPSILON_DECAY,
    epsilon_min: float = EPSILON_MIN,
    reward_goal: float = REWARD_GOAL,
    reward_step: float = REWARD_STEP,
    reward_invalid: float = REWARD_INVALID,
    shuffles: int = 50,
    verbose: bool = True
) -> Trainer:
    """
    TASK-09: Crea un entrenador con hiperparámetros personalizados.
    
    Útil para experimentar con diferentes configuraciones de hiperparámetros.
    
    Args:
        alpha (float): Tasa de aprendizaje.
        gamma (float): Factor de descuento.
        epsilon (float): Probabilidad de exploración inicial.
        num_episodes (int): Número de episodios de entrenamiento.
        max_steps (int): Máximo de pasos por episodio.
        epsilon_decay (float): Factor de decaimiento de epsilon.
        epsilon_min (float): Epsilon mínimo.
        reward_goal (float): Recompensa por alcanzar el objetivo.
        reward_step (float): Recompensa por cada paso.
        reward_invalid (float): Penalización por movimiento inválido.
        shuffles (int): Complejidad del estado inicial.
        verbose (bool): Si True, imprime información durante el entrenamiento.
    
    Returns:
        Trainer: Instancia del entrenador configurada.
    """
    # Crear entorno con recompensas personalizadas
    env = EightPuzzle(
        use_reachable_states=False,
        reward_goal=reward_goal,
        reward_step=reward_step,
        reward_invalid=reward_invalid
    )
    
    # Crear Q-Table y agente
    q_table = QTable(initial_value=0.0)
    agent = QLearningAgent(
        q_table=q_table,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma
    )
    
    # Crear entrenador
    trainer = Trainer(
        environment=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        verbose=verbose
    )
    
    return trainer


def run_hyperparameter_experiment(
    param_grid: Dict,
    base_params: Optional[Dict] = None,
    verbose: bool = False
) -> List[Dict]:
    """TASK-09: Ejecuta experimentos con diferentes combinaciones de hiperparámetros."""
    if base_params is None:
        base_params = {}
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    results = []
    
    for combination in itertools.product(*param_values):
        params = base_params.copy()
        params.update(dict(zip(param_names, combination)))
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Experimentando con: {params}")
            print(f"{ '='*80}")
        
        trainer = create_trainer_with_params(**params, verbose=verbose)
        stats = trainer.train()
        
        result = {
            'params': params,
            'stats': stats
        }
        results.append(result)
    
    return results


# Bloque principal: Ejemplo de uso con exportación de logs
if __name__ == "__main__":
    print("\n" + "="*80)
    print("EJEMPLO DE ENTRENAMIENTO Y LOGGING - TASK-10")
    print("="*80 + "\n")
    
    # Configuración y creación del entrenador
    env = EightPuzzle()
    q_table = QTable(initial_value=0.0)
    agent = QLearningAgent(q_table=q_table)
    trainer = Trainer(environment=env, agent=agent, num_episodes=100, verbose=True)
    
    # Entrenar
    trainer.train()

    # Exportar log de episodios de entrenamiento a CSV (TASK-10)
    trainer.analytics.to_csv("training_log.csv")
    print("\n✓ Log de entrenamiento exportado a training_log.csv")

    # Evaluación del agente y exportación de log de evaluación (TASK-10)
    eval_stats = trainer.evaluate(num_episodes=50, shuffles=50, use_greedy=True)
    
    with open("evaluation_log.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "steps", "total_reward", "success"])
        for i in range(len(eval_stats["rewards"])):
            writer.writerow([
                i + 1,
                eval_stats["steps"][i],
                eval_stats["rewards"][i],
                eval_stats["successes"][i]
            ])
    
    print("✓ Log de evaluación exportado a evaluation_log.csv\n")