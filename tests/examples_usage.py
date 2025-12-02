"""
Ejemplos de uso del entorno 8-Puzzle con las mejoras implementadas.

Este archivo demuestra las nuevas funcionalidades configurables:
1. Recompensas personalizables
2. Modo verbose opcional en step()
3. Modo return_string en render()
"""

import sys
from pathlib import Path

# Añadir el directorio padre al path para importar el módulo app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.environment import EightPuzzle
from app.config import REWARD_GOAL, REWARD_STEP, REWARD_INVALID

def ejemplo_1_recompensas_por_defecto():
    """Ejemplo 1: Usar recompensas por defecto"""
    print("="*70)
    print("EJEMPLO 1: Recompensas por defecto")
    print("="*70)
    
    env = EightPuzzle()
    print(f"Recompensa objetivo: {env.reward_goal}")
    print(f"Recompensa paso: {env.reward_step}")
    print(f"Recompensa inválido: {env.reward_invalid}")
    
    # Probar recompensas
    print(f"\nRecompensa al objetivo: {env.get_reward(env.GOAL_STATE, True)}")
    print(f"Recompensa paso normal: {env.get_reward((1,2,3,4,5,6,7,0,8), True)}")
    print(f"Recompensa movimiento inválido: {env.get_reward(env.state, False)}")


def ejemplo_2_recompensas_personalizadas():
    """Ejemplo 2: Configurar recompensas personalizadas"""
    print("\n" + "="*70)
    print("EJEMPLO 2: Recompensas personalizadas")
    print("="*70)
    
    # Crear entorno con recompensas diferentes
    env = EightPuzzle(
        reward_goal=500.0,      # Recompensa más baja para objetivo
        reward_step=-0.5,       # Penalización más suave por paso
        reward_invalid=-50.0    # Penalización más suave por inválido
    )
    
    print(f"Recompensa objetivo: {env.reward_goal}")
    print(f"Recompensa paso: {env.reward_step}")
    print(f"Recompensa inválido: {env.reward_invalid}")
    
    # Probar recompensas
    print(f"\nRecompensa al objetivo: {env.get_reward(env.GOAL_STATE, True)}")
    print(f"Recompensa paso normal: {env.get_reward((1,2,3,4,5,6,7,0,8), True)}")
    print(f"Recompensa movimiento inválido: {env.get_reward(env.state, False)}")


def ejemplo_3_verbose_activado():
    """Ejemplo 3: Usar verbose=True para ver detalles de cada paso"""
    print("\n" + "="*70)
    print("EJEMPLO 3: Modo verbose activado (útil para debugging)")
    print("="*70)
    
    env = EightPuzzle()
    env.state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    
    print("\nEstado inicial:")
    env.render()
    
    print("\nEjecutando movimiento VÁLIDO con verbose=True:")
    env.step(0, verbose=True)  # Mover arriba
    
    print("\nEjecutando movimiento INVÁLIDO con verbose=True:")
    env.step(1, verbose=True)  # Intentar mover abajo (inválido desde esquina)


def ejemplo_4_verbose_desactivado():
    """Ejemplo 4: Usar verbose=False para entrenamiento sin prints"""
    print("\n" + "="*70)
    print("EJEMPLO 4: Modo verbose desactivado (ideal para entrenamiento)")
    print("="*70)
    
    env = EightPuzzle()
    env.state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    
    print("\nEstado inicial:")
    env.render()
    
    print("\nEjecutando 5 pasos sin verbose (sin prints molestos):")
    actions = [0, 2, 1, 3, 0]  # Secuencia de acciones
    for i, action in enumerate(actions):
        new_state, valid = env.step(action, verbose=False)
        print(f"  Paso {i+1}: Acción {action} -> Válida: {valid}")
    
    print("\nEstado final:")
    env.render()


def ejemplo_5_render_return_string():
    """Ejemplo 5: Usar render con return_string para logging"""
    print("\n" + "="*70)
    print("EJEMPLO 5: render() con return_string (útil para logs)")
    print("="*70)
    
    env = EightPuzzle()
    env.state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    
    # Obtener representación como string sin imprimir
    estado_str = env.render(return_string=True)
    
    print("String del estado capturado:")
    print(repr(estado_str))
    
    print("\nPuede usarse para logging o guardar en archivos:")
    print(f"LOG: Estado actual = {estado_str}")
    
    # Comparación: render normal sí imprime
    print("\nRender normal (imprime directamente):")
    env.render(return_string=False)


def ejemplo_6_uso_desde_config():
    """Ejemplo 6: Usar valores de config.py"""
    print("\n" + "="*70)
    print("EJEMPLO 6: Configuración desde config.py")
    print("="*70)
    
    # Importar valores de config
    print(f"Valores en config.py:")
    print(f"  REWARD_GOAL = {REWARD_GOAL}")
    print(f"  REWARD_STEP = {REWARD_STEP}")
    print(f"  REWARD_INVALID = {REWARD_INVALID}")
    
    # Crear entorno usando valores de config
    env = EightPuzzle(
        reward_goal=REWARD_GOAL,
        reward_step=REWARD_STEP,
        reward_invalid=REWARD_INVALID
    )
    
    print(f"\nEntorno configurado con valores de config.py")
    print(f"  env.reward_goal = {env.reward_goal}")
    print(f"  env.reward_step = {env.reward_step}")
    print(f"  env.reward_invalid = {env.reward_invalid}")


def ejemplo_7_entrenamiento_silencioso():
    """Ejemplo 7: Simulación de entrenamiento silencioso"""
    print("\n" + "="*70)
    print("EJEMPLO 7: Simulación de entrenamiento (sin prints)")
    print("="*70)
    
    env = EightPuzzle()
    
    print("\nEjecutando 3 episodios de ejemplo...")
    for episode in range(3):
        # Reset con estado aleatorio
        state = env.reset(random_start=True, shuffles=10)
        steps = 0
        total_reward = 0.0
        
        # Simular hasta 20 pasos o resolver
        while steps < 20 and not env.is_goal(env.state):
            # Elegir acción aleatoria de las válidas
            import random
            valid_actions = env.get_valid_actions()
            action = random.choice(valid_actions)
            
            # Ejecutar sin verbose
            new_state, valid = env.step(action, verbose=False)
            reward = env.get_reward(new_state, valid)
            total_reward += reward
            steps += 1
            
            if env.is_goal(env.state):
                break
        
        # Solo imprimir resumen del episodio
        solved = "RESUELTO" if env.is_goal(env.state) else "NO RESUELTO"
        print(f"  Episodio {episode+1}: {steps} pasos, Recompensa: {total_reward:.1f} {solved}")


def main():
    """Ejecuta todos los ejemplos"""
    print("\n" + "=" * 70)
    print("EJEMPLOS DE USO - ENTORNO 8-PUZZLE MEJORADO")
    print("=" * 70 + "\n")
    
    ejemplo_1_recompensas_por_defecto()
    ejemplo_2_recompensas_personalizadas()
    ejemplo_3_verbose_activado()
    ejemplo_4_verbose_desactivado()
    ejemplo_5_render_return_string()
    ejemplo_6_uso_desde_config()
    ejemplo_7_entrenamiento_silencioso()
    
    print("\n" + "="*70)
    print("Todos los ejemplos ejecutados correctamente")
    print("="*70)


if __name__ == "__main__":
    main()
