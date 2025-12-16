"""
Punto de entrada principal del proyecto.

Este script actúa como un lanzador que invoca al verdadero orquestador
de la aplicación, ubicado en app/main.py.

Al ejecutar este archivo, se inicia el proceso completo:
1. Entrenamiento del agente.
2. Generación de un log con los resultados (`training_log.csv`).
3. Creación de gráficos de rendimiento a partir del log.
"""

from app.main import main

if __name__ == "__main__":
    main()