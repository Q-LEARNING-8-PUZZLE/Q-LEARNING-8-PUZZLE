
"""
Analytics - Dev 4
TASK-10: Registro de métricas por episodio y logging
"""

import csv
from typing import List, Dict, Any, Optional

class Analytics:
	"""
	Clase para registrar y exportar métricas de entrenamiento por episodio.
	Registra: número de pasos, recompensa total, éxito/fallo y permite exportar a CSV o listas.
	"""
	def __init__(self):
		self.episodes: List[Dict[str, Any]] = []

	def log_episode(self, episode: int, steps: int, total_reward: float, success: bool):
		"""
		Registra los datos de un episodio.
		Args:
			episode (int): Número de episodio
			steps (int): Pasos realizados
			total_reward (float): Recompensa total acumulada
			success (bool): True si resolvió el puzzle, False si no
		"""
		self.episodes.append({
			'episode': episode,
			'steps': steps,
			'total_reward': total_reward,
			'success': success
		})

	def to_list(self) -> List[Dict[str, Any]]:
		"""Devuelve la lista de episodios registrados."""
		return self.episodes

	def to_csv(self, filename: str = "training_log.csv"):
		"""
		Exporta los datos registrados a un archivo CSV.
		Args:
			filename (str): Nombre del archivo CSV de salida
		"""
		if not self.episodes:
			return
		fieldnames = ['episode', 'steps', 'total_reward', 'success']
		with open(filename, mode='w', newline='', encoding='utf-8') as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for row in self.episodes:
				writer.writerow(row)

	def reset(self):
		"""Limpia todos los registros almacenados."""
		self.episodes.clear()


