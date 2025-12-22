"""
Visualizer - Animación de imagen
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from typing import List, Dict, Tuple, Optional

class PuzzleVisualizer:
    def __init__(self, image_path: str = "data/image-1.png", title: str = "8-Puzzle Solver"):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.canvas.manager.set_window_title(title)
        self.ax.axis('off')
        self.slices: Dict[int, np.ndarray] = {}
        self.full_image = mpimg.imread(image_path)
        self._slice_image()
        
        plt.ion()  # Modo interactivo
        plt.show()

    def _slice_image(self):
        """Corta la imagen en 9 pedazos (3x3)."""
        h, w, _ = self.full_image.shape
        sh, sw = h // 3, w // 3
        
        # Mapeo: Posición meta (0-8) -> Slice de imagen
        # El estado objetivo es (1,2,3,4,5,6,7,8,0)
        # 1-8 son las piezas de imagen, 0 es el vacío (último trozo o color sólido)
        
        idx = 1
        for i in range(3):
            for j in range(3):
                # Extraer sub-imagen
                piece = self.full_image[i*sh:(i+1)*sh, j*sw:(j+1)*sw]
                
                # Asignar al número correspondiente en el estado objetivo
                if i == 2 and j == 2:
                    # Última pieza (la que será 0/vacía en el objetivo)
                    # La guardamos como negra o del color de fondo si preferimos
                    # Para el efecto "puzzle", un bloque negro/vacío es típico
                    self.slices[0] = np.zeros_like(piece) 
                else:
                    self.slices[idx] = piece
                    idx += 1

    def update(self, state: Tuple[int, ...], step_info: str = ""):
        """Actualiza el gráfico con el estado actual."""
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_title(step_info)
        
        # Reconstruir imagen desde el estado
        # Estado es una tupla de 9 ints, ej: (1, 2, 3, 4, 5, 6, 7, 8, 0)
        
        # Tamaño de cada slice
        h, w, c = self.slices[1].shape
        combined_image = np.zeros((h*3, w*3, c))
        
        for idx, tile_num in enumerate(state):
            row = idx // 3
            col = idx % 3
            
            tile_img = self.slices[tile_num]
            combined_image[row*h:(row+1)*h, col*w:(col+1)*w] = tile_img
            
        self.ax.imshow(combined_image)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05) # Pausa pequeña para que la animación sea fluida pero visible

    def close(self):
        plt.ioff()
        plt.close()