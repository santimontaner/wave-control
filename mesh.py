import numpy as np

class Mesh:
    def __init__(self, n_x, n_y, height):
        self.number_of_vertices = (n_x + 1) * (n_y + 1)
        self.number_of_elements = 2 * n_x * n_y
        self.n_x = n_x
        self.n_y = n_y
        self.h_y = height / n_y
        self.h_x = 1 / n_x
        
        self.initialize_vertices()
        self.initialize_connectivity_array()
         
        self.base_boundary_idx = np.arange(0, n_x + 1, dtype=int)
        self.right_boundary_idx = n_x + (n_x + 1) * np.arange(0, n_y + 1, dtype=int)
        self.top_boundary_idx = (n_x + 1) * (n_y + 1) - 1 - np.arange(0, n_x + 1, dtype=int)
        self.left_boundary_idx = (n_x + 1) * (n_y - np.arange(0, n_y + 1, dtype=int))       
        self.base_boundary_elements_idx = np.arange(1, 2 * (n_x), 2, dtype=int)

    def initialize_vertices(self):
        self.vertices = np.empty(((self.n_x + 1) * (self.n_y + 1), 2))

        for index_y in range(self.n_y + 1):
            for index_x in range(self.n_x + 1):
                self.vertices[index_y * (self.n_x + 1) + index_x] = np.array([index_x * self.h_x, index_y * self.h_y])

    @property
    def connectivity_array(self):
        return self._connectivity_array

    def initialize_connectivity_array(self):
        self._connectivity_array = np.empty((self.number_of_elements, 4), dtype=int)

        for k in range(self.n_y):
            for n in range(self.n_x):
                # Indices of the vertices in the upper triangle (no segment on R boundary)
                self._connectivity_array[2*(k*self.n_x + n), 0] = k     * (self.n_x + 1) + n
                self._connectivity_array[2*(k*self.n_x + n), 1] = (k+1) * (self.n_x + 1) + n + 1
                self._connectivity_array[2*(k*self.n_x + n), 2] = (k+1) * (self.n_x + 1) + n

                # Indices of the vertices in the lower triangle (no segment on L boundary)
                # The edge is on the T0 (k=0) subtriangle
                self._connectivity_array[2*(k*self.n_x + n) + 1, 0] = k     * (self.n_x + 1) + n
                self._connectivity_array[2*(k*self.n_x + n) + 1, 1] = k     * (self.n_x + 1) + n + 1
                self._connectivity_array[2*(k*self.n_x + n) + 1, 2] = (k+1) * (self.n_x + 1) + n + 1

                if k == 0 and n < self.n_x-1: # base
                    self._connectivity_array[2 * (k * self.n_x + n) + 1, 3] = 1
                elif k == 0: # base/right                     
                    self._connectivity_array[2 * (k * self.n_x + n) + 1, 3] = 5                
                elif n == self.n_x-1: # right
                    self._connectivity_array[2 * (k * self.n_x + n) + 1, 3] = 2
                elif k == self.n_y-1: # top
                    self._connectivity_array[2 * (k * self.n_x + n), 3] = 3
                elif n == 0: # left
                    self._connectivity_array[2 * (k * self.n_x + n), 3] = 4
                elif n == 0 and k == self.n_y-1: # top/left
                    self._connectivity_array[2 * (k * self.n_x + n), 3] = 6
                else:
                    self._connectivity_array[2 * (k * self.n_x + n), 3] = 0
                    self._connectivity_array[2 * (k * self.n_x + n) + 1, 3] = 0
