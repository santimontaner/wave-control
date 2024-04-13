import numpy as np

class Mesh:
    def __init__(self, x_subdivs, y_subdivs, height):
        self.number_of_vertices = (x_subdivs + 1) * (y_subdivs + 1)
        self.number_of_elements = 2 * x_subdivs * y_subdivs
        self.x_subdivs = x_subdivs
        self.y_subdivs = y_subdivs
        self.y_step_size = height / y_subdivs
        self.x_step_size = 1 / x_subdivs
        
        self.initialize_vertices()
        self.initialize_connectivity_graph()
 
        self.base_vertices = np.linspace(0, 1, 2 * self.x_subdivs + 1, dtype=int)
        self.base_boundary_idx = np.arange(0, self.x_subdivs + 1, dtype=int)
        self.right_boundary_idx = self.x_subdivs + (self.x_subdivs+1)*np.arange(0,self.y_subdivs+1,dtype=int)
        self.top_boundary_idx = (self.x_subdivs + 1) * (self.y_subdivs + 1) - 1 - np.arange(0,self.x_subdivs + 1, dtype=int)
        self.left_boundary_idx = (self.x_subdivs + 1) * (self.y_subdivs - np.arange(0, self.y_subdivs + 1, dtype=int))       
        self.base_boundary_elements_idx = np.arange(1,2*x_subdivs,2,dtype=int)

    def initialize_vertices(self):
        self.vertices = np.empty(((self.x_subdivs + 1) * (self.y_subdivs + 1), 2))

        for k in range(self.y_subdivs+1):
            for n in range(self.x_subdivs+1):
                self.vertices[k*(self.x_subdivs+1)+n] = np.array([n*self.x_step_size,k*self.y_step_size])

    def initialize_connectivity_graph(self):
        self.connectivity_array = np.empty((self.number_of_elements, 4), dtype=int)

        for k in range(self.y_subdivs):
            for n in range(self.x_subdivs):
                # Indices of the points in the upper triangle (no segment on R boundary)
                self.connectivity_array[2*(k*self.x_subdivs + n), 0] = k*(self.x_subdivs+1)+n
                self.connectivity_array[2*(k*self.x_subdivs + n), 1] = (k+1)*(self.x_subdivs+1)+n+1
                self.connectivity_array[2*(k*self.x_subdivs + n), 2] = (k+1)*(self.x_subdivs+1)+n

                # Indices of the points in the lower triangle (no segment on L boundary)
                # The edge is on the T0 (k=0) subtriangle
                self.connectivity_array[2*(k*self.x_subdivs + n)+1, 0] = k*(self.x_subdivs+1)+n
                self.connectivity_array[2*(k*self.x_subdivs + n)+1, 1] = k*(self.x_subdivs+1)+n+1
                self.connectivity_array[2*(k*self.x_subdivs + n)+1, 2] = (k+1)*(self.x_subdivs+1)+n+1


                if k == 0:
                    # sets the label = '1', the base boundary
                    if n < self.x_subdivs-1:
                        self.connectivity_array[2*(k*self.x_subdivs+n)+1, 3] = 1
                    else:
                        # sets the label = '5' to the base-right corner
                        self.connectivity_array[2*(k*self.x_subdivs+n)+1 , 3] = 5
                # sets the label = '2', the right-side boundary
                elif n == self.x_subdivs-1:
                    self.connectivity_array[2*(k*self.x_subdivs+n)+1, 3] = 2
                # sets the label = '3', the top-side boundary
                elif k == self.y_subdivs-1:
                    self.connectivity_array[2*(k*self.x_subdivs+n)  , 3] = 3
                # sets the label = '4', the left-side boundary
                elif n == 0:
                    self.connectivity_array[2*(k*self.x_subdivs+n)  , 3] = 4
                # sets the label = '6' to the top-left corner
                elif n == 0 and k == self.y_subdivs-1:
                    self.connectivity_array[2*(k*self.x_subdivs+n) , 3] = 6
                else:
                    self.connectivity_array[2*(k*self.x_subdivs+n)  , 3] = 0
                    self.connectivity_array[2*(k*self.x_subdivs+n)+1, 3] = 0
