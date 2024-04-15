from enum import Enum
import numpy as np


class TriangleType(Enum):
    Base = 1
    BaseRight = 5
    Right = 2
    Top = 3
    Left = 4
    TopLeft = 6
    Inteior = 0


class Mesh:
    def __init__(self, n_x, n_y, height):
        self.number_of_vertices = (n_x + 1) * (n_y + 1)
        self.number_of_elements = 2 * n_x * n_y
        self.n_x = n_x
        self.n_y = n_y
        self.h_y = height / n_y
        self.h_x = 1 / n_x

        self._initialize_vertices()
        self._initialize_connectivity_array()

        self.base_boundary_idx = np.arange(0, n_x + 1, dtype=int)
        self.right_boundary_idx = n_x + (n_x + 1) * np.arange(0, n_y + 1, dtype=int)
        self.top_boundary_idx = (n_x + 1) * (n_y + 1) - 1 - np.arange(0, n_x + 1, dtype=int)
        self.left_boundary_idx = (n_x + 1) * (n_y - np.arange(0, n_y + 1, dtype=int))
        self.base_boundary_elements_idx = np.arange(1, 2 * (n_x), 2, dtype=int)

    def _initialize_vertices(self):
        self.vertices = np.zeros(((self.n_x + 1) * (self.n_y + 1), 2))

        for index_y in range(self.n_y + 1):
            for index_x in range(self.n_x + 1):
                self.vertices[index_y * (self.n_x + 1) + index_x] = np.array([index_x * self.h_x, index_y * self.h_y])

    @property
    def connectivity_array(self):
        return self._connectivity_array

    def _initialize_connectivity_array(self):
        self._connectivity_array = np.zeros((self.number_of_elements, 4), dtype=int)

        for i in range(self.n_y):
            for j in range(self.n_x):
                # Upper triangles
                self._connectivity_array[2 * (i * self.n_x + j), 0] = i * (self.n_x + 1) + j
                self._connectivity_array[2 * (i * self.n_x + j), 1] = (i + 1) * (self.n_x + 1) + j + 1
                self._connectivity_array[2 * (i * self.n_x + j), 2] = (i + 1) * (self.n_x + 1) + j

                # Lower triangles
                self._connectivity_array[2 * (i * self.n_x + j) + 1, 0] = i * (self.n_x + 1) + j
                self._connectivity_array[2 * (i * self.n_x + j) + 1, 1] = i * (self.n_x + 1) + j + 1
                self._connectivity_array[2 * (i * self.n_x + j) + 1, 2] = (i + 1) * (self.n_x + 1) + j + 1

                triangle_type_int = self.map_triangle_type_to_int(self._get_triangle_type(i, j))
                self._connectivity_array[2 * (i * self.n_x + j) + 1, 3] = triangle_type_int

    def _get_triangle_type(self, k, n):
        if k == 0 and n < self.n_x - 1:
            return TriangleType.Base
        elif k == 0:
            return TriangleType.BaseRight
        elif n == self.n_x - 1:
            return TriangleType.Right
        elif k == self.n_y - 1:
            return TriangleType.Top
        elif n == 0:
            return TriangleType.Left
        elif n == 0 and k == self.n_y - 1:
            return TriangleType.TopLeft
        else:
            return TriangleType.Inteior

    _type_to_int_map = {
        TriangleType.Base: 1,
        TriangleType.BaseRight: 5,
        TriangleType.Right: 2,
        TriangleType.Top: 3,
        TriangleType.Left: 4,
        TriangleType.TopLeft: 6,
        TriangleType.Inteior: 0
    }

    def map_int_to_triangle_type(self, type: int) -> TriangleType:
        inverse_map = {v: k for k, v in self._type_to_int_map}
        return inverse_map[type]

    def map_triangle_type_to_int(self, triangle_type: TriangleType) -> int:
        return self._type_to_int_map[triangle_type]
