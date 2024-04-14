import logging
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix

from ..mesh import Mesh
from . import HctElementMatrixBuilder as fe
from .HctMasterFunctions import HctMasterFunctions

logger = logging.getLogger(__name__)

def build_stiffness_matrix(Th: Mesh, master_eval: HctMasterFunctions):
    """
    Assembly of matrix corresponding to
    int_{Q_T} b (p_{tt}-p_{xx})(q_{tt}-q_{xx})dxdt + int_0^T p_xq_x dt
    for functions p and q vanishing on x=0 and x=1
    """    
    number_of_triangles = Th.connectivity_array.shape[0]
    number_of_vertices = Th.vertices.shape[0]
    number_of_edges = number_of_triangles + number_of_vertices - 1 # Euler formula
    number_of_nonzero_elems = (2 * number_of_edges + number_of_vertices) * 9

    logger.debug("#triangles = %s", number_of_triangles)
    logger.debug("#vertices = %s", number_of_vertices)
    logger.debug("#edges = %s", number_of_edges)
    logger.debug("#nonzero elements in stiffness matrix = %s", number_of_nonzero_elems)

    # 1st row: 'i' indices
    # 2nd row: 'j' indices
    # 3rd row: (i,j) element of the stiffness matrix
    data = np.zeros((3, number_of_triangles*81), order='F')    
    vertices = np.zeros((3,2))
    elem_indices = np.zeros(9, dtype=int)
    
    for triangle_idx, triangle in enumerate(Th.connectivity_array):
        vertices[0,:] = Th.vertices[triangle[0],:]
        vertices[1,:] = Th.vertices[triangle[1],:]
        vertices[2,:] = Th.vertices[triangle[2],:]
        matrix_builder = fe.HctElementMatrixBuilder(vertices, master_eval)        
        
        local_matrix  = matrix_builder.build_interior()        
        if (triangle[3] == 2 or triangle[3] == 5):
            local_matrix += matrix_builder.build_boundary(0)
                
        elem_indices = [3*triangle[0], 3*triangle[0]+1, 3*triangle[0]+2,
                        3*triangle[1], 3*triangle[1]+1, 3*triangle[1]+2,
                        3*triangle[2], 3*triangle[2]+1, 3*triangle[2]+2]
        
        for local_idx, global_idx in enumerate(elem_indices):
            data[0, 81*triangle_idx + 9*local_idx : 81*triangle_idx + 9*local_idx + 9] = global_idx * np.ones(9, dtype=int)
            data[1, 81*triangle_idx + 9*local_idx : 81*triangle_idx + 9*local_idx + 9] = elem_indices
        data[2, 81*triangle_idx:81*(triangle_idx+1)] = np.reshape(local_matrix, (81,))
        
    matrix = lil_matrix(
        coo_matrix((data[2,:], (data[0,:], data[1,:])), shape=(3*Th.number_of_vertices, 3*Th.number_of_vertices))
        )
    
    # Homogeneous Boundary conditions
    for local_idx in np.append(Th.right_boundary_idx, Th.left_boundary_idx):
        matrix[3*local_idx,:]   = 0
        matrix[:,3*local_idx]   = 0
        matrix[3*local_idx+2,:] = 0
        matrix[:,3*local_idx+2] = 0
        matrix[3*local_idx,3*local_idx] = 1
        matrix[3*local_idx+2,3*local_idx+2] = 1
    return csr_matrix(matrix)

def build_initial_conditions_matrix(Th: Mesh, master_eval):
    base_triangles = Th.base_boundary_elements_idx
    
    initial_pos_matrix = lil_matrix((3*Th.number_of_vertices, 2*Th.n_x+1))
    initial_vel_matrix = lil_matrix((3*Th.number_of_vertices, 2*Th.n_x+1))
    
    for triangle_idx in base_triangles:
        triangle = Th.connectivity_array[triangle_idx]
        vertices = np.array([Th.vertices[triangle[0]], Th.vertices[triangle[1]], Th.vertices[triangle[2]]])
        matrix_builder = fe.HctElementMatrixBuilder(vertices, master_eval)
        # on the bottom boundary we always integrate over the 3rd subtriangle,
        # which corresponds to k=2, thus, we set:
        initial_local_pos_matrix = matrix_builder.build_init_pos(2)
        initial_local_vel_matrix = matrix_builder.build_init_vel(2)
        # We allocate the local contribution into the global matrix
        for i in range(3):
            initial_pos_matrix[3*triangle[i]:3*triangle[i]+3, 2*triangle[0]:2*triangle[0]+2] += initial_local_pos_matrix[3*i:3*i+3, 0:2]
            initial_vel_matrix[3*triangle[i]:3*triangle[i]+3, 2*triangle[0]:2*triangle[0]+2] += initial_local_vel_matrix[3*i:3*i+3, 0:2]

    initial_pos_matrix = csr_matrix(initial_pos_matrix)
    initial_vel_matrix = csr_matrix(initial_vel_matrix)
    return initial_pos_matrix, initial_vel_matrix

def InterpolationP1(f,N):
    DelX = 1/N
    P = np.zeros((2*N+1,1))
    for i in range(0,N):
        x1 = i*DelX
        x2 = x1+DelX*0.5
        P[2*i] = f(x1)
        P[2*i+1] = f(x2)
    P[2*N] = f(N*DelX)
    return P
