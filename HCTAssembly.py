import logging
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from timeit import default_timer as timer
from mesh import Mesh

# Own packages
import HctElementMatrixBuilder as fe

logger = logging.getLogger(__name__)

# Assembly of matrix corresponding to
# \int_{Q_T} b (p_{tt}-p_{xx})(q_{tt}-q_{xx})dxdt + \int_0^T p_xq_x dt
# for functions p and q vanishing on x=0 and x=1
def build_stiffness_matrix(Th: Mesh, D):    
    number_of_triangles = Th.connectivity_array.shape[0]
    number_of_vertices = Th.vertices.shape[0]
    number_of_edges = number_of_triangles + number_of_vertices - 1 # Euler formula
    number_of_nonzero_elems = (2 * number_of_edges + number_of_vertices) * 9

    logger.debug("#triangles = %s", number_of_triangles)
    logger.debug("#vertices = %s", number_of_vertices)
    logger.debug("#edges = %s", number_of_edges)
    logger.debug("#nonzero elements in stiffness matrix = %s", number_of_nonzero_elems)

    # The first row in 'data' array contains the 'i' indices
    # The second row in 'data' array contains the 'j' indices
    # The third row in 'data' array contains the (i,j) element of the stiffness
    # matrix
    data = np.empty((3, number_of_triangles*81),order='F')

    start = timer()
    vertices = np.empty((3,2))
    localI = np.empty(9, dtype=int)
    
    for triangle_idx, triangle in enumerate(Th.connectivity_array):
        vertices[0,:] = Th.vertices[triangle[0],:]
        vertices[1,:] = Th.vertices[triangle[1],:]
        vertices[2,:] = Th.vertices[triangle[2],:]
        matrix_builder = fe.HctElementMatrixBuilder(vertices, D)        
        local_matrix  = matrix_builder.build_interior()
        # triangle[2] is the 'boundary label' corresponding to x=1.
        if (triangle[3] == 2 or triangle[3] == 5):
            local_matrix += matrix_builder.build_boundary(0)
        local_matrix = np.reshape(local_matrix, (81,))
        localI = [3*triangle[0], 3*triangle[0]+1, 3*triangle[0]+2,
                  3*triangle[1], 3*triangle[1]+1, 3*triangle[1]+2,
                  3*triangle[2], 3*triangle[2]+1, 3*triangle[2]+2]
        for i, Iindex in enumerate(localI):
            data[0, 81*triangle_idx + 9*i : 81*triangle_idx + 9*i + 9] = Iindex * np.ones(9, dtype=int)
            data[1, 81*triangle_idx + 9*i : 81*triangle_idx + 9*i + 9] = localI
        data[2,81*triangle_idx:81*(triangle_idx+1)] = local_matrix
    
    stiffness_matrix = coo_matrix((data[2,:], (data[0,:], data[1,:])), shape=(3*Th.number_of_vertices, 3*Th.number_of_vertices))
    stiffness_matrix = lil_matrix(stiffness_matrix)
    end = timer()
    print("Matrix assembled! ("+str(end-start)+"s.)")

    # Homogeneous Boundary conditions
    for i in np.append(Th.right_boundary_idx, Th.left_boundary_idx):
        stiffness_matrix[3*i,:]   = 0
        stiffness_matrix[:,3*i]   = 0
        stiffness_matrix[3*i+2,:] = 0
        stiffness_matrix[:,3*i+2] = 0
        stiffness_matrix[3*i,3*i] = 1
        stiffness_matrix[3*i+2,3*i+2] = 1
    stiffness_matrix  = csr_matrix(stiffness_matrix)
    return stiffness_matrix

def build_initial_conditions_matrix(Th: Mesh, D):
    base_triangles = Th.base_boundary_elements_idx
    
    initial_pos_matrix = lil_matrix((3*Th.number_of_vertices,2*Th.n_x+1))
    initial_vel_matrix = lil_matrix((3*Th.number_of_vertices,2*Th.n_x+1))
    
    for triangle_idx in base_triangles:
        triangle = Th.connectivity_array[triangle_idx]
        vertices = np.array([Th.vertices[triangle[0]], Th.vertices[triangle[1]], Th.vertices[triangle[2]]])
        matrix_builder = fe.HctElementMatrixBuilder(vertices, D)
        # on the bottom boundary we always integrate over the 3rd subtriangle,
        # which corresponds to k=2, thus, we set:
        initial_local_pos_matrix = matrix_builder.build_init_pos(2)
        initial_local_vel_matrix = matrix_builder.build_init_vel(2)
        # We allocate the local contribution into the global matrix
        for i in range(3):
            initial_pos_matrix[3*triangle[i]:3*triangle[i]+3,2*triangle[0]:2*triangle[0]+2] += initial_local_pos_matrix[3*i:3*i+3,0:2]
            initial_vel_matrix[3*triangle[i]:3*triangle[i]+3,2*triangle[0]:2*triangle[0]+2] += initial_local_vel_matrix[3*i:3*i+3,0:2]

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
