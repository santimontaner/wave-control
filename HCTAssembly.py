import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from timeit import default_timer as timer
from mesh import Mesh

# Own packages
import rHCTelement as fe

# Assembly of matrix corresponding to
# \int_{Q_T} b (p_{tt}-p_{xx})(q_{tt}-q_{xx})dxdt + \int_0^T p_xq_x dt
# for functions p and q vanishing on x=0 and x=1
def StiffAssembly(Th: Mesh, D):
    ConnArr = Th.connectivity_array
    RboundaryNodes = Th.right_boundary_idx
    LboundaryNodes = Th.left_boundary_idx
    Nodes = Th.vertices

    # Number of triangles
    Ntri = ConnArr.shape[0]
    # Number of nodes
    Nnodes = Nodes.shape[0]
    # Number of edges (Euler formula)
    Nedges = Ntri+Nnodes-1
    Nonzero = (2*Nedges+Nnodes)*9
    print("Nb of triangles = "+str(Ntri))
    print("Nb of nodes = "+str(Nnodes))
    print("Nb of edges = "+str(Nedges))
    print("Nb of nonzero elements in stiffness matrix = "+str(Nonzero))

    # The first row in 'data' array contains the 'i' indices
    # The second row in 'data' array contains the 'j' indices
    # The third row in 'data' array contains the (i,j) element of the stiffness
    # matrix
    data = np.empty((3,Ntri*81),order='F')

    start = timer()
    points = np.empty((3,2))
    localI = np.empty(9,dtype=int)
    for tri,triangle in enumerate(ConnArr):
        points[0,:] = Nodes[triangle[0],:]
        points[1,:] = Nodes[triangle[1],:]
        points[2,:] = Nodes[triangle[2],:]
        El = fe.rHCT_FE(points, D)
        localA  = El.InteriorStiffness()
        # triangle[2] is the 'boundary label' corresponding to x=1.
        if (triangle[3] == 2 or triangle[3] == 5):
            localA = localA + El.BoundaryStiffness(0)
        localA = np.reshape(localA,(81,))
        localI = [3*triangle[0],3*triangle[0]+1,3*triangle[0]+2, \
                  3*triangle[1],3*triangle[1]+1,3*triangle[1]+2, \
                  3*triangle[2],3*triangle[2]+1,3*triangle[2]+2]
        for i,Iindex in enumerate(localI):
            data[0,81*tri+9*i:81*tri+9*i+9] = Iindex*np.ones(9,dtype=int)
            data[1,81*tri+9*i:81*tri+9*i+9] = localI
        data[2,81*tri:81*(tri+1)] = localA
    A = coo_matrix((data[2,:],(data[0,:],data[1,:])),shape=(3*Th.number_of_vertices, 3*Th.number_of_vertices))
    A = lil_matrix(A)
    end = timer()
    print("Matrix assembled! ("+str(end-start)+"s.)")

    # Homogeneous Boundary conditions
    for i in np.append(RboundaryNodes,LboundaryNodes):
        A[3*i,:]   = 0
        A[:,3*i]   = 0
        A[3*i+2,:] = 0
        A[:,3*i+2] = 0
        A[3*i,3*i] = 1
        A[3*i+2,3*i+2] = 1
    A  = csr_matrix(A)
    return A

def PosVelAssembly(Th: Mesh, D):
    baseTriangles = Th.base_boundary_elements_idx
    
    initial_pos_matrix = lil_matrix((3*Th.number_of_vertices,2*Th.n_x+1))
    # Velocity Matrix Assembly
    Lv = lil_matrix((3*Th.number_of_vertices,2*Th.x_subdivs+1))

    print(baseTriangles.shape)
    for triIndex in baseTriangles:
        triangle = Th.connectivity_array[triIndex]
        vertices = np.array([Th.vertices[triangle[0]],
                           Th.vertices[triangle[1]],
                           Th.vertices[triangle[2]]])
        El = fe.rHCT_FE(vertices, D)
        # on the bottom boundary we always integrate over the 3rd subtriangle,
        # which corresponds to k=2, thus, we set:
        localLp = El.InitPositionMatrix(2)
        localLv = El.InitVelocityMatrix(2)
        # We allocate the local contribution into the global matrix
        for i in range(3):
            Lp[3*triangle[i]:3*triangle[i]+3,2*triangle[0]:2*triangle[0]+2] \
                    += localLp[3*i:3*i+3,0:2]
            Lv[3*triangle[i]:3*triangle[i]+3,2*triangle[0]:2*triangle[0]+2] \
                    += localLv[3*i:3*i+3,0:2]

    Lp = csr_matrix(Lp)
    Lv = csr_matrix(Lv)
    return Lp, Lv

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
