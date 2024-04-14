######################################################################
# Matrix for the implicit scheme:
# Returns the LU factorization of the
# matrix
# N: number of spatial subdivisions
# alfa1 : latest-weight for the coupling term in the implicit scheme
# theta1: latest-weight for the Laplacian in the implicit scheme
# DelT: time step
# DelX: spatial step

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

def Matrix(N,alfa1,theta1,DelT,DelX):
    ratio = (DelX/DelT)**2
    # (1,1) block
    asup = -np.ones(N-1)
    ainf = -np.ones(N-1)
    ainf[N-2] = 0
    b = (2+ratio/theta1)*np.ones(N)
    b[N-1] = 1
    diagBlock11 = np.array([ainf,b,asup])
    diags = [-1, 0, 1]
    block11 = sp.diags(diagBlock11, diags, format ="csc")
    # (2,2) block
    diagBlock22 = np.array([-np.ones(N-2),(2+ratio/theta1)*np.ones(N-1),-np.ones(N-2)])
    diags = [-1, 0, 1]
    block22 = sp.diags(diagBlock22, diags, format ="csc")
    # (1,2) block
    diagBlock12 = np.array([np.zeros(N-2),np.zeros(N-1)])
    diagBlock12[0][N-3] = -0.5/DelX
    diagBlock12[1][N-2] = 2./DelX
    diags = [-2,-1]
    block12 = sp.diags(diagBlock12, diags, shape=(N, N-1), format ="csc")
    # (2,1) block
    diagBlock21 = [-(alfa1/theta1)*np.ones(N)*(DelX)**2]
    block21 = sp.diags(diagBlock21, [0], shape=(N-1, N),format ="csc")
    # Block assembling
    A = sp.bmat([[block11, block12], [block21, block22]],format="csc")
    # Matrix LU factorization
    C = la.splu(A)
    return C
