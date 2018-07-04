import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
# Own modules
import mesh as msh
import TestFunctions as tf
import HCTAssembly as HCT
import HCTMasterFunctions as mf

N = 800
K = 2
T = 0.1

# Creation of the mesh
Th = msh.Mesh(N, K, T)
print("1. Mesh created")

ConnArr = Th.connect
BboundaryNodes = Th.base
RboundaryNodes = Th.right
TboundaryNodes = Th.left

# Evaluation of master functions at gaussian points
D = mf.MasterFunctions([0,0,1,0])

# Matrix assembly
Lp = HCT.PosVelAssembly(Th,D)

# Interpolation of the L^2 initial data u_0(x)
P = HCT.InitPosInter(lambda x: np.sin(5*np.pi*x),N)

# Plot the interpolated initial datum u_0(x)
x = np.linspace(0,1,2*Th.N+1)
plt.plot(x,P)
plt.show()
plt.close()

# Interpolate the function phi(x,y)=y
Q = np.zeros((3*Th.NbPoints,1))
for i, point in enumerate(Th.points):
    x = point[0]
    y = point[1]
    Q[3*i:3*i+3,0] = [y,0,1]

# Compute $\int_0^1 u_0(x) phi_y(x,0) dx$
print(np.dot(np.transpose(Q),Lp.dot(P)))
