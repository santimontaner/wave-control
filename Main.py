import numpy as np
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
from timeit import default_timer as timer
# Own packages
import mesh
import HCTAssembly as HCT
import HctMasterFunctions as mf
import WaveTimeMarching as wtm
import Plots

# Evaluation of master functions at gaussian points
master_eval = mf.HctMasterFunctions([True,True,True,True])

# Data for the mesh
T = 2
N = 30
K = int(2.2*N)
DelX = 1/N
DelT = T/K
# Creation of the mesh
Th = mesh.Mesh(N, K, T)

print("Uniform mesh dimension: [N,K]=["+str(N)+","+str(K)+"]")
print("T= ",T)
print("DelX= ",DelX)
print("DelT= ",DelT)

# Ellipticity parameter, related to the weight that we put in the functional
# that we want to solve

# Stiffness and Load Matrices assembly
A = HCT.build_stiffness_matrix(Th, master_eval)
Lp, Lv = HCT.build_initial_conditions_matrix(Th,master_eval)

# Interpolation of the L^2 initial data
f = lambda x: np.sin(2*np.pi*x)

#g = lambda x: np.sin(3*np.pi*x)
g = lambda x: 0
P = HCT.InterpolationP1(f,N)
Q = HCT.InterpolationP1(g,N)

xx = np.linspace(0,1,P.shape[0])
plt.plot(xx,P)
plt.show()
plt.close()

# Set the Load Vector from the Load Matrix and the interpolated initial data
F1 = -Lp.dot(P)
F2 = Lv.dot(Q)
# Homogeneous Dirichlet boundary conditions
F1[3*Th.right_boundary_idx]=0
F1[3*Th.left_boundary_idx]=0
F1[3*Th.right_boundary_idx+2]=0
F1[3*Th.left_boundary_idx+2]=0
F2[3*Th.right_boundary_idx]=0
F2[3*Th.left_boundary_idx]=0
F2[3*Th.right_boundary_idx+2]=0
F2[3*Th.left_boundary_idx+2]=0

# Linear Solver
start= timer()
U = la.spsolve(A,F1,use_umfpack = True)
end = timer()
print("Solved! ("+str(end-start)+"s.)")
print("MAX: ",np.amax(U))
print("MIN: ",np.amin(U))

# Store the vector solution in a (K+1)x(N+1)x3 array
u = wtm.toGrid(U,K,N)
# Store the gradient in x at the boundary x=1
GradientAtBoundary = u[:,N,1]

# Sampling points
x = np.linspace(0,1,N+1)
t = np.linspace(0,T,K+1)

# Plot control
plt.plot(t,GradientAtBoundary)
plt.show()
plt.close()

# Plot Initial data
plt.plot(x,u[0,:,1:])
plt.show()
plt.close()

# Initial data for the a posteriori computation
y0 = f(x)
y1 = g(x)

L = 1
y = wtm.Explicit(y0,y1,GradientAtBoundary,None, L, T, N, K)
# Plot Final data
plt.plot(x,y[K,:])
plt.show()
plt.close()
# Plot space-time grid
Plots.SpaceTimePlot(y, L, T, N, K)
