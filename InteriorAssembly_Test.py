import numpy as np
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Own packages
import mesh as mesh
import HCTAssembly as HCT
import HCTMasterFunctions as mf
import TestFunctions as tf
import rHCTelement as HCTel

# Evaluation of master functions at gaussian points
D = mf.MasterFunctions([1,1,1,1])

# Global Matrix Test
T = 2
N = 80
K = 2*N
DelX = 1/N
DelT = T/K
b = 1

# Creation of the mesh
Th = mesh.Mesh(N, K, T)
# Matrix assembly
A = HCT.StiffAssembly(Th,D,b)
print("Global Matrix Test:")
# Interpolation of the test functions (Global Matrix)
P = np.zeros((3*Th.number_of_vertices,1))
Q = np.zeros((3*Th.number_of_vertices,1))

# tf.p4, tf.p4x, tf.p4y are defined in in TestFunctions.py and vanish on x=0,1.
f = tf.p4
fx = tf.p4x
fy = tf.p4y

for i, point in enumerate(Th.vertices):
    x = point[0]
    y = point[1]
    P[3*i:3*i+3,0] = [f(x,y),fx(x,y),fy(x,y)]
Q = A.dot(P)

# This should print the value of the integral
## int_Q_T (dtt(f)(x,t)-dxx(f)(x,t))^2 dxdt + int_0^T (dx(f)(1,t))^2 dt
print(np.dot(np.transpose(Q),P))
