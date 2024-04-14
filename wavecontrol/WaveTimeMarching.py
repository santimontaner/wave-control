import numpy as np
import scipy.sparse.linalg as la
import math as math
import matplotlib.pyplot as plt
import scipy.sparse as sp
from .matrix import *

def Implicit(y0,y1,boundaryData,f, L, T, N, K):
    # Spatial discretization step
    DelX = L/N
    # Time discretization step
    DelT = T/K
    ratio = (DelX/DelT)**2
    # Weights for the implicit scheme
    theta1 = 1./4
    theta2 = 1./2
    theta3 = 1./4
    # Matrix for the implicit scheme and factorization
    asup = -np.ones(N-1)
    ainf = -np.ones(N-1)
    ainf[N-2] = 0
    b = (2+ratio/theta1)*np.ones(N)
    b[N-1] = 1
    diagBlock11 = np.array([ainf,b,asup])
    diags = [-1, 0, 1]
    A = sp.diags(diagBlock11, diags, format ="csc")
    C = la.splu(A)
    # y is a (K+1) x (N+1) matrix
    y = np.zeros((K+1,N+1))
    # Initial conditions
    y[0,:] = y0
    y[1,:] = y0+DelT*y1
    y[1,1:-1] += (0.5/ratio)*(y0[0:-2]-2*y0[1:-1]+y0[2:])
    ###################################
    ###### Time marching method #######
    ###################################
    F = np.zeros(N)
    for k in range(0,K-1,1):
        F[0:N-1] = (ratio*(2*y[k+1,1:N]-y[k,1:N])
                    +theta2*(np.roll(y[k+1,:],-1)-2*y[k+1,:]+np.roll(y[k+1,:],1))[1:N]
                    +theta3*(np.roll(y[k  ,:],-1)-2*y[k  ,:]+np.roll(y[k  ,:],1))[1:N]
                    )/theta1
        if f != None:
            F[0:N-1] -= DelX**2*(f(y[k+1,1:N]))

        F[N-1] = boundaryData[k+2]
        y[k+2,1:] = C.solve(F)
    y[0,N] = boundaryData[0]
    y[1,N] = boundaryData[1]
    return y

def Explicit(y0,y1,boundaryData,f, L, T, N, K):
    if f == None:
        def f(x):
         return np.zeros(x.shape)

    #Spatial discretization step
    DelX = L/N
    # Time discretization step
    DelT = T/K
    ratio = (DelX/DelT)**2
    # y is a K+1 x N+1 matrix
    y = np.zeros((K+1,N+1))
    # Initial Conditions
    y[0,:] = y0
    y[1,:] = y0+DelT*y1
    y[1,1:-1] += (0.5/ratio)*(y0[0:-2]-2*y0[1:-1]+y0[2:])
    # Boundary conditions
    y[:,N] = boundaryData

    ###################################
    ###### Time marching method #######
    ###################################

    for k in range(2,K+1,1):
        Laplacian = (y[k-1,0:N-1]-2*y[k-1,1:N]+y[k-1,2:])/ratio
        y[k,1:N] = 2*y[k-1,1:N]-y[k-2,1:N]+Laplacian \
                   -f(y[k-1,1:N])*DelT**2
    return y


def ImplicitSystem(y0,y1,p0,p1,f,Df,T,N,K,b):
    if f == None:
        def f(x):
         return np.zeros(x.shape)
    if Df == None:
        def Df(x):
         return np.zeros(x.shape)

    L = 1.0
    # Spatial discretization step
    DelX = L/N
    # Time discretization step
    DelT = T/K
    ratio = (DelX/DelT)**2
    # Weights for the implicit scheme
    theta1 = 1./4.
    theta2 = 1./2.
    theta3 = 1-theta1-theta2

    alfa1 = 1/4
    alfa2 = 1/2
    alfa3 = 1/4
    C = Matrix(N,alfa1,theta1,DelT,DelX)
    ###########################################################
    ## Solution vector W = (y,q) is a (K+1) x 2(N+1) matrix  ##
    ###########################################################
    W = np.zeros((K+1,2*(N+1)))
    # Frequency
    m = 2
    # Initial conditions for y
    W[0,0:N+1] = y0
    W[1,0:N+1] = y0+DelT*y1
    # Initial conditions for q
    W[0,N+1:] = p0
    W[1,N+1:] = p0+DelT*p1

    ############################################
    ###### Implicit time marching method #######
    ############################################

    F = np.zeros(N+(N-1))

    for k in range(2,K+1,1):
        F[0:N-1] = (ratio*(2*W[k-1,1:N]-W[k-2,1:N])
                    +theta2*(W[k-1,0:N-1]-2*W[k-1,1:N]+W[k-1,2:N+1])
                    +theta3*(W[k-2,0:N-1]-2*W[k-2,1:N]+W[k-2,2:N+1]) \
                    -DelX**2*(f(W[k-1,1:N]))
                    )/theta1
        F[N-1] = 0
        F[N:]  = (ratio*(2*W[k-1,N+2:2*N+1]-W[k-2,N+2:2*N+1])
                    +theta2*(W[k-1,N+1:2*N]-2*W[k-1,N+2:2*N+1]+W[k-1,N+3:])
                    +theta3*(W[k-2,N+1:2*N]-2*W[k-2,N+2:2*N+1]+W[k-2,N+3:]) \
                    -DelX**2*Df(W[k-1,1:N])
                    +b*DelX**2*(alfa2*W[k-1,0:N-1]+alfa3*W[k-2,0:N-1])
                    )/theta1

        Sol = C.solve(F)
        W[k,1:N+1] = Sol[0:N]
        W[k,N+2:2*N+1] = Sol[N:]

    y = W[:,0:N+1]
    q = W[:,N+1:]
    return y, q

def ExplicitSystem(y0,y1,p0,p1,f,Df,T,N,K,b):
    if f == None:
        def f(x):
         return np.zeros(x.shape)
    if Df == None:
        def Df(x):
         return np.zeros(x.shape)

    L = 1.0

    # Spatial discretization step
    DelX = L/N
    # Time discretization step
    DelT = T/K
    ratio = (DelX/DelT)**2

    # y is a K+1 x N+1 matrix
    y = np.zeros((K+1,N+1))
    # q is a K+1 x N+1 matrix
    q = np.zeros((K+1,N+1))

    # Initial conditions
    y[0,:] = y0
    y[1,:] = y0+DelT*y1
    q[0,:] = p0
    q[1,:] = p0+DelT*p1

    # Time marching method

    for k in range(2,K+1,1):
        y[k-1,N] = 0.5*(-4*q[k-1,N-1]+q[k-1,N-2])/DelX
        Laplacian = (y[k-1,0:N-1]-2*y[k-1,1:N]+y[k-1,2:])/ratio
        y[k,1:N] = 2*y[k-1,1:N]-y[k-2,1:N]+Laplacian \
                   -f(y[k-1,1:N])*DelT**2

        Laplacian = (q[k-1,0:N-1]-2*q[k-1,1:N]+q[k-1,2:])/ratio
        q[k,1:N] = 2*q[k-1,1:N]-q[k-2,1:N]+Laplacian \
                   +b*(0.8*y[k-1,1:N]+0.2*y[k-2,1:N])*DelT**2 \
                   -Df(y[k-1,1:N])*DelT**2

    y[K,N] = 0.5*(-4*q[K,N-1]+q[K,N-2])/DelX
    return y, q

def toGrid(U,K,N):
    u = np.zeros((K+1,N+1,3))
    for k in range(K+1):
        for n in range(N+1):
            u[k,n,:] = U[3*(k*(N+1)+n):3*(k*(N+1)+n)+3]
    return u
