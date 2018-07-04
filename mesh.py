import numpy as np
import matplotlib.pyplot as plt


class Mesh(object):
    def __init__(self, N, K, T):
        self.NbPoints = (N+1)*(K+1)
        self.Nelem = 2*N*K
        self.N = N
        self.K = K
        self.DelT = T/K
        self.DelX = 1/N

        # Array of points in the mesh
        self.points = np.empty(((N+1)*(K+1),2))
        # Connectivity array
        self.connect = np.empty((self.Nelem,4),dtype=int)

        # Initialization of the points and connectivity arrays of the mesh
        self.initPoints()
        self.connectArray()

        self.basePoints = np.linspace(0,1,2*self.N+1,dtype=int)
        ## 'base' contains the indices of the 'points' in the base of the
        # rectagle
        self.base = np.arange(0,self.N+1,dtype=int)
        ## 'right' contains the indices of the 'points' in the right of the
        # rectagle
        self.right = self.N+(self.N+1)*np.arange(0,self.K+1,dtype=int)
        ## 'top' contains the indices of the 'points' in the top of the
        # rectagle
        self.top = (self.N+1)*(self.K+1)-1-np.arange(0,self.N+1,dtype=int)
        ## 'left' contains the indices of the 'points' in the top of the
        # rectagle
        self.left = (self.N+1)*(self.K-np.arange(0,self.K+1,dtype=int))


        ## 'bottomTriangles' contains the indices of the triangles
        # whose boundary intersects the bottom boundary.
        self.baseTriangles = np.arange(1,2*N,2,dtype=int)


    def initPoints(self):
        for k in range(self.K+1):
            for n in range(self.N+1):
                self.points[k*(self.N+1)+n] = np.array([n*self.DelX,k*self.DelT])

    def connectArray(self):
        for k in range(self.K):
            for n in range(self.N):
                # Indices of the points in the upper triangle (no segment on R boundary)
                self.connect[2*(k*self.N + n), 0] = k*(self.N+1)+n
                self.connect[2*(k*self.N + n), 1] = (k+1)*(self.N+1)+n+1
                self.connect[2*(k*self.N + n), 2] = (k+1)*(self.N+1)+n

                # Indices of the points in the lower triangle (no segment on L boundary)
                # The edge is on the T0 (k=0) subtriangle
                self.connect[2*(k*self.N + n)+1, 0] = k*(self.N+1)+n
                self.connect[2*(k*self.N + n)+1, 1] = k*(self.N+1)+n+1
                self.connect[2*(k*self.N + n)+1, 2] = (k+1)*(self.N+1)+n+1


                if k == 0:
                    # sets the label = '1', the base boundary
                    if n < self.N-1:
                        self.connect[2*(k*self.N+n)+1, 3] = 1
                    else:
                        # sets the label = '5' to the base-right corner
                        self.connect[2*(k*self.N+n)+1 , 3] = 5
                # sets the label = '2', the right-side boundary
                elif n == self.N-1:
                    self.connect[2*(k*self.N+n)+1, 3] = 2
                # sets the label = '3', the top-side boundary
                elif k == self.K-1:
                    self.connect[2*(k*self.N+n)  , 3] = 3
                # sets the label = '4', the left-side boundary
                elif n == 0:
                    self.connect[2*(k*self.N+n)  , 3] = 4
                # sets the label = '6' to the top-left corner
                elif n == 0 and k == self.K-1:
                    self.connect[2*(k*self.N+n) , 3] = 6
                else:
                    self.connect[2*(k*self.N+n)  , 3] = 0
                    self.connect[2*(k*self.N+n)+1, 3] = 0
