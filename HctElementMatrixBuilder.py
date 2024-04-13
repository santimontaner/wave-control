# Santiago Montaner, 2018

# References
# [1]
# Author: Arnd Meyer
# Title: A simplified calculation of reduced HCT-basis in FE context
# Journal: Computational Methods in Applied Mathematics
# [2]
# Author: Dunavant
# Title: "something on quadrature rules"

import numpy as np
import numpy.typing as npt
import mesh as mesh
import QuadratureRules as qr

# Input for the constructor:
# 'points' is a 3x2 matrix containing the coordinates of the three points
# defining an element

class HctElementMatrixBuilder:
    # Constructor
    def __init__(self, vertices: npt.ArrayLike, ev):
        # Evaluation of the master functions at gaussian points
        self.ev = ev
        self._vertices = vertices
        self._mesh_barycenter = np.mean(vertices, 0)
        # Exterior edges
        self._ext_edges = np.empty((3,2))
        self._ext_edges[0,:] = self._vertices[2,:] - self._vertices[1,:]
        self._ext_edges[1,:] = self._vertices[0,:] - self._vertices[2,:]
        self._ext_edges[2,:] = -(self._ext_edges[0,:] + self._ext_edges[1,:])

        self._init_interior_edges()

        self._jacobian_mat = np.column_stack((self._ext_edges[2],-self._ext_edges[1]))       
        self._rotation_mat = np.array([[0,-1],[1,0]])        
        self._outward_normal = np.matmul(self._ext_edges, np.transpose(self._rotation_mat))
        self._mu = np.abs(np.linalg.det(self._jacobian_mat))/3

        # Some arrays
        self.b = np.empty((3,3,3,1))
        self.M = np.empty((3,3,3))

        self._initialize()

    @staticmethod
    def _rotate_index(i):
        next = (i+1) % 3
        prev = (i+2) % 3
        return next, prev

    def _init_interior_edges(self):
        self.f = np.empty((3,2))
        for j in range(3):
            jp, jm = HctElementMatrixBuilder._rotate_index(j)
            self.f[j,:] = (self._ext_edges[jp,:] - self._ext_edges[jm,:])/3

    def _initialize(self):
        t = np.empty((3,3))
        T = np.empty((3,3,3))
        S = np.empty((3,3))
        for k in range(3):
            S[k,0] = 3
            S[k,1:] = self.f[k,:]
        S = 6*S

        for k in range(3):
            kp, km = HctElementMatrixBuilder._rotate_index(k)
            normE = np.dot(self._ext_edges[k,:],self._ext_edges[k,:])
            self.b[k,kp,0,0] =  6*np.dot(self._ext_edges[k,:],self.f[km,:])/normE
            self.b[k,kp,1:,0] = 2*self.f[km,:]+(3*self._mu/normE)*self._outward_normal[k,:]

            self.b[k,km,0,0] = -6*np.dot(self._ext_edges[k,:],self.f[kp,:])/normE
            self.b[k,km,1:,0] = 2*self.f[kp,:]+(3*self._mu/normE)*self._outward_normal[k,:]

        for j in range(3):
            jp, jm = HctElementMatrixBuilder._rotate_index(j)
            t[jm,:] = self.b[jp,j,:,0]
            t[jp,:] = self.b[jm,j,:,0]
            t[j,:]  = (t[jm,:]+t[jp,:])
            t[j,0]  += 6
            t[j,1:] += -2*self.f[j,:]
            for k in range(3):
                T[j,k,:] = t[k,:]
            self.M[j] = np.matmul(np.linalg.inv(S),T[j])

    def build_interior(self) -> np.ndarray:
        number_of_nodes = qr.gauss_2d.shape[0]
        quad_weights = np.empty((number_of_nodes, 1))        
        quad_weights[:,0] = qr.gauss_2d[:,2]
        
        K = np.zeros((9,9))
        H = np.zeros((3,3))
        
        for k in range(3):
            kp, km = HctElementMatrixBuilder._rotate_index(k)
            J = np.column_stack((self.f[kp,:], self.f[km,:]))
            H[0,0] = 1
            H[1:,1:] = np.transpose(J)
            # a= tau1, b= tau2, c= tau3, d= tau4 in notation of reference [1]
            a = J[0,0]
            b = J[0,1]
            c = J[1,0]
            d = J[1,1]
            ## G matrix for the wave control problem
            G = np.array([b**2-d**2,   a**2-c**2,  2*(c*d-a*b)])/(self._mu**2)
            DD  = np.empty((number_of_nodes,3,9))
            DD1 = np.matmul(self.ev.gD2Phi0,np.matmul(H,self.M[k]))
            DD2 = (
                  np.matmul(self.ev.gD2Phi1, H)
                + np.matmul(self.ev.gD2Phi0, np.matmul(H,self.M[kp]))
                + np.matmul(self.ev.gD2beta, np.transpose(self.b[k,kp]))
            )
            DD3 = (
                  np.matmul(self.ev.gD2Phi2, H)
                + np.matmul(self.ev.gD2Phi0, np.matmul(H,self.M[km]))
                + np.matmul(self.ev.gD2beta, np.transpose(self.b[k,km]))
            )
            DD[:,:,3*k:3*k+3  ] = DD1
            DD[:,:,3*kp:3*kp+3] = DD2
            DD[:,:,3*km:3*km+3] = DD3
            DD                  = np.matmul(G,DD)
            w = self._mu*0.5            
            wD = np.multiply(quad_weights,DD)
            K += w*np.tensordot(wD,DD,axes =(0,0))
        return K
    
    def build_boundary(self, edge_tri_idx):
        """
        # `edge_tri_idx` is the subtriangle where the edge is located
        """
        N = qr.gauss_1d.shape[0]
        quad_weights = np.zeros((N,1))        
        quad_weights[:,0] = qr.gauss_1d[:,1]
        
        C = np.array([1,0])
        kp, km = HctElementMatrixBuilder._rotate_index(edge_tri_idx)
        J = np.column_stack((self.f[kp,:], self.f[km,:]))
        H = np.zeros((3,3))
        H[0,0] = 1
        H[1:,1:] = np.transpose(J)
        G = np.transpose(np.linalg.inv(J))
        D  = np.empty((N,2,9))
        D[:,:,3*edge_tri_idx:3*edge_tri_idx+3  ] = np.matmul(self.ev.gDPhi0, np.matmul(H, self.M[edge_tri_idx]))
        D[:,:,3*kp:3*kp+3] = (
              np.matmul(self.ev.gDPhi1, H)                           
            + np.matmul(self.ev.gDPhi0, np.matmul(H, self.M[kp]))
            + np.matmul(self.ev.gDbeta, np.transpose(self.b[edge_tri_idx,kp]))
        )
        D[:,:,3*km:3*km+3] = (
              np.matmul(self.ev.gDPhi2, H)
            + np.matmul(self.ev.gDPhi0, np.matmul(H, self.M[km]))
            + np.matmul(self.ev.gDbeta, np.transpose(self.b[edge_tri_idx,km]))
        )

        D = np.matmul(G,D)
        D = np.matmul(C,D)
        
        w = np.sqrt(np.dot(self._ext_edges[edge_tri_idx,:],self._ext_edges[edge_tri_idx,:]))*0.5        
        wD = np.multiply(quad_weights,D)
        return w*np.tensordot(wD,D,axes =(0,0))

    def build_init_pos(self,k):
        L = np.zeros((9,3))
        C = np.array([[0,1]])
        kp, km = HctElementMatrixBuilder._rotate_index(k)
        J = np.column_stack((self.f[kp,:],self.f[km,:]))
        H = np.zeros((3,3))
        H[0,0] = 1
        H[1:,1:] = np.transpose(J)
        G = np.transpose(np.linalg.inv(J))
        
        quad_weights = qr.gauss_1d
        quad_weights = (quad_weights + 1) * 0.5

        for j, gauss in enumerate(quad_weights):
            x, y = gauss[0], 1 - gauss[0]            
            D  = np.zeros((2,9))
            D0  = np.zeros((1,3))
            D[:, 3*k:3*k+3  ] = np.matmul(self.ev.gDPhi0[j],np.matmul(H,self.M[k]))
            D[:, 3*kp:3*kp+3] = (
                np.matmul(self.ev.gDPhi1[j],H)
                + np.matmul(self.ev.gDPhi0[j],np.matmul(H,self.M[kp]))
                + np.matmul(self.ev.gDbeta[j],np.transpose(self.b[k,kp]))
            )
            D[:, 3*km:3*km+3] = (
                np.matmul(self.ev.gDPhi2[j],H)
                + np.matmul(self.ev.gDPhi0[j], np.matmul(H,self.M[km]))
                + np.matmul(self.ev.gDbeta[j],np.transpose(self.b[k,km]))
            )

            D0[0,:] = [(1-x-y), (1+2*x-y), (1-x+2*y)]
            D0 = D0*(1./3.)
            D2 = np.matmul(G,D)
            w = 0.5*np.sqrt(np.dot(self._ext_edges[k,:],self._ext_edges[k,:]))*gauss[1]
            L += w*np.matmul(np.transpose(np.matmul(C,D2)),D0)
        return L

    def build_init_vel(self,k):
        L = np.zeros((9,3))
        kp, km = HctElementMatrixBuilder._rotate_index(k)
        J = np.column_stack((self.f[kp,:],self.f[km,:]))
        H = np.zeros((3,3))
        H[0,0] = 1
        H[1:,1:] = np.transpose(J)
        quad_weights = qr.gauss_1d
        quad_weights = (quad_weights + 1)*0.5
        for j, gauss in enumerate(quad_weights):
            x, y = gauss[0], 1-gauss[1]
            
            D0  = np.zeros((1,9))
            DP1 = np.zeros((1,3))
            D0[:,3*k:3*k+3  ] = np.matmul(self.ev.gPhi01d[j],np.matmul(H,self.M[k]))
            D0[:,3*kp:3*kp+3] = (
                  np.matmul(self.ev.gPhi11d[j],H)
                + np.matmul(self.ev.gPhi01d[j],np.matmul(H,self.M[kp]))
                + np.matmul(self.ev.gbeta1d[j],np.transpose(self.b[k,kp]))
            )
            D0[:,3*km:3*km+3] = (
                  np.matmul(self.ev.gPhi21d[j],H)
                + np.matmul(self.ev.gPhi01d[j],np.matmul(H,self.M[km]))
                + np.matmul(self.ev.gbeta1d[j],np.transpose(self.b[k,km]))
            )
            DP1[0,:] = [(1-x-y), (1+2*x-y),(1-x+2*y)]
            DP1 = DP1*(1./3.)

            w = 0.5*np.sqrt(np.dot(self._ext_edges[k,:],self._ext_edges[k,:]))*gauss[1]
            L += w*np.matmul(np.transpose(D0),DP1)
        return L
