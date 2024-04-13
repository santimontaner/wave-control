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
import mesh as mesh
import QuadratureRules as qr
#import weight as carleman

# Input for the constructor:
# 'points' is a 3x2 matrix containing the coordinates of the three points
# defining an element

class rHCT_FE(object):
    # Constructor
    def __init__(self, points, ev):
        # Evaluation of the master functions at gaussian points
        self.ev = ev
        # Vertices
        self.points = points
        # Barycenter
        self.a0 = np.mean(points,0)
        # Exterior edges
        self.E = np.empty((3,2))
        self.E[0,:] = self.points[2,:]-self.points[1,:]
        self.E[1,:] = self.points[0,:]-self.points[2,:]
        self.E[2,:] = -(self.E[0,:]+self.E[1,:])

        # Interior edges
        self.f = np.empty((3,2))
        self.interiorEdges()

        # Jacobian matrix
        self.J = np.column_stack((self.E[2],-self.E[1]))

        # Rotation matrix
        self.R = np.array([[0,-1],[1,0]])

        # Unit outward normals
        self.N = np.matmul(self.E,np.transpose(self.R))

        # mu = |det J|/3
        self.mu = np.abs(np.linalg.det(self.J))/3

        # Some arrays
        self.b = np.empty((3,3,3,1))
        self.M = np.empty((3,3,3))

        # Perform preparatory steps
        self.preparatory()

    # Class Methods
    def rotateIndex(self,i):
        ip = (i+1) % 3
        im = (i+2) % 3
        return ip, im

    def interiorEdges(self):
        for j in range(3):
            jp, jm = self.rotateIndex(j)
            self.f[j,:] = (self.E[jp,:] - self.E[jm,:])/3

    def preparatory(self):
        t = np.empty((3,3))
        T = np.empty((3,3,3))
        S = np.empty((3,3))
        for k in range(3):
            S[k,0] = 3
            S[k,1:] = self.f[k,:]
        S = 6*S

        for k in range(3):
            kp, km = self.rotateIndex(k)
            normE = np.dot(self.E[k,:],self.E[k,:])
            self.b[k,kp,0,0] =  6*np.dot(self.E[k,:],self.f[km,:])/normE
            self.b[k,kp,1:,0] = 2*self.f[km,:]+(3*self.mu/normE)*self.N[k,:]

            self.b[k,km,0,0] = -6*np.dot(self.E[k,:],self.f[kp,:])/normE
            self.b[k,km,1:,0] = 2*self.f[kp,:]+(3*self.mu/normE)*self.N[k,:]

        for j in range(3):
            jp, jm = self.rotateIndex(j)
            t[jm,:] = self.b[jp,j,:,0]
            t[jp,:] = self.b[jm,j,:,0]
            t[j,:]  = (t[jm,:]+t[jp,:])
            t[j,0]  += 6
            t[j,1:] += -2*self.f[j,:]
            for k in range(3):
                T[j,k,:] = t[k,:]
            self.M[j] = np.matmul(np.linalg.inv(S),T[j])


    def InteriorStiffness(self):
        N = qr.gaussRule2D.shape[0]
        gaussW = np.empty((N,1))
        #gausscoor = qr.gaussRule[:,0:2]
        gaussW[:,0] = qr.gaussRule2D[:,2]
        K = np.zeros((9,9))
        H = np.zeros((3,3))
        for k in range(3):
            kp, km = self.rotateIndex(k)
            J = np.column_stack((self.f[kp,:],self.f[km,:]))
            I = np.linalg.inv(J)
            H[0,0] = 1
            H[1:,1:] = np.transpose(J)
            # a= tau1, b= tau2, c= tau3, d= tau4 in notation of reference [1]
            a = J[0,0]
            b = J[0,1]
            c = J[1,0]
            d = J[1,1]
            ## G matrix for the wave control problem
            G = np.array([b**2-d**2,   a**2-c**2,  2*(c*d-a*b)])/(self.mu**2)
            DD  = np.empty((N,3,9))
            DD1 = np.matmul(self.ev.gD2Phi0,np.matmul(H,self.M[k]))
            DD2 = np.matmul(self.ev.gD2Phi1,H)                                       \
                            + np.matmul(self.ev.gD2Phi0,np.matmul(H,self.M[kp]))     \
                            + np.matmul(self.ev.gD2beta,np.transpose(self.b[k,kp]))
            DD3 = np.matmul(self.ev.gD2Phi2,H)                                       \
                            + np.matmul(self.ev.gD2Phi0,np.matmul(H,self.M[km]))     \
                            + np.matmul(self.ev.gD2beta,np.transpose(self.b[k,km]))
            DD[:,:,3*k:3*k+3  ] = DD1
            DD[:,:,3*kp:3*kp+3] = DD2
            DD[:,:,3*km:3*km+3] = DD3
            DD                  = np.matmul(G,DD)
            #x = np.matmul(gausscoor,np.transpose(J))+self.a0
            w = self.mu*0.5
            #gaussW = np.multiply(gaussW,carleman.invweight2(x))
            wD = np.multiply(gaussW,DD)
            K += w*np.tensordot(wD,DD,axes =(0,0))
        return K

    # k is the subtriangle where the edge is located
    def BoundaryStiffness(self,k):
        N = qr.gaussRule1D.shape[0]
        gaussW = np.zeros((N,1))
        gausscoor = np.reshape(qr.gaussRule1D[:,0],(gaussW.shape[0],1))
        x = np.zeros((gaussW.shape[0],2))
        gaussW[:,0] = qr.gaussRule1D[:,1]
        C = np.array([1,0])
        kp, km = self.rotateIndex(k)
        J = np.column_stack((self.f[kp,:],self.f[km,:]))
        H = np.zeros((3,3))
        H[0,0] = 1
        H[1:,1:] = np.transpose(J)
        G = np.transpose(np.linalg.inv(J))
        D  = np.empty((N,2,9))
        D[:,:,3*k:3*k+3  ] = np.matmul(self.ev.gDPhi0,np.matmul(H,self.M[k]))
        D[:,:,3*kp:3*kp+3] = np.matmul(self.ev.gDPhi1,H)                           \
                           + np.matmul(self.ev.gDPhi0,np.matmul(H,self.M[kp]))   \
                           + np.matmul(self.ev.gDbeta,np.transpose(self.b[k,kp]))
        D[:,:,3*km:3*km+3] = np.matmul(self.ev.gDPhi2,H)                           \
                           + np.matmul(self.ev.gDPhi0,np.matmul(H,self.M[km]))   \
                           + np.matmul(self.ev.gDbeta,np.transpose(self.b[k,km]))

        D = np.matmul(G,D)
        D = np.matmul(C,D)
        # x[:,0] = 0.5*(1-gausscoor[:,0])
        # x[:,1] = 0.5*(1+gausscoor[:,0])
        # x = np.matmul(x,np.transpose(J))+self.a0
        w = np.sqrt(np.dot(self.E[k,:],self.E[k,:]))*0.5
        #gaussW = np.multiply(gaussW,carleman.invweight2(x))
        wD = np.multiply(gaussW,D)
        return w*np.tensordot(wD,D,axes =(0,0))

    def InitPositionMatrix(self,k):
        L = np.zeros((9,3))
        C = np.array([[0,1]])
        kp, km = self.rotateIndex(k)
        J = np.column_stack((self.f[kp,:],self.f[km,:]))
        H = np.zeros((3,3))
        H[0,0] = 1
        H[1:,1:] = np.transpose(J)
        G = np.transpose(np.linalg.inv(J))
        Phi = np.empty((1,3))
        gaussRule = qr.gaussRule1D
        gaussRule = (gaussRule + 1)*0.5
        for j, gauss in enumerate(gaussRule):
            x = gauss[0]
            y = 1-gauss[0]
            D  = np.zeros((2,9))
            D0  = np.zeros((1,3))
            D[:,3*k:3*k+3  ] = np.matmul(self.ev.gDPhi0[j],np.matmul(H,self.M[k]))
            D[:,3*kp:3*kp+3] = np.matmul(self.ev.gDPhi1[j],H)                           \
                               + np.matmul(self.ev.gDPhi0[j],np.matmul(H,self.M[kp]))   \
                               + np.matmul(self.ev.gDbeta[j],np.transpose(self.b[k,kp]))

            D[:,3*km:3*km+3] = np.matmul(self.ev.gDPhi2[j],H)                           \
                               + np.matmul(self.ev.gDPhi0[j],np.matmul(H,self.M[km]))   \
                               + np.matmul(self.ev.gDbeta[j],np.transpose(self.b[k,km]))

            D0[0,:] = [(1-x-y), (1+2*x-y),(1-x+2*y)]
            D0 = D0*(1./3.)
            D2 = np.matmul(G,D)
            w = 0.5*np.sqrt(np.dot(self.E[k,:],self.E[k,:]))*gauss[1]
            L = L + w*np.matmul(np.transpose(np.matmul(C,D2)),D0)
        return L

    def InitVelocityMatrix(self,k):
        L = np.zeros((9,3))
        kp, km = self.rotateIndex(k)
        J = np.column_stack((self.f[kp,:],self.f[km,:]))
        H = np.zeros((3,3))
        H[0,0] = 1
        H[1:,1:] = np.transpose(J)
        gaussRule = qr.gaussRule1D
        gaussRule = (gaussRule + 1)*0.5
        for j, gauss in enumerate(gaussRule):
            x = gauss[0]
            y = 1-gauss[0]
            D0  = np.zeros((1,9))
            DP1 = np.zeros((1,3))
            D0[:,3*k:3*k+3  ] = np.matmul(self.ev.gPhi01d[j],np.matmul(H,self.M[k]))
            D0[:,3*kp:3*kp+3] = np.matmul(self.ev.gPhi11d[j],H)                           \
                               + np.matmul(self.ev.gPhi01d[j],np.matmul(H,self.M[kp]))   \
                               + np.matmul(self.ev.gbeta1d[j],np.transpose(self.b[k,kp]))
            D0[:,3*km:3*km+3] = np.matmul(self.ev.gPhi21d[j],H)                           \
                               + np.matmul(self.ev.gPhi01d[j],np.matmul(H,self.M[km]))   \
                               + np.matmul(self.ev.gbeta1d[j],np.transpose(self.b[k,km]))

            DP1[0,:] = [(1-x-y), (1+2*x-y),(1-x+2*y)]
            DP1 = DP1*(1./3.)

            w = 0.5*np.sqrt(np.dot(self.E[k,:],self.E[k,:]))*gauss[1]
            L = L + w*np.matmul(np.transpose(D0),DP1)
        return L
