import numpy as np
import QuadratureRules as qr


class HctMasterFunctions:
    
    def __init__(self, deg=[True,True,True,True]):
        if deg[0]:
            self.N = qr.gauss_1d.shape[0]
            self.gPhi01d = np.zeros((self.N,1,3))
            self.gPhi11d = np.zeros((self.N,1,3))
            self.gPhi21d = np.zeros((self.N,1,3))
            self.gbeta1d = np.zeros((self.N,1))
            for j, gauss in enumerate(qr.gauss_1d):
                x = gauss[0]
                y = gauss[1]
                self.gPhi01d[j] = self.Phi0(x,y)
                self.gPhi11d[j] = self.Phi1(x,y)
                self.gPhi21d[j] = self.Phi2(x,y)
                self.gbeta1d[j] = self.beta(x,y)
        if deg[1]:
            self.N = qr.gauss_2d.shape[0]
            self.gPhi02d = np.zeros((self.N,1,3))
            self.gPhi12d = np.zeros((self.N,1,3))
            self.gPhi22d = np.zeros((self.N,1,3))
            self.gbeta2d = np.zeros((self.N,1))
            for j, gauss in enumerate(qr.gauss_2d):
                x = gauss[0]
                y = gauss[1]
                self.gPhi02d[j] = self.Phi0(x,y)
                self.gPhi12d[j] = self.Phi1(x,y)
                self.gPhi22d[j] = self.Phi2(x,y)
                self.gbeta2d[j] = self.beta(x,y)

        if deg[2] :
            self.N = qr.gauss_1d.shape[0]
            self.gDPhi0 = np.zeros((self.N,2,3))
            self.gDPhi1 = np.zeros((self.N,2,3))
            self.gDPhi2 = np.zeros((self.N,2,3))
            self.gDbeta = np.zeros((self.N,2,1))
            for j, gauss in enumerate(qr.gauss_1d):
                x = 0.5*(1-gauss[0])
                y = 0.5*(1+gauss[0])
                self.gDPhi0[j] = self.DPhi0(x,y)
                self.gDPhi1[j] = self.DPhi1(x,y)
                self.gDPhi2[j] = self.DPhi2(x,y)
                self.gDbeta[j] = self.Dbeta(x,y)
        if deg[3] :
            self.N = qr.gauss_2d.shape[0]
            self.gD2Phi0 = np.zeros((self.N,3,3))
            self.gD2Phi1 = np.zeros((self.N,3,3))
            self.gD2Phi2 = np.zeros((self.N,3,3))
            self.gD2beta = np.zeros((self.N,3,1))
            for j, gauss in enumerate(qr.gauss_2d):
                x = gauss[0]
                y = gauss[1]
                self.gD2Phi0[j] = self.D2Phi0(x,y)
                self.gD2Phi1[j] = self.D2Phi1(x,y)
                self.gD2Phi2[j] = self.D2Phi2(x,y)
                self.gD2beta[j] = self.D2beta(x,y)
  
    # Master functions
    def Phi0(self, x, y):
        return ((1-x-y)**2)*np.array([[1+2*x+2*y,x,y]])

    def Phi1(self,x,y):
        return np.array([[x**2*(3-2*x),x**2*(x-1),y*x**2]])

    def Phi2(self,x,y):
        return np.array([[y**2*(3-2*y),y**2*x,(y-1)*y**2]])

    def beta(self,x,y):
        return x*y*(1-x-y)
    #######################################
    # First derivatives of master functions
    def DPhi0(self,x,y):
        return np.array([[6*(x+y)*(x+y-1), (x+y-1)*(3*x+y-1), 2*y*(x+y-1)],
                         [6*(x+y)*(x+y-1), 2*x*(x+y-1), (x+y-1)*(x+3*y-1)]])

    def DPhi1(self,x,y):
        return np.array([[6*x*(-x+1), x*(3*x-2), 2*x*y],
                        [         0,         0,  x**2]])

    def DPhi2(self,x,y):
        return np.array([[        0,  y**2,          0],
                        [6*y*(-y+1), 2*x*y, y*(3*y-2)]])

    def Dbeta(self,x,y):
        return np.array([y*(-2*x-y+1),x*(-x-2*y+1)]).reshape((2,1))

    ########################################
    # Second derivatives of master functions
    def D2Phi0(self,x,y):
        return np.array([[-6+12*x+12*y, -4+6*x+4*y , 2*y         ],
                         [-6+12*x+12*y,  2*x       , -4+4*x+6*y  ],
                         [-6+12*x+12*y, -2+4*x+2*y , -2+2*x+4*y  ]])

    def D2Phi1(self,x,y):
        return np.array([[6-12*x    , -2+6*x , 2*y    ],
                         [0         , 0      , 0      ],
                         [0         , 0      , 2*x    ]])

    def D2Phi2(self,x,y):
        return np.array([[0         , 0      , 0      ],
                         [6-12*y    , 2*x    ,-2+6*y ],
                         [0         , 2*y    , 0      ]])

    def D2beta(self,x,y):
        return np.array([-2*y,-2*x,1-2*(x+y)]).reshape((3,1))
