import numpy as np
import matplotlib.pyplot as plt
# Own modules
import WaveTimeMarching as wtm
import Plots

# Geometric data
T = 2
L = 1
N = 400
K = 2*N

# Nonlinearities
f = None
Df = None

x = np.linspace(0,L,N+1)
t = np.linspace(0,T,K+1)

# Initial Data
m = 5
u0 = np.sin(np.pi*m*x)
u1 = np.zeros(N+1)
boundaryData = 0.5*(-1)**(m+1)*np.sin(m*np.pi*t)

# # Implicit Method
# u = wtm.Implicit(u0,u1,boundaryData,f, L, T, N, K)
# # Plot Final data
# plt.plot(x,u[K,:])
# plt.show()
# plt.close()
# Plots.SpaceTime(u, L, T, N, K)

# Explicit Method
u = wtm.Explicit(u0,u1,boundaryData,f, L, T, N, K)
# Plot Final data
plt.plot(x,u[K,:])
plt.show()
plt.close()
Plots.SpaceTime(u, L, T, N, K)

# # Coupled Optimality System (Implicit method)
# # Coupling parameter
# b = 0
# # Initial data for the adjoint system
# p0 = np.zeros(N+1)
# p1 = -0.5*np.sin(m*np.pi*x)
#
# u, p = wtm.ImplicitSystem(u0,u1,p0,p1,f,Df,T,N,K,b)
# # Plot Final data
# plt.plot(x,u[K,:])
# plt.show()
# plt.close()
# Plots.SpaceTime(u, L, T, N, K)
# Plots.SpaceTime(p, L, T, N, K)


# Coupled Optimality System (Explicit method)
# Coupling parameter
b = 0
# Initial data for the adjoint system
DelT = T/K
p0 = np.zeros(N+1)
p1 = -0.5*np.sin(m*np.pi*x)

u, p = wtm.ExplicitSystem(u0,u1,p0,p1,f,Df,T,N,K,b)
# Plot Final data
plt.plot(x,u[K,:])
plt.show()
plt.close()
# Space-time grid plot
Plots.SpaceTime(u, L, T, N, K)
#Plots.SpaceTime(p, L, T, N, K)
