import numpy as np
import matplotlib.pyplot as plt
# Own packages
import mesh as mesh

N = 3
K = 6
T = 2

# Initialization of a mesh
Th = mesh.Mesh(N, K, T)

base   = Th.base
right  = Th.right
top    = Th.top
left   = Th.left

## Plot all the points in all the triangles using the connectivity array
formats = ['b.','g.','r.','y.','w.']
color = formats[0]

for triangle in Th.connect:
    if  triangle[3] == 2 or triangle[3] == 5:
        color = formats[0]
        for i in range(3):
            plt.plot(Th.points[triangle[i],0],Th.points[triangle[i],1],color)
plt.show()
plt.close()

for point in left:
    coor = Th.points[point,:]
    plt.plot(coor[0],coor[1],'b.')
plt.show()
plt.close()
