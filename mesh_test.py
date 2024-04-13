import numpy as np
import matplotlib.pyplot as plt
# Own packages
import mesh as mesh

N = 3
K = 6
T = 2

# Initialization of a mesh
Th = mesh.Mesh(N, K, T)

base   = Th.base_boundary_idx
right  = Th.right_boundary_idx
top    = Th.top_boundary_idx
left   = Th.left_boundary_idx

## Plot all the points in all the triangles using the connectivity array
formats = ['b.','g.','r.','y.','w.']
color = formats[0]

for triangle in Th.connectivity_array:
    if  triangle[3] == 2 or triangle[3] == 5:
        color = formats[0]
        for i in range(3):
            plt.plot(Th.vertices[triangle[i],0],Th.vertices[triangle[i],1],color)
plt.show()
plt.close()

for point in left:
    coor = Th.vertices[point,:]
    plt.plot(coor[0],coor[1],'b.')
plt.show()
plt.close()
