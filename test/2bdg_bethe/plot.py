
import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("zvu.dat")
u1 = data[:,0] 
z1 = data[:,1]
n1 = data[:,0].shape[0]
data = np.loadtxt("zvu_xydeng_bethe2bnd.txt")
u2 = data[:,0]
z2 = data[:,1]
n2 = data[:,0].shape[0]

# Set font
font = {'family' : 'Times New Roman' ,
            'weight' : 'normal' ,
            'size'   : '18'}
plt.rc('font',**font)

fig = plt.figure(figsize=(8.0,8.0))

plt.xlim(0, 5.2)
plt.ylim(0, 1.1)
plt.xlabel("$U/D$")
plt.ylabel("$Z$")
plt.plot(u2, z2, '-^', linewidth=2, markersize=12,color='b', label="XY Deng")
plt.plot(u1, z1, '-o', linewidth=2, markersize=12,color='r', label="This work")
plt.legend(prop={'size':16})

# plt.show()
plt.savefig("zvu.png", format="png", dpi=300)

