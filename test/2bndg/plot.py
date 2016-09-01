
import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("ovu02.dat")
breaks = [i for i in range(len(data)) if i > 0 and data[i, 0] < data[i-1, 0]]
borders = [0] + breaks + [len(data)]
subplots1 = [data[borders[i]:borders[i+1]] for i in range(len(borders)-1)]
data = np.loadtxt("ovu_nicola_cubic2bndg.txt")
breaks = [i for i in range(len(data)) if i > 0 and data[i, 0] < data[i-1, 0]]
borders = [0] + breaks + [len(data)]
subplots2 = [data[borders[i]:borders[i+1]] for i in range(len(borders)-1)]
data = np.loadtxt("ovu04.dat")
breaks = [i for i in range(len(data)) if i > 0 and data[i, 0] < data[i-1, 0]]
borders = [0] + breaks + [len(data)]
subplots3 = [data[borders[i]:borders[i+1]] for i in range(len(borders)-1)]

# Set font
font = {'family' : 'Times New Roman' ,
            'weight' : 'normal' ,
            'size'   : '18'}
plt.rc('font',**font)

fig = plt.figure(figsize=(8.0,8.0))

plt.xlim(0, 2.0)
plt.ylim(0.4,1.0)
plt.xlabel("$U (eV)$")
plt.ylabel("$n_{1\sigma}$")

# Sigma = 0.2
t,E = subplots1[0][:,0],subplots1[0][:,2]
plt.plot(t,E, '-^', linewidth=2, markersize=8,color='r',label='$\sigma=0.2,J=0$')
t,E = subplots1[1][:,0],subplots1[1][:,2]
plt.plot(t,E, '-^', linewidth=2, markersize=8,color='b',label='$\sigma=0.2,J=0.05U$')
t,E = subplots1[2][:,0],subplots1[2][:,2]
plt.plot(t,E, '-^', linewidth=2, markersize=8,color='g',label='$\sigma=0.2,J=0.1U$')
t,E = subplots1[3][:,0],subplots1[3][:,2]
plt.plot(t,E, '-^', linewidth=2, markersize=8,color='c',label='$\sigma=0.2,J=0.25U$')

# Nicola
# t,E = subplots2[0][:,0],subplots2[0][:,1]
# plt.plot(t, E, '-o', linewidth=2, markersize=8,color='r',label='Nicola,U=0')
# t,E = subplots2[1][:,0],subplots2[1][:,1]
# plt.plot(t, E, '-o', linewidth=2, markersize=8,color='b',label='Nicola,U=0.05U')
# t,E = subplots2[2][:,0],subplots2[2][:,1]
# plt.plot(t, E, '-o', linewidth=2, markersize=8,color='g',label='Nicola,U=0.1U')
# t,E = subplots2[3][:,0],subplots2[3][:,1]
# plt.plot(t, E, '-o', linewidth=2, markersize=8,color='c',label='Nicola,U=0.25U')

# Sigma = 0.4
t,E = subplots3[0][:,0],subplots3[0][:,2]
plt.plot(t, E, '-s', linewidth=2, markersize=8,color='r',label='$\sigma=0.4,J=0$')
t,E = subplots3[1][:,0],subplots3[1][:,2]
plt.plot(t, E, '-s', linewidth=2, markersize=8,color='b',label='$\sigma=0.4,J=0.05U$')
t,E = subplots3[2][:,0],subplots3[2][:,2]
plt.plot(t, E, '-s', linewidth=2, markersize=8,color='g',label='$\sigma=0.4,J=0.1U$')
t,E = subplots3[3][:,0],subplots3[3][:,2]
plt.plot(t, E, '-s', linewidth=2, markersize=8,color='c',label='$\sigma=0.4,J=0.25U$')
plt.legend(prop={'size':12})

# plt.show()
plt.savefig("zvu.png", format="png", dpi=300)

