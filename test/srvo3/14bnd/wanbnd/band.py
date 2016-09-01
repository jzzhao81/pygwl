#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

basename = "wanbnd"

# Define X labels
tics = [0.0, 0.22538, 0.35550, 0.48563, 0.66965]
labl = ['${\\mathrm{R}}$','${\\Gamma}$','${\\mathrm{X}}$','${\\mathrm{M}}$','${\\Gamma}$']

# Set font
font = {'family' : 'Times New Roman' ,
			'weight' : 'normal' ,
			'size'   : '18'}
plt.rc('font',**font)

# Load data rho.dat
data = np.loadtxt(basename+'.dat')
kpts = np.array( map(float,OrderedDict.fromkeys(data[:,0])), dtype=np.float )
nkpt = kpts.shape[0]
eigs = data[:,1]
nbnd = eigs.shape[0] / nkpt
eigs = eigs.reshape(nbnd,nkpt)
prob = data[:,2].reshape(nbnd,nkpt)

data = np.loadtxt('vaspband.dat')
ngkp = 200
ngbd = 96
gkpt = data[:ngkp,0]
geig = data[:,1].reshape(ngbd, ngkp) - 0.80


# define fig parameters
majorLocator   = MultipleLocator(2.00)
minorLocator   = MultipleLocator(1.00)

fig, ax = plt.subplots(figsize=(6.0,6.0))
ax.set_xlim(kpts.min(), kpts.max())
ax.set_ylim(-8.0, 6.0)
plt.xticks(tics, labl)
ax.yaxis.set_major_locator(majorLocator)
ax.yaxis.set_minor_locator(minorLocator)

# Draw figure
for ibnd in range(ngbd):
    plt.plot(gkpt, geig[ibnd,:], 'ro', color='blue', ms=4.0)

# TB
for ibnd in range(nbnd):
    plt.plot(kpts, eigs[ibnd,:], '-', color='red', linewidth=1.5)

# Draw fermi level
plt.text(kpts.max()+0.004, -0.05, r'$E_F$')
plt.plot([kpts.min(),kpts.max()], [0.0,0.0], color='black', linestyle='--',linewidth=1)

# save figure
# plt.show()
plt.savefig(basename+".png", dpi=300, format="png")

# job done
print "Job Done !"
