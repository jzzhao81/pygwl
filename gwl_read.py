# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:22:22 2015

@author: jzzhao
"""

import numpy as np
from   gwl_atom import corr_atom

def read_dft(filename_enk, filename_ovlp):

    finp = open(filename_enk,'r')
    nkpt = int(finp.readline().split()[0])
    nbnd = int(finp.readline().split()[0])
    norb = int(finp.readline().split()[0])
    natm = int(finp.readline().split()[0])
    ntot = float(finp.readline().split()[0])
    finp.close()

    data = np.loadtxt(filename_enk, skiprows=5, comments="#", usecols=[1])
    eigs = data.reshape(nkpt,nbnd)

    data = np.loadtxt(filename_ovlp, skiprows=5, comments="#", usecols=[2,3])
    ovlp = (data[:,0] + 1j*data[:,1]).reshape(nkpt,norb,nbnd)

    gatm = corr_atom(nkpt,nbnd,norb,ntot)
    gatm.eigs = eigs
    gatm.smat = ovlp

    return gatm

def gen_bethe_kwt(eigs):
    nkpt = eigs.shape[0]
    norb = eigs.shape[1]
    kwt  = np.zeros(nkpt,dtype=np.float)
    wid  = np.absolute( eigs[0,0] - 0.0 )
    for ikpt in range(nkpt):
        kpt = 2.0*np.float(ikpt)/np.float(nkpt-1) - 1.0
        kwt[ikpt] = 4.0*np.sqrt(1.0-kpt**2.0)/np.pi/np.float(nkpt)
    return kwt

