# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:22:22 2015

@author: jzzhao
"""

import numpy as np
import time


def atom_umat(norb,uj):

    # U_{abcd}f^\dag_{a}f^\dag_{b}f_{c}f_{d}
    umat = np.zeros((norb,norb,norb,norb),dtype=np.complex)

    for alpha in range(norb):
        for betta in range(norb):
            for delta in range(norb):
                for gamma in range(norb):
                    
                    abnd = alpha/2; aspn = alpha%2
                    bbnd = betta/2; bspn = betta%2
                    cbnd = delta/2; cspn = delta%2
                    dbnd = gamma/2; dspn = gamma%2
                    
                    # density-density interaction
                    if alpha == gamma and betta == delta :
                        # intraband coulomb interaction
                        if abnd == bbnd and aspn != bspn :
                            umat[alpha,betta,delta,gamma] += uj[0]
                        if abnd != bbnd :
                            # interband coulomb interaction
                            umat[alpha,betta,delta,gamma] += uj[1]
                            # Hund's exchange interaction
                            if aspn == bspn : 
                                umat[alpha,betta,delta,gamma] -= uj[2]

                    # spin-flip
                    if abnd == cbnd and bbnd == dbnd and abnd != bbnd and \
                       aspn != cspn and bspn != dspn and aspn != bspn :
                       umat[alpha,betta,delta,gamma] += uj[3]

                    # pair-hopping
                    if abnd == bbnd and cbnd == dbnd and abnd != cbnd and \
                       aspn != bspn and cspn != dspn and aspn != cspn :
                       umat[alpha,betta,delta,gamma] += uj[4]
                  
    dump_umat(umat)

    return umat

def atom_hmat(eimp, umat):

    norb = umat.shape[0]
    ncfg = 2**norb

    hmat = np.zeros((ncfg,ncfg),dtype=np.complex)

    # two  fermion terms
    for jcfg in range(ncfg):
        code = format(jcfg,'0'+str(norb)+'b')
        for alpha in range(norb):
            for betta in range(norb):

                sign = 0
                cnew = code

                if cnew[betta] == '1' :
                    for iorb in range(betta):
                        if cnew[iorb] == '1' : sign += 1
                    cnew = cnew[:betta]+'0'+cnew[betta+1:]

                    if cnew[alpha] == '0' :
                        for iorb in range(alpha):
                            if cnew[iorb] == '1' : sign += 1
                        cnew = cnew[:alpha]+'1'+cnew[alpha+1:]
                        icfg = int(cnew,2)
                        if np.absolute(eimp[alpha,betta])>1e-3 :
                            hmat[icfg,jcfg] += eimp[alpha,betta]*(-1.0)**float(sign)

    # four fermion terms
    indx = np.where( np.absolute(umat) > 1e-8 )
    ntot = len(indx[0])

    for jcfg in range(ncfg):
        code = format(jcfg,'0'+str(norb)+'b')

        for itot in range(ntot):
            alpha = indx[0][itot]; betta = indx[1][itot]
            delta = indx[2][itot]; gamma = indx[3][itot]

            sign  = 0
            cnew  = code
            
            if cnew[delta] == '1' and cnew[gamma] == '1' :
                for iorb in range(gamma):
                    if cnew[iorb] == '1' : sign += 1
                cnew = cnew[:gamma]+'0'+cnew[gamma+1:]
                for iorb in range(delta):
                    if cnew[iorb] == '1' : sign += 1
                cnew = cnew[:delta]+'0'+cnew[delta+1:]
                
                if cnew[alpha] == '0' and cnew[betta] == '0' :
                    for iorb in range(betta):
                        if cnew[iorb] == '1' : sign += 1
                    cnew = cnew[:betta]+'1'+cnew[betta+1:]
                    for iorb in range(alpha):
                        if cnew[iorb] == '1' : sign += 1
                    cnew = cnew[:alpha]+'1'+cnew[alpha+1:]

                    icfg = int(cnew, 2)
                    hmat[icfg, jcfg] += 0.5*umat[alpha,betta,delta,gamma]*(-1.0)**float(sign)

    dump_hmat(hmat)

    return hmat

def dump_umat(umat):

    indx = np.where( np.absolute(umat) > 1e-8 )
    ntot = len(indx[0])

    fout = open('atom.umat.out','w')
    for itot in range(ntot):
        alpha = indx[0][itot] ; betta = indx[1][itot]
        delta = indx[2][itot] ; gamma = indx[3][itot]
        print >> fout, ("%5d  %5d  %5d  %5d  %10.5f  %10.5f") \
                %(alpha+1,betta+1,delta+1,gamma+1, \
                umat[alpha,betta,delta,gamma].real,\
                umat[alpha,betta,delta,gamma].imag)
    fout.close()

def dump_hmat(hmat):

    indx = np.where( np.absolute(hmat) > 1e-8 )
    ntot = len(indx[0])

    fout = open('atom.hmat.out','w')
    for itot in range(ntot):
        icfg = indx[0][itot]; jcfg = indx[1][itot]
        print >> fout, ("  %5d  %5d  %10.5f  %10.5f") \
        %(icfg+1,jcfg+1,hmat[icfg,jcfg].real,hmat[icfg,jcfg].imag)
    fout.close()

def dump_basis(norb):

    ncfg = 2**norb

    fout = open('atom.basis.out','w')
    for icfg in range(ncfg):
        code = map(int,list(format(icfg,'0'+str(norb)+'b')))
        print >> fout, ("%5d") %(icfg),"  ", \
        "".join(("%3d") %(code[iorb]) for iorb in range(norb))
    fout.close()

def dump_eigs(eigs, evec):

    ntot = eigs.shape[0]
    norb = int(np.log2(ntot))

    fout = open('atom.eigs.out','w')
    for itot in range(ntot):
        print >> fout, ("  %5d%20.10f") %(itot+1, eigs[itot])
    fout.close()

    fout = open('atom.evec.out','w')
    for jtot in range(ntot):
        for itot in range(ntot):
            if np.absolute(evec[itot,jtot]) > 1e-8 :
                print >> fout, ("  %5d  %5d%20.10f%20.10f") \
                %(itot+1, jtot+1, evec[itot,jtot].real, evec[itot,jtot].imag), \
                ("     %"+str(norb)+"s") %(format(itot, "0"+str(norb)+"b"))
    fout.close()


if __name__ == "__main__" :
    
    nbnd = 2
    nspn = 2
    norb = nbnd * nspn
    ncfg = 2**norb

    dump_basis(norb)
    
    u1 = 1.0
    j1 = 0.1
    u2 = u1 - 2.0*j1
    j2 = j1
    j3 = j1

    uj = np.zeros(5,dtype=np.float)
    uj[0] = u1; uj[1] = u2
    uj[2] = j1; uj[3] = j2; uj[4] = j3

    eimp = np.zeros((norb,norb), dtype=np.complex)
    # eimp[0,0] =  1.0
    # eimp[1,1] =  1.0
    # eimp[2,2] =  2.0
    # eimp[3,3] =  2.0

    t1 = time.clock()
    umat = atom_umat(norb, uj)
    t2 = time.clock()
    print ' Time for atom_umat :', t2-t1
    
    t1 = time.clock()
    hmat = atom_hmat(eimp, umat)
    t2 = time.clock()
    print ' Time for atom_hmat :', t2-t1

    t1 = time.clock() 
    eigs, evec = np.linalg.eigh(hmat)
    t2 = time.clock()
    print ' Time for Eigenval  :', t2-t1

    dump_eigs(eigs, evec)
    
