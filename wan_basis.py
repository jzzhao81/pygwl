# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 16:25:43 2014

@author: jzzhao

Basis definition :
================================================================
%%real basis ( general set )%%
----------------------------------------------------------------
l = 0 : s
l = 1 : pz, px, py
l = 2 : dz2, dxz, dyz, dx2-y2, dxy
l = 3 : fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
----------------------------------------------------------------
%%real basis ( cubic set )%%
----------------------------------------------------------------
ref : PRB,79,045107(2009)
l = 3 : |f1,±>,|f2,±>,|f3,±>,|f4,±>,|f5,±>,|f6,±>,|f7,±>
----------------------------------------------------------------
%%complex basis%%
----------------------------------------------------------------
|-3,±>, |-2,±>, |-1,±>, |0,±>, |+1,±>, |+2,±>, |+3,±>"
----------------------------------------------------------------
%%|J2, Jz>%%
----------------------------------------------------------------
|3/2,-3/2>,|3/2,+3/2>,|3/2,-1/2>,|3/2,+1/2>
|5/2,-5/2>,|5/2,+5/2>,|5/2,-3/2>,|5/2,+3/2>,|5/2,-1/2>,|5/2,+1/2>
================================================================
"""

import numpy as np

def clebsch_gordan(ll):

    # print 
    # print " C-G transform Definition :"
    # print " U = <jj, jz | lz, sz>"
    # print 
    
    nm = 2*ll+1
    ns = 2
    ss = np.float(ns-1)/2.0
    norb = nm * ns
    cgmat = np.zeros((norb,norb),dtype=np.complex)

    ndeg = np.int(2*(ll+ss)+1.0)
    mdeg = np.int(2*(ll-ss)+1.0)

    for iorb in range(norb):

        if iorb < mdeg :
            jj  = ll - ss
            iss = iorb%2
            jz  = (-1.0)**(iss+1) * (jj-iorb/2)
        else :
            jj  = ll + ss
            iss = iorb%2
            jz  = (-1.0)**(iss+1) * (jj-(iorb-mdeg)/2)

        # print "jj, jz :", jj, jz

        for jorb in range(norb):

            lz = -ll + (jorb/2)
            iss = jorb%2
            sz = (-1.0)**iss/2.0

            if np.absolute(np.float(lz)+sz-jz) < 1e-3 :
                cgmat[iorb,jorb] = clebsch(ll, ss, jj, lz, sz, jz)

    return cgmat

# https://qutip.googlecode.com/svn/qutip/qutip/clebsch.py
def clebsch(j1,j2,j3,m1,m2,m3):


    try:#for scipy v <= 0.90
        from scipy import factorial
    except:#for scipy v >= 0.10
        from scipy.misc import factorial

    """Calculates the Clebsch-Gordon coefficient 
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).
    
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
        
    j2 : float
        Total angular momentum 2.
    
    j3 : float
        Total angular momentum 3.
    
    m1 : float 
        z-component of angular momentum 1.
    
    m2 : float
        z-component of angular momentum 2.
    
    m3 : float
        z-component of angular momentum 3.
    
    Returns
    -------
    cg_coeff : float 
        Requested Clebsch-Gordan coefficient.
    
    """
    if m3!=m1+m2:
        return 0
    vmin=int(max([-j1+j2+m3,-j1+m1,0]))
    vmax=int(min([j2+j3+m1,j3-j1+j2,j3+m3]))
    C=np.sqrt((2.0*j3+1.0)*factorial(j3+j1-j2)*factorial(j3-j1+j2)*factorial(j1+j2-j3)*factorial(j3+m3)*factorial(j3-m3)/(factorial(j1+j2+j3+1)*factorial(j1-m1)*factorial(j1+m1)*factorial(j2-m2)*factorial(j2+m2)))
    S=0
    for v in range(vmin,vmax+1):
        S+=(-1.0)**(v+j2+m2)/factorial(v)*factorial(j2+j3+m1-v)*factorial(j1-m1+v)/factorial(j3-j1+j2-v)/factorial(j3+m3-v)/factorial(v+j1-j2-m3)
    C=C*S
    return C

def real2cubic(ll,ns):

    nbnd = 2*ll+1
    norb = nbnd*ns

    temp = np.zeros((nbnd,nbnd), dtype=np.float)
    if   ll == 0 :
        temp[0,0] = +1.0
    elif ll == 1 :
        temp[1,0] = +1.0
        temp[2,1] = +1.0
        temp[0,2] = +1.0
    elif ll == 2 :
        temp[4,0] = +1.0
        temp[2,1] = +1.0
        temp[1,2] = +1.0
        temp[3,3] = +1.0
        temp[0,4] = +1.0
    elif ll == 3 :
        temp = np.zeros((nbnd,nbnd), dtype=np.float)
        temp[4,0] = +1.0
        temp[1,1] = -np.sqrt( 6.0)/4.0
        temp[5,1] = +np.sqrt(10.0)/4.0
        temp[2,2] = -np.sqrt( 6.0)/4.0
        temp[6,2] = -np.sqrt(10.0)/4.0
        temp[0,3] = +1.0
        temp[1,4] = -np.sqrt(10.0)/4.0
        temp[5,4] = -np.sqrt( 6.0)/4.0
        temp[2,5] = +np.sqrt(10.0)/4.0
        temp[6,5] = -np.sqrt( 6.0)/4.0
        temp[3,6] = +1.0
    else :
        print " Unsupported orbital moment in wan_real2cubic ! "
        print


    tran = np.zeros((norb,norb), dtype=np.float)
    if   ns == 1 :
        tran = temp
    elif ns == 2 :
        tran[0:norb:ns, 0:nbnd] = temp
        tran[1:norb:ns, nbnd:norb] = temp
    else :
        print " Unsupported ns in wan_real2cubic ! "
        print

    return tran

def complex2real(ll,ns):

    nbnd = 2*ll+1
    norb = nbnd*ns

    temp = np.zeros((nbnd,nbnd),dtype=np.complex)
    if   ll == 3 :
        temp[3,0] = +1.0
        temp[2,1] = +np.sqrt(0.5)
        temp[4,1] = -np.sqrt(0.5)
        temp[2,2] = +np.sqrt(0.5)*1j
        temp[4,2] = +np.sqrt(0.5)*1j
        temp[1,3] = +np.sqrt(0.5)
        temp[5,3] = +np.sqrt(0.5)
        temp[1,4] = +np.sqrt(0.5)*1j
        temp[5,4] = -np.sqrt(0.5)*1j
        temp[0,5] = +np.sqrt(0.5)
        temp[6,5] = -np.sqrt(0.5)
        temp[0,6] = +np.sqrt(0.5)*1j
        temp[6,6] = +np.sqrt(0.5)*1j
    elif ll == 2 :
        temp[2,0] = +1.0
        temp[1,1] = +np.sqrt(0.5)
        temp[3,1] = -np.sqrt(0.5)
        temp[1,2] = +np.sqrt(0.5)*1j
        temp[3,2] = +np.sqrt(0.5)*1j
        temp[0,3] = +np.sqrt(0.5)
        temp[4,3] = +np.sqrt(0.5)
        temp[0,4] = +np.sqrt(0.5)*1j
        temp[4,4] = -np.sqrt(0.5)*1j
    elif ll == 1 :
        temp[1,0] = +1.0
        temp[0,1] = +np.sqrt(0.5)
        temp[2,1] = -np.sqrt(0.5)
        temp[0,2] = +np.sqrt(0.5)*1j
        temp[2,2] = +np.sqrt(0.5)*1j
    elif ll == 0 :
        temp[0,0] = +1.0

    tran = np.zeros((norb,norb), dtype=np.complex)
    if   ns == 1 :
        tran = temp
    elif ns == 2 :
        tran[0:norb:ns, 0:norb:ns] = temp
        tran[1:norb:ns, 1:norb:ns] = temp
    else :
        print " Unsupported ns in wan_basis.complex2real ! "
        print

    return tran


if __name__ == "__main__" :
    
    cgmat = clebsch_gordan(2)

    u1 = np.zeros((10,10),dtype=np.float)
    u2 = np.zeros((10,10),dtype=np.float)

    u1[ 0, 4] = 1.0
    u1[ 1, 6] = 1.0
    u1[ 2, 8] = 1.0
    u1[ 3, 9] = 1.0
    u1[ 4, 7] = 1.0
    u1[ 5, 5] = 1.0
    u1[ 6, 0] = 1.0
    u1[ 7, 2] = 1.0
    u1[ 8, 3] = 1.0
    u1[ 9, 1] = 1.0

    u2[ 0, 8] = 1.0
    u2[ 1, 9] = 1.0
    u2[ 2, 6] = 1.0
    u2[ 3, 7] = 1.0
    u2[ 4, 4] = 1.0
    u2[ 5, 5] = 1.0
    u2[ 6, 2] = 1.0
    u2[ 7, 3] = 1.0
    u2[ 8, 0] = 1.0
    u2[ 9, 1] = 1.0

    np.set_printoptions(precision=3,suppress=True)
    print "C-G Mat:"
    print(cgmat.real)
    print "U1 :"
    print(u1)
    print "U2 :"
    print(u2)
    print "C-G Mat (NEW) :"
    print(np.dot(np.dot(u1,cgmat),u2).transpose().real)

    #     