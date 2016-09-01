# -*- coding: utf-8 -*-
import numpy as np
from   scipy import constants

np.set_printoptions(precision=3,suppress=True,linewidth=160)

def PauliMatrix():
    
    sgmx = np.zeros((2,2),dtype=np.complex)
    sgmy = np.zeros((2,2),dtype=np.complex)
    sgmz = np.zeros((2,2),dtype=np.complex)
    
    sgmx[0,1] =  1
    sgmx[1,0] =  1
    
    sgmy[0,1] = -1j
    sgmy[1,0] =  1j
    
    sgmz[0,0] =  1
    sgmz[1,1] = -1
    
    return sgmx, sgmy, sgmz

def genladder(ll,ns):
    
    print 
    print " Generate ladder operator !"
    print " For f orbitals, L=3"
    print " The basis :"
    print " |-3,up,down>, |-2,up,down>, |-1,up,down>, |0,up,down>"
    print " | 1,up,down>, | 2,up,down>, | 3,up,down>"
    print 

    nm = 2*ll + 1
    
    Lpls = np.zeros((nm*ns,nm*ns),dtype=np.float)
    Lmin = np.zeros((nm*ns,nm*ns),dtype=np.float)
    
    for mm in range(nm-1):
        mj = -1.0*ll + np.float(mm)
        mi = mj + 1.0
        for ss in range(ns):
            Lpls[ns*(mm+1)+ss,ns*mm+ss] = np.sqrt((ll+mi)*(ll-mj))
            
    for mm in range(1,nm):
        mj = -1.0*ll + np.float(mm)
        mi = mj - 1.0
        for ss in range(ns):
            Lmin[ns*(mm-1)+ss,ns*mm+ss] = np.sqrt((ll-mi)*(ll+mj))
    
    return Lpls, Lmin
    
def genLxyz(ll, ns, Lpls, Lmin):
    
    nm = 2*ll+1
    
    LL = np.zeros((3,nm*ns,nm*ns),dtype=np.complex)
    
    LL[0,:,:] = (Lpls+Lmin)/(2.0+1j*0.0)
    LL[1,:,:] = (Lpls-Lmin)/(1j*2.0)
    
    for mm in range(nm):
        mi = -1.0*ll + np.float(mm)
        for ss in range(ns):
            LL[2,ns*mm+ss,ns*mm+ss] = mi
    
    return LL
    
def genSxyz(ll,ns):
    
    sgmx,sgmy,sgmz = PauliMatrix()
    
    nm = 2*ll+1
    
    SS = np.zeros((3,nm*ns,nm*ns),dtype=np.complex)
    
    for mi in range(nm):
        SS[0,ns*mi:ns*(mi+1), ns*mi:ns*(mi+1)] = sgmx
        SS[1,ns*mi:ns*(mi+1), ns*mi:ns*(mi+1)] = sgmy
        SS[2,ns*mi:ns*(mi+1), ns*mi:ns*(mi+1)] = sgmz

    print SS[0,:,:]
    print SS[1,:,:]
    print SS[2,:,:]
    
    return SS
    
def Zeeman(mag, LL, SS):
    
    norb = SS.shape[1]
    muB  = constants.physical_constants["Bohr magneton in eV/T"][0]
    
    print " Magnetic field in x, y, z :"
    print mag
    print 

    Hzee = np.zeros((norb,norb),dtype=np.complex)
    Hzee = muB * ( LL[0,:,:]*mag[0] + SS[0,:,:]*mag[0] + \
                   LL[1,:,:]*mag[1] + SS[1,:,:]*mag[1] + \
                   LL[2,:,:]*mag[2] + SS[2,:,:]*mag[2] )
    
    return Hzee

def dump_Hzee(Hzee):

    norb = Hzee.shape[0]

    file = open('Hzee.dat', 'w')
    for iorb in range(norb):
        for jorb in range(norb):
            print >> file , "%5d%5d%20.10e%20.10e" %(iorb, jorb, Hzee[iorb,jorb].real, Hzee[iorb,jorb].imag)
    file.close()


def Add_Hzee(natm,enks,smat,Hzee):

    from scipy import linalg

    nkpt = smat.shape[0]
    norb = smat.shape[1]/natm
    nbnd = smat.shape[2]

    eimp = np.zeros((norb*natm,norb*natm),dtype=np.complex)
    kwt  = 1.0/np.float(nkpt)

    for ikpt in range(nkpt):
        ekk = np.zeros((nbnd,nbnd),dtype=np.complex)
        np.fill_diagonal( ekk, enks[ikpt,:] )
        eimp += ( prjloc( ekk, smat[ikpt,:,:] ) * kwt )

    print " Eimp (Real Part) :"
    print eimp.real
    print

    Heff = np.zeros((norb*natm,norb*natm), dtype=np.complex)
    Hloc = np.zeros((norb,norb), dtype=np.complex)
    eloc = np.zeros((norb*natm), dtype=np.float)
    lvec = np.zeros((norb*natm,norb*natm), dtype=np.complex)
    for iatm in range(natm):
        Hloc = eimp[iatm*norb:(iatm+1)*norb,iatm*norb:(iatm+1)*norb] #+ Hzee
        Heff[iatm*norb:(iatm+1)*norb,iatm*norb:(iatm+1)*norb] = Hloc
        eloc[iatm*norb:(iatm+1)*norb],lvec[iatm*norb:(iatm+1)*norb,iatm*norb:(iatm+1)*norb] = \
            linalg.eigh(Hloc)

    print " Eloc (Real Part) :"
    for iatm in range(natm):
        print eloc[iatm*norb:(iatm+1)*norb]
    print

    lcef = np.zeros((norb*natm,norb*natm),dtype=np.complex)
    np.fill_diagonal(lcef, eloc)
    lcef = eimp - lcef

    vtmp = np.zeros((nbnd,nbnd),dtype=np.complex)
    Hblh = np.zeros((nbnd,nbnd),dtype=np.complex)
    enks_new = np.zeros((nkpt,nbnd),dtype=np.float)
    smat_new = np.zeros((nkpt,norb*natm,nbnd),dtype=np.complex)
    for ikpt in range(nkpt):
        Hblh = prjblh(lcef,smat[ikpt,:,:])
        ekk  = np.zeros((nbnd,nbnd),dtype=np.complex)
        np.fill_diagonal( ekk, enks[ikpt,:] )
        Hblh = ekk - Hblh

        enks_new[ikpt,:],vtmp = linalg.eigh(Hblh)
        smat_new[ikpt,:,:] = np.dot( smat[ikpt,:,:],vtmp )

    # Check New Projector
    #========================================================================#
    eimp = np.zeros((norb*natm,norb*natm),dtype=np.complex)
    for ikpt in range(nkpt):
        ekk = np.zeros((nbnd,nbnd),dtype=np.complex)
        np.fill_diagonal( ekk, enks_new[ikpt,:] )
        stmp = smat_new[ikpt,:,:]
        eimp += ( prjloc( ekk, stmp ) * kwt )
    print " Eimp (Real Part) :"
    for iatm in range(natm):
        print eimp[iatm*norb:(iatm+1)*norb, iatm*norb:(iatm+1)*norb].real
    print
    #========================================================================#

    return enks_new, smat_new

def magnetic_field(lmom,ns,mag):

    from cgmat import clebsch_gordan
    
    # gen Ladder operator
    Lpls, Lmin = genladder(lmom,ns)
    
    # gen Lx, Ly, Lz
    LL = genLxyz(lmom, ns, Lpls, Lmin)
    
    # gen Sx, Sy, Sz
    SS = genSxyz(lmom,ns)
    
    # gen Zeeman matrix
    Hzee = Zeeman(mag,LL,SS)
    print " H_zeeman in | Lz, Sz > :"
    print Hzee
    print 

    # gen c-g matrix
    cgmat = clebsch_gordan(lmom)

    # transform basis from | lz, sz > to | jj, jz >
    Hzee = np.dot( np.dot(cgmat,Hzee), cgmat.transpose().conj() )

    print " H_zeeman in | jj, jz > :"
    print Hzee.real
    print Hzee.imag
    print 

    return Hzee

if __name__ == "__main__" :

    mag  = np.array((0.0, 0.0, 1.0),dtype=np.float)

    magnetic_field(1,2,mag)
    