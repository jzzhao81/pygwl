# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def gen_kham_bethe():
    nbnd = 3
    nspn = 2
    norb = nbnd * nspn
    nkpt = 2001
    kmin = -1.0
    kmax =  1.0
    delt = -0.5

    ham  = np.zeros((nkpt,norb,norb),dtype=np.complex)
    for ikpt in range(nkpt):
        kpt = kmin + np.float(ikpt)*2.0/np.float(nkpt-1)
        for iorb in range(norb):
            if iorb < 2 :
                ham[ikpt,iorb,iorb] = np.float(kpt)-delt/2.0
            else :
                ham[ikpt,iorb,iorb] = np.float(kpt)+delt/2.0

    return ham

def gen_kham_2d():
    
    # dimension variable
    nbnd = 5
    nspn = 2
    norb = nbnd * nspn
    nkpx = 100
    nkpy = 100
    nkpt = nkpx * nkpy
    kxst = -np.pi
    kxed =  np.pi
    kyst = -np.pi
    kyed =  np.pi
    
    # energy variable
    eimp = 0.0
    thop = 1.0
    
    ham0 = np.zeros((norb,norb),dtype=np.complex)
    ham1 = np.zeros((nkpt,norb,norb),dtype=np.complex)
    ham  = np.zeros((nkpt,norb,norb),dtype=np.complex)
    
    for iorb in range(norb):
        if iorb < norb/2 :
            ham0[iorb,iorb] = -eimp
        else:
            ham0[iorb,iorb] =  eimp
    
    print 'ham0:'
    print ham0
    
    ikpt = 0
    for ikpx in range(nkpx):
        kxpt = kxst+ikpx*(kxed-kxst)/nkpx
        for ikpy in range(nkpy):
            kypt = kyst+ikpy*(kyed-kyst)/nkpy
            for iorb in range(norb):
                ham1[ikpt,iorb,iorb] = -0.5*thop*(np.cos(kxpt)+np.cos(kypt))
            ham[ikpt,:,:] = ham0[:,:] + ham1[ikpt,:,:]
            ikpt += 1
    return ham
    
def gen_kham_3d():
    
    # dimension variable
    nbnd = 2
    nspn = 2
    norb = nbnd * nspn
    nkpx = 20
    nkpy = 20
    nkpz = 20
    nkpt = nkpx * nkpy * nkpz
    kxst = -np.pi
    kxed =  np.pi
    kyst = -np.pi
    kyed =  np.pi
    kzst = -np.pi
    kzed =  np.pi
    
    # energy variable
    eimp = 0.10
    thop = 1.0
    
    ham  = np.zeros((nkpt,norb,norb),dtype=np.complex)

    for iorb in range(norb):
        if iorb < 2 :
            ham[:,iorb,iorb] = -eimp
        else :
            ham[:,iorb,iorb] = eimp
        
    print 'ham0:'
    print ham[0,:,:]
    
    ikpt = 0
    for ikpx in range(nkpx):
        kxpt = kxst+ikpx*(kxed-kxst)/nkpx
        for ikpy in range(nkpy):
            kypt = kyst+ikpy*(kyed-kyst)/nkpy
            for ikpz in range(nkpz):
                kzpt = kzst+ikpz*(kzed-kzst)/nkpz
                for iorb in range(norb):
                    ham[ikpt,iorb,iorb] += -1.0/3.0*thop*(np.cos(kxpt)+np.cos(kypt)+np.cos(kzpt))
                ikpt += 1
    
    return ham
    
def diag_ham(ham):
    
    from scipy import linalg
    
    nkpt = ham.shape[0]
    norb = ham.shape[1]
    
    eigs = np.zeros((nkpt,norb),dtype=np.float)
    evec = np.zeros((nkpt,norb,norb),dtype=np.complex)
    nloc = np.zeros((norb,norb), dtype=np.float)
    
    for ikpt in range(nkpt):
        eigs[ikpt,:],evec[ikpt,:,:] = linalg.eigh(ham[ikpt,:,:])
        for iorb in range(norb):
            if eigs[ikpt,iorb] < 0.0 :
                nloc += np.outer(evec[ikpt,:,iorb],evec[ikpt,:,iorb].conj()).real / nkpt
                
    print 'nloc:'
    print np.diag(nloc)
         
    return eigs, evec
    
def dump_results(eigs, evec):
    
    nkpt = eigs.shape[0]
    norb = eigs.shape[1]
    
    # abinitio parameter
    natm = 1
    ntot = 4.0
    
    # efile = open('eigs.out','w')
    # for ikpt in range(nkpt):
    #     for iorb in range(norb):
    #         print >> efile, "%5d%5d%10.5f" %(ikpt+1,iorb+1,eigs[ikpt,iorb])
    #     print >> efile
    # efile.close()
    
    efile = open('output.enk','w')
    print >> efile, "%10d%-40s" %(nkpt, ' : number of k-points')
    print >> efile, "%10d%-40s" %(norb, ' : number of bands')
    print >> efile, "%10d%-40s" %(norb, ' : number of correlated orbitals')
    print >> efile, "%10d%-40s" %(natm, ' : number of correlated atoms')
    print >> efile, "%10.5f%-40s" %(ntot, ' : number of total electrons')
    for ikpt in range(nkpt):
        print >> efile, "%10s%10d" %(' #ikpt ', ikpt+1)
        for iorb in range(norb):
            print >> efile, "%10d%20.15f" %(iorb+1,eigs[ikpt,iorb])
        print >> efile
    efile.close()
    
    efile = open('output.ovlp','w')
    print >> efile, "%10d%-40s" %(nkpt, ' : number of k-points')
    print >> efile, "%10d%-40s" %(norb, ' : number of bands')
    print >> efile, "%10d%-40s" %(norb, ' : number of correlated orbitals')
    print >> efile, "%10d%-40s" %(natm, ' : number of correlated atoms')
    print >> efile, "%10.5f%-40s" %(ntot, ' : number of total electrons')
    for ikpt in range(nkpt):
        print >> efile, "%10s%10d" %(' #ikpt ', ikpt+1)
        for iorb in range(norb):
            for jorb in range(norb):
                print >> efile, "%10d%10d%20.15f%20.15f" %(iorb+1,jorb+1,\
                evec[ikpt,iorb,jorb].real,evec[ikpt,iorb,jorb].imag)
        print >> efile
    efile.close()
    
    
    # feig = open("eig_new.dat","w")
    # print >> feig, "%10d%5d%5d%3d%3d%50.30s" %(nkpt,1,norb,1,2,\
    # "# nkpt, nsymop, norb, L, nspin")
    # print >> feig, "%5d%5d%5d%16.9f%40.30s" %(1,norb,norb,ntot,\
    # "# nemin, nemax, nbands, ntotal")
    # print >> feig, "%16.9f%29.10s" %(0.0,"# Mu")
    # print >> feig
    # for ikpt in range(nkpt):
    #     print >> feig, "%5d%20.12f%31.20s" %(ikpt+1, 1.0/np.float(nkpt)," # ikpt, k-weight")
    #     for iorb in range(norb):
    #         print >> feig, "%5d%20.12f" %(iorb+1, eigs[ikpt,iorb])
    #     print >> feig
    #     print >> feig
    # feig.close()

    # fprj = open("udmft_new.dat","w")
    # print >> fprj, "%10d%5d%5d%3d%3d%50.32s" %(nkpt,1,norb,1,2,"# nkpt, nsymop, norb, L, nspin")
    # print >> fprj, "%5d%5d%5d%16.9f%40.30s" %(1,norb,norb,2.0,"# nemin, nemax, nbands, ntotal")
    # print >> fprj
    # for ikpt in range(nkpt):
    #     print >> fprj, "%5d%20.12f%31.20s" %(ikpt+1, 1.0/np.float(nkpt),"# ikpt, kweight")
    #     print >> fprj, "%5d%44.20s" %(1,"# isymop")
    #     for iorb in range(norb):
    #         for jorb in range(norb):
    #             print >> fprj, "%5d%5d%20.12f%20.12f" %(jorb+1,iorb+1, \
    #             evec[ikpt,iorb,jorb].real, -evec[ikpt,iorb,jorb].imag )
    #     print >> fprj
    #     print >> fprj
    # fprj.close()

    return
    
if __name__ == "__main__":
    
    ham = gen_kham_bethe()
    
    eigs, evec = diag_ham(ham)
    
    dump_results(eigs, evec)
    
    
