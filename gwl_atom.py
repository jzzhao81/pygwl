
import sys
import numpy as np
from gwl_symm import symmetry

class corr_atom:

    def __init__(self, nkpt=1, nbnd=1, norb=1, ntot=0.0):
        self.nkpt = nkpt
        self.nbnd = nbnd
        self.norb = norb
        self.ntot = ntot
        self.ncfg = 2**self.norb
        self.mu   = 0.0
        self.smat = np.zeros((self.nkpt,self.norb,self.nbnd), dtype=np.complex)
        self.fdis = np.zeros((self.nkpt,self.nbnd), dtype=np.float)
        self.nloc = np.zeros(self.norb, dtype=np.float)
        self.eimp = np.zeros(self.norb, dtype=np.float)
        self.eigs = np.zeros((self.nkpt,self.nbnd), dtype=np.float)
        self.kwt  = np.repeat(1.0/np.float(nkpt), nkpt)
        self.elm  = np.zeros(self.norb, dtype=np.float )
        self.udc  = np.zeros(self.norb, dtype=np.float )
        self.qre  = np.ones(self.norb, dtype=np.float )
        self.dedr = np.zeros(self.norb, dtype=np.float)
        self.aeig = np.zeros(self.ncfg, dtype=np.float)
        self.avec = np.zeros((self.ncfg,self.ncfg), dtype=np.complex)
        self.uj   = np.zeros(5, dtype=np.float)
        self.symm = symmetry()
        self.iter = 1

    def geneimp(self,eigs,smat):
        from gwl_tools import chkrnd
        nkpt = smat.shape[0]
        norb = smat.shape[1]
        nbnd = smat.shape[2]
        eimp = np.zeros(norb, dtype=np.float)
        emat = np.zeros((norb,norb),dtype=np.complex)
        for ikpt in range(nkpt):
            emat += self.prjloc(np.diag(eigs[ikpt,:]),smat[ikpt,:,:])*self.kwt[ikpt] 
        # Check Eimp is diagonal and real, or not
        if not(chkrnd(emat)) : 
            print emat
            sys.exit(" Eimp should be REAL & DIAGONAL in this code !\n")
        eimp = np.diag(emat).real
        del emat
        return eimp

    def genfdis(self, eigs):
        nkpt = eigs.shape[0]
        nbnd = eigs.shape[1]
        fdis = np.zeros((nkpt,nbnd),dtype=np.float)
        for ikpt in range(nkpt):
            for ibnd in range(nbnd):
                fdis[ikpt,ibnd] = self.fermidis(eigs[ikpt,ibnd])
        return fdis

    def gennloc(self, eigs, smat):
        from gwl_tools import chkrnd
        nkpt = smat.shape[0]
        norb = smat.shape[1]
        nbnd = smat.shape[2]
        nmat = np.zeros((norb,norb),dtype=np.complex)
        fdis = self.genfdis(eigs)
        for ikpt in range(nkpt):
            nmat += self.prjloc(np.diag(fdis[ikpt,:]),smat[ikpt,:,:]) * self.kwt[ikpt]
        if not(chkrnd(nmat)) : sys.exit(" Local density matrix should be REAL & DIAGONAL in gennloc !\n")
        nloc = np.diag(nmat).real
        del nmat
        return nloc

    def prjloc(self, imat, smat):
        omat = np.dot( np.dot(smat, imat), smat.transpose().conj() )
        return omat

    def prjblh(self, imat, smat):
        omat = np.dot( smat.transpose().conj(), np.dot(imat, smat) )
        return omat

    def searchmu(self, eigs):
        from scipy.optimize import brentq
        f = lambda x : self.genntot(eigs-x)-self.ntot
        mu,r  = brentq(f,eigs.min(),eigs.max(),maxiter=200,full_output=True)
        if not(r.converged) : print " Failed to search mu ! "
        return mu

    def genntot(self, eigs):
        nkpt = eigs.shape[0]; nbnd = eigs.shape[1]
        if nkpt != self.nkpt or nbnd != self.nbnd : 
            print " Error array size in corr_atom.genntot !"
        ntot = 0.0
        fdis = self.genfdis(eigs)
        for ikpt in range(nkpt):
            for ibnd in range(nbnd):
                ntot += self.kwt[ikpt]*fdis[ikpt,ibnd]
        return ntot

    def fermidis(self,ene):
        from gwl_constants import beta
        if ene <= 0.0 : 
            dist = 1.0/(np.exp(ene*beta)+1.0)
        else :
            dist = np.exp(-1.0*ene*beta)/(1.0+np.exp(-1.0*ene*beta))
        return dist

    def eigenstate(self):
        from atm_hmat import atom_umat, atom_hmat, dump_eigs
        from gwl_constants import u, j
        from scipy import linalg
        self.uj[0] = u; self.uj[1] = u-2.0*j
        self.uj[2] = j; self.uj[3] = j; self.uj[4] = j
        umat = atom_umat(self.norb,self.uj)
        hmat = atom_hmat(np.diag(self.eimp),umat)
        self.aeig, self.avec = linalg.eigh(hmat)
        dump_eigs(self.aeig, self.avec)
        del umat, hmat
        return

    def outerloop(self,inp):
        from gwl_ksum import gwl_ksum
        from gwl_core import gwl_core
        if inp.shape[0] != 2*self.norb :
            sys.exit(" Error input array in outloop !\n")

        print " Iter  :", self.iter
        self.elm = np.copy(inp[:self.norb])
        self.qre = np.copy(inp[self.norb:])
        eold = np.copy(self.elm)
        qold = np.copy(self.qre)

        # k summation
        gwl_ksum(self)
        # inner loop
        gwl_core(self)

        diff = np.append( (self.elm-eold), (self.qre-qold) )

        self.iter += 1
        print " diff :"
        print diff[:self.norb]
        print diff[self.norb:]
        print

        return diff

