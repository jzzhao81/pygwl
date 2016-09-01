
import sys
import numpy as np
from gwl_symm import symmetry
from gwl_tools import prjloc

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
            emat += prjloc(np.diag(eigs[ikpt,:]),smat[ikpt,:,:])*self.kwt[ikpt]

        # # generate local diagonal basis
        # emat = self.genldbs(emat)

        # # Check Eimp is diagonal and real, or not
        # if not(chkrnd(emat)) :
        #     print emat
        #     sys.exit(" Eimp should be REAL & DIAGONAL in this code !\n")
        eimp = np.diag(emat).real
        del emat
        return eimp

    def gennloc(self, eigs, smat):
        from gwl_tools import chkrnd, genfdis
        nkpt = smat.shape[0]
        norb = smat.shape[1]
        nbnd = smat.shape[2]
        nmat = np.zeros((norb,norb),dtype=np.complex)
        fdis = genfdis(eigs)
        for ikpt in range(nkpt):
            nmat += prjloc(np.diag(fdis[ikpt,:]),smat[ikpt,:,:]) * self.kwt[ikpt]
        # if not(chkrnd(nmat)) :
        #     print nmat
        #     sys.exit(" Local density matrix should be REAL & DIAGONAL in gennloc !\n")
        nloc = np.diag(nmat).real
        del nmat
        return nloc

    def genudc(self,nloc):
        from gwl_constants import nudc

        if nudc <= 0.0 :
            nudc = np.sum(self.nloc)

        norb   = nloc.shape[0]
        udc    = np.zeros(norb, dtype=np.float)
        udc[:] = self.uj[0]*(nudc-0.5)-self.uj[2]*(nudc-0.5)
        return udc

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

    # determine degenerate atomic energy eigenstate
    def degenerate(self):
        from atm_hmat import sort_basis
        bsort = sort_basis(self.norb)
        nsort = []
        inew  = []
        for basis in bsort :
            eigs = []
            symm = symmetry()
            conf = []
            for icfg in range(self.ncfg):
                idx = np.where( np.absolute(self.avec[:,icfg])>1e-8 )[0]
                test = [(jdx in basis) for jdx in idx]
                if sum(test) > 0 :
                    eigs.append(self.aeig[icfg]); conf.append(icfg)
            symm.defsym(eigs)
            for isym in range(symm.nsym):
                data = []
                for idat in range(symm.npsy[isym]):
                    data.append(conf[symm.indx[isym][idat]])
                nsort.append(data)
            for icfg in symm.data : inew.append(conf[icfg])

        self.asym = symmetry(inew)
        self.cfg2sym = np.zeros(self.ncfg,dtype=int)
        for icfg in range(self.ncfg):
            self.cfg2sym[icfg] = self.getindex(nsort,icfg)
        self.nasy = self.asym.nsym
        del bsort, nsort, inew, eigs, symm, conf, self.asym
        return

    # get symmetry index from configuration index
    def getindex(self,sort,inp):
        status = 0
        for isym, sublist in enumerate(sort):
            if inp in sublist: return isym
        return -1

    # generate local diagonal basis
    def genldbs(self,emat):
        from gwl_constants import ldbs
        from scipy import linalg
        ndim = emat.shape[0]
        if ndim != self.norb : sys.exit(" Error in corr_atom.genldbs !\n")
        if ldbs == 1 :
            eigs, self.ldmx = linalg.eigh(emat)
            emat = np.diag(eigs)
            for ikpt in range(self.nkpt):
                self.smat[ikpt,:,:] = np.dot( linalg.inv(self.ldmx),self.smat[ikpt,:,:] )
        if ldbs == 0 :
            self.ldmx = np.eye(ndim, dtype=np.float)
        else :
            print " LDBS = ",ldbs," : under construction !\n"
            self.ldmx = np.eye(self.norb, dtype=np.float)

        return emat

    # gutzwiller inner loop
    def gwl_core(self):

        from scipy import optimize
        from gwl_tools import chkrnd

        # mfck = |I><I|, mgmm = |G><G|
        mfck = np.ones(self.ncfg, dtype=np.float)
        mgmm = np.zeros(self.nasy, dtype=np.float)
        meta = np.zeros((self.norb, self.nasy), dtype=np.float)
        fadd = np.zeros((self.norb, self.ncfg, self.ncfg), dtype=np.float)
        frmv = np.zeros((self.norb, self.ncfg, self.ncfg), dtype=np.float)
        hbsn = np.zeros((self.nasy, self.nasy), dtype=np.complex)

        for icfg in range(self.ncfg):
            code = format(icfg, '0' + str(self.norb) + 'b')
            for iorb in range(self.norb):
                if code[iorb] == '0':
                    mfck[icfg] *= (1.0 - self.nloc[iorb])
                else:
                    mfck[icfg] *= self.nloc[iorb]

        mgmm = np.diag(np.dot(np.linalg.inv(self.avec), np.dot(np.diag(mfck), self.avec))).real
        fmat = np.zeros(self.nasy, dtype=np.float)
        for icfg in range(self.ncfg):
            isym = self.cfg2sym[icfg]
            fmat[isym] += mgmm[icfg]

        # N = Tr\psi\psi n
        for iorb in range(self.norb):
            for jcfg in range(self.ncfg):
                code = format(jcfg, '0' + str(self.norb) + 'b')
                if code[iorb] == '0': continue
                for icfg in range(self.ncfg):
                    isym = self.cfg2sym[icfg]
                    zz = np.absolute(self.avec[jcfg, icfg])
                    meta[iorb, isym] += zz * zz * float(code[iorb]) * mfck[jcfg]
        # f^\dag
        for iorb in range(self.norb):
            for jcfg in range(self.ncfg):
                code = format(jcfg, '0' + str(self.norb) + 'b')
                sign = 1
                if code[iorb] == '0':
                    for jorb in range(iorb):
                        if code[jorb] == '1': sign = -sign
                    code = code[:iorb] + '1' + code[iorb + 1:]
                    icfg = int(code, 2)
                    fadd[iorb, icfg, jcfg] = sign
            # tranform from fock state to atomic eigen state
            fadd[iorb, :, :] = np.dot(self.avec.transpose().conj(), \
                                      np.dot(fadd[iorb, :, :], self.avec)).real

        # f
        for iorb in range(self.norb):
            nume = np.sqrt(self.nloc[iorb] * (1.0 - self.nloc[iorb]))
            for jcfg in range(self.ncfg):
                code = format(jcfg, '0' + str(self.norb) + 'b')
                sign = 1
                if code[iorb] == '1':
                    for jorb in range(iorb):
                        if code[jorb] == '1': sign = -sign
                    code = code[:iorb] + '0' + code[iorb + 1:]
                    icfg = int(code, 2)
                    frmv[iorb, icfg, jcfg] = sign * np.sqrt(mfck[icfg] * mfck[jcfg]) / nume
            # tranform from fock state to atomic eigen state
            frmv[iorb, :, :] = np.dot(self.avec.transpose().conj(), \
                                      np.dot(frmv[iorb, :, :], self.avec)).real

        # construct Boson Hamiltonian in atomic eigen state
        # Add atomic eigen value
        for icfg in range(self.ncfg):
            isym = self.cfg2sym[icfg]
            hbsn[isym, isym] += mgmm[icfg] * self.aeig[icfg]

        for iorb in range(self.norb):
            for icfg in range(self.ncfg):
                for jcfg in range(self.ncfg):
                    isym = self.cfg2sym[icfg]
                    jsym = self.cfg2sym[jcfg]
                    hbsn[isym, jsym] += self.dedr[iorb] * fadd[iorb, icfg, jcfg] * frmv[iorb, jcfg, icfg]
                    hbsn[jsym, isym] += self.dedr[iorb] * fadd[iorb, icfg, jcfg] * frmv[iorb, jcfg, icfg]

        for isym in range(self.nasy):
            for jsym in range(self.nasy):
                hbsn[isym, jsym] /= np.sqrt(fmat[isym] * fmat[jsym])

        for isym in range(self.nasy):
            meta[:, isym] /= fmat[isym]

        ini = np.repeat(self.uj[0] * (np.sum(self.nloc) - 0.5), self.norb)
        dltn = lambda lamda: self.gwl_gennloc(lamda, hbsn, meta)
        rslt = optimize.root(dltn, ini, method='lm', tol=1e-8)
        lamda = rslt.x
        if not (rslt.success):
            print rslt
            print
            sys.exit(" gwl_core loop does not converged !\n")

        # Ground state wave function
        wtemp = self.gwl_gwf(lamda, hbsn, meta)
        gwf = np.zeros(self.ncfg, dtype=np.complex)
        for icfg in range(self.ncfg):
            isym = self.cfg2sym[icfg]
            gwf[icfg] = wtemp[isym] / np.sqrt(fmat[isym])

        self.qre[:] = 0.0
        for iorb in range(self.norb):
            fmat = np.zeros((self.ncfg, self.ncfg), dtype=np.float)
            for icfg in range(self.ncfg):
                for jcfg in range(self.ncfg):
                    fmat[icfg, jcfg] = fadd[iorb, icfg, jcfg] * frmv[iorb, jcfg, icfg]
            fmat = np.dot(fmat.transpose(), gwf)
            for icfg in range(self.ncfg):
                self.qre[iorb] += (gwf[icfg] * fmat[icfg]).real

        for iorb in range(self.norb):
            nume = 2.0 * (self.dedr[iorb] * self.qre[iorb]) * \
                   (self.nloc[iorb] - 0.5) / (self.nloc[iorb] * (1.0 - self.nloc[iorb]))
            self.elm[iorb] = nume + lamda[iorb] - self.eimp[iorb]

        self.qre = self.osym.symmetrize(self.qre)
        self.elm = self.osym.symmetrize(self.elm)

        print ' gatm.dedr :'
        print self.dedr
        print ' gatm.elm :'
        print self.elm
        print ' gatm.qre :'
        print self.qre
        print ' gatm.eimp :'
        print self.eimp
        print ' gatm.nloc :', (" %10.5f") % (np.sum(self.nloc))
        print self.nloc

        return

    def gwl_gennloc(self, lamda, hbsn, meta):

        from scipy import linalg

        hnew = np.copy(hbsn)
        for isym in range(self.nasy):
            ceff = 0.0
            for iorb in range(self.norb):
                ceff += -lamda[iorb] * meta[iorb,isym]
            hnew[isym,isym] += ceff

        eigs, evec = linalg.eigh(hnew)

        nnew = np.zeros(self.norb, dtype=np.float)
        for iorb in range(self.norb):
            for isym in range(self.nasy):
                nnew[iorb] += (meta[iorb,isym]*evec[isym,0]**2.0).real

        # make new local occupation symmetrized
        # nnew = gatm.symm.symmetrize(nnew)

        return nnew-self.nloc

    def gwl_gwf(self, lamda, hbsn, meta):

        from scipy import linalg

        hnew = np.copy(hbsn)
        for isym in range(self.nasy):
            ceff = 0.0
            for iorb in range(self.norb):
                ceff += -lamda[iorb] * meta[iorb,isym]
            hnew[isym,isym] += ceff

        eigs, evec = linalg.eigh(hnew)

        return evec[:,0]


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from scipy.special import ndtr

    atom = corr_atom()

    ene  = np.zeros(501)
    mdis = np.zeros(501)
    fdis = np.zeros(501)
    gdis = np.zeros(501)

    print "Gen mpdis"
    for idat in range(501):
        # print -1.0+idat*0.02,atom.mpdis(-1.0+idat*0.02)
        ene[idat] = -5.0+idat*0.02
        mdis[idat] = atom.mpdis(ene[idat],2)

    print "Gen fermidis"
    for idat in range(501):
        fdis[idat] = atom.fermidis(ene[idat],5)

    print "Gen gaussdis"
    for idat in range(501):
        gdis[idat] = atom.gaussdis(ene[idat],0.2)

    plt.plot(ene,mdis,label="MP")
    plt.plot(ene,fdis,label="Fermi")
    plt.plot(ene,gdis,label="Gauss")
    plt.legend()
    plt.show()

