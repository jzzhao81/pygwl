
import sys
import numpy as np

def gwl_core(gatm):

    from scipy import optimize
    from gwl_tools import chkrnd

    # mfck = |I><I|, mgmm = |G><G|
    mfck = np.ones(gatm.ncfg, dtype=np.float)
    mgmm = np.zeros(gatm.nasy, dtype=np.float)
    meta = np.zeros((gatm.norb,gatm.nasy), dtype=np.float)
    fadd = np.zeros((gatm.norb,gatm.ncfg,gatm.ncfg), dtype=np.float)
    frmv = np.zeros((gatm.norb,gatm.ncfg,gatm.ncfg), dtype=np.float)
    hbsn = np.zeros((gatm.nasy,gatm.nasy), dtype=np.complex)

    for icfg in range(gatm.ncfg):
        code = format(icfg,'0'+str(gatm.norb)+'b')
        for iorb in range(gatm.norb):
            if code[iorb] == '0' :
                mfck[icfg] *= (1.0-gatm.nloc[iorb])
            else :
                mfck[icfg] *= gatm.nloc[iorb]

    mgmm = np.diag(np.dot(np.linalg.inv(gatm.avec),np.dot(np.diag(mfck),gatm.avec))).real
    fmat = np.zeros( gatm.nasy, dtype=np.float )
    for icfg in range(gatm.ncfg):
        isym = gatm.cfg2sym[icfg]
        fmat[isym] += mgmm[icfg]

    # N = Tr\psi\psi n
    for iorb in range(gatm.norb):
        for jcfg in range(gatm.ncfg):
            code = format(jcfg,'0'+str(gatm.norb)+'b')
            if code[iorb] == '0' : continue
            for icfg in range(gatm.ncfg):
                isym=gatm.cfg2sym[icfg]
                zz = np.absolute(gatm.avec[jcfg,icfg])
                meta[iorb,isym] += zz*zz*float(code[iorb])*mfck[jcfg]
    # f^\dag
    for iorb in range(gatm.norb):
        for jcfg in range(gatm.ncfg):
            code = format(jcfg,'0'+str(gatm.norb)+'b')
            sign = 1
            if code[iorb] == '0' :
                for jorb in range(iorb):
                    if code[jorb] == '1' : sign = -sign
                code = code[:iorb]+'1'+code[iorb+1:]
                icfg = int(code,2)
                fadd[iorb,icfg,jcfg] = sign
        # tranform from fock state to atomic eigen state
        fadd[iorb,:,:] = np.dot( gatm.avec.transpose().conj(), \
            np.dot(fadd[iorb,:,:],gatm.avec) ).real

    # f
    for iorb in range(gatm.norb):
        nume = np.sqrt(gatm.nloc[iorb]*(1.0-gatm.nloc[iorb]))
        for jcfg in range(gatm.ncfg):
            code = format(jcfg,'0'+str(gatm.norb)+'b')
            sign = 1
            if code[iorb] == '1' :
                for jorb in range(iorb):
                    if code[jorb] == '1' : sign = -sign
                code = code[:iorb]+'0'+code[iorb+1:]
                icfg = int(code,2)
                frmv[iorb,icfg,jcfg] = sign * np.sqrt(mfck[icfg]*mfck[jcfg])/nume
        # tranform from fock state to atomic eigen state
        frmv[iorb,:,:] = np.dot( gatm.avec.transpose().conj(), \
            np.dot(frmv[iorb,:,:],gatm.avec) ).real

    # construct Boson Hamiltonian in atomic eigen state
    # Add atomic eigen value
    for icfg in range(gatm.ncfg):
        isym=gatm.cfg2sym[icfg]
        hbsn[isym,isym] += mgmm[icfg]*gatm.aeig[icfg]

    for iorb in range(gatm.norb):
        for icfg in range(gatm.ncfg):
            for jcfg in range(gatm.ncfg):
                isym = gatm.cfg2sym[icfg]
                jsym = gatm.cfg2sym[jcfg]
                hbsn[isym,jsym] += gatm.dedr[iorb]*fadd[iorb,icfg,jcfg]*frmv[iorb,jcfg,icfg]
                hbsn[jsym,isym] += gatm.dedr[iorb]*fadd[iorb,icfg,jcfg]*frmv[iorb,jcfg,icfg]

    for isym in range(gatm.nasy):
        for jsym in range(gatm.nasy):
            hbsn[isym,jsym] /= np.sqrt(fmat[isym]*fmat[jsym])

    for isym in range(gatm.nasy):
        meta[:,isym] /= fmat[isym]

    ini   = np.repeat(gatm.uj[0]*(np.sum(gatm.nloc)-0.5), gatm.norb)
    dltn  = lambda lamda: gwl_gennloc(lamda, gatm, hbsn, meta)
    rslt  = optimize.root(dltn, ini, method='lm', tol=1e-8)
    lamda = rslt.x
    if not(rslt.success) :
        print rslt
        print
        sys.exit(" gwl_core loop does not converged !\n")

    # Ground state wave function
    wtemp = gwl_gwf(lamda, gatm, hbsn, meta)
    gwf   = np.zeros(gatm.ncfg,dtype=np.complex)
    for icfg in range(gatm.ncfg):
        isym = gatm.cfg2sym[icfg]
        gwf[icfg] = wtemp[isym]/np.sqrt(fmat[isym])

    gatm.qre[:] = 0.0
    for iorb in range(gatm.norb):
        fmat = np.zeros((gatm.ncfg,gatm.ncfg),dtype=np.float)
        for icfg in range(gatm.ncfg):
            for jcfg in range(gatm.ncfg):
                fmat[icfg,jcfg] = fadd[iorb,icfg,jcfg] * frmv[iorb,jcfg,icfg]
        fmat = np.dot(fmat.transpose(), gwf)
        for icfg in range(gatm.ncfg):
            gatm.qre[iorb] += (gwf[icfg]*fmat[icfg]).real

    for iorb in range(gatm.norb):
        nume = 2.0*(gatm.dedr[iorb]*gatm.qre[iorb])*\
            (gatm.nloc[iorb]-0.5)/(gatm.nloc[iorb]*(1.0-gatm.nloc[iorb]))
        gatm.elm[iorb] = nume + lamda[iorb] - gatm.eimp[iorb]

    gatm.qre = gatm.osym.symmetrize(gatm.qre)
    gatm.elm = gatm.osym.symmetrize(gatm.elm)

    print ' gatm.dedr :'
    print gatm.dedr[0::2]
    print gatm.dedr[1::2]
    print ' gatm.elm :'
    print gatm.elm[0::2]
    print gatm.elm[1::2]
    print ' gatm.qre :'
    print gatm.qre[0::2]
    print gatm.qre[1::2]
    print ' gatm.eimp :'
    print gatm.eimp[0::2]
    print gatm.eimp[1::2]
    print ' gatm.nloc :',(" %10.5f") %(np.sum(gatm.nloc))
    print gatm.nloc[0::2]
    print gatm.nloc[1::2]

    return

def gwl_gennloc(lamda, gatm, hbsn, meta):

    from scipy import linalg

    hnew = np.copy(hbsn)
    for isym in range(gatm.nasy):
        ceff = 0.0
        for iorb in range(gatm.norb):
            ceff += -lamda[iorb] * meta[iorb,isym]
        hnew[isym,isym] += ceff

    etmp, vtmp = linalg.eigh(hnew)

    nnew = np.zeros(gatm.norb, dtype=np.float)
    for iorb in range(gatm.norb):
        for isym in range(gatm.nasy):
            nnew[iorb] += (meta[iorb,isym]*vtmp[isym,0]**2.0).real

    # make new local occupation symmetrized
    # nnew = gatm.symm.symmetrize(nnew)

    return nnew-gatm.nloc

def gwl_gwf(lamda, gatm, hbsn, meta):

    from scipy import linalg

    hnew = np.copy(hbsn)
    for isym in range(gatm.nasy):
        ceff = 0.0
        for iorb in range(gatm.norb):
            ceff += -lamda[iorb] * meta[iorb,isym]
        hnew[isym,isym] += ceff

    etmp, vtmp = linalg.eigh(hnew)

    return vtmp[:,0]

