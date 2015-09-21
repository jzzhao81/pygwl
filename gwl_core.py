
import sys
import numpy as np

def gwl_core(gatm):

    from scipy import optimize
    from gwl_tools import chkrnd

    # mfck = |I><I|, mgmm = |G><G|
    mfck = np.ones(gatm.ncfg, dtype=np.float)
    mgmm = np.zeros(gatm.ncfg, dtype=np.float)
    meta = np.zeros((gatm.norb,gatm.ncfg), dtype=np.float)
    fadd = np.zeros((gatm.norb,gatm.ncfg,gatm.ncfg), dtype=np.float)
    frmv = np.zeros((gatm.norb,gatm.ncfg,gatm.ncfg), dtype=np.float)
    hbsn = np.zeros((gatm.ncfg,gatm.ncfg), dtype=np.complex)

    for icfg in range(gatm.ncfg):
        code = format(icfg,'0'+str(gatm.norb)+'b')
        for iorb in range(gatm.norb):
            if code[iorb] == '0' :
                mfck[icfg] *= (1.0-gatm.nloc[iorb])
            else :
                mfck[icfg] *= gatm.nloc[iorb]

    fmat = np.dot( np.linalg.inv(gatm.avec),np.dot(np.diag(mfck),gatm.avec) )
    # if not(chkrnd(fmat)) : 
    #     print fmat.real
    #     print
    #     sys.exit(" <G|m|G> should be REAL & DIAGONAL in this code !\n")
    mgmm = np.diag( fmat ).real
    del fmat

    # N = Tr\psi\psi n
    for iorb in range(gatm.norb):
        for jcfg in range(gatm.ncfg):
            code = format(jcfg,'0'+str(gatm.norb)+'b')
            if code[iorb] == '0' : continue
            for icfg in range(gatm.ncfg):
                zz = np.absolute(gatm.avec[jcfg,icfg])
                if zz < 1e-8 : continue
                meta[iorb,icfg] += zz*zz*float(code[iorb])*mfck[jcfg]

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
        hbsn[icfg,icfg] = mgmm[icfg] * gatm.aeig[icfg]

    for iorb in range(gatm.norb):
        for icfg in range(gatm.ncfg):
            for jcfg in range(gatm.ncfg):
                hbsn[icfg,jcfg] += gatm.dedr[iorb]*fadd[iorb,icfg,jcfg]*frmv[iorb,jcfg,icfg]
                hbsn[jcfg,icfg] += gatm.dedr[iorb]*fadd[iorb,icfg,jcfg]*frmv[iorb,jcfg,icfg]

    for icfg in range(gatm.ncfg):
        for jcfg in range(gatm.ncfg):
            hbsn[icfg,jcfg] /= np.sqrt(mgmm[icfg]*mgmm[jcfg])

    for icfg in range(gatm.ncfg):
        meta[:,icfg] /= mgmm[icfg]

    ini   = np.repeat(gatm.uj[0]*(np.sum(gatm.nloc)-0.5), gatm.norb)
    dltn  = lambda lamda: gwl_gennloc(lamda, gatm, hbsn, meta)
    rslt  = optimize.root(dltn, ini, method='hybr', tol=1e-8)
    lamda = rslt.x
    if not(rslt.success) : 
        sys.exit(" gwl_core loop does not converged !\n")

    # Ground state wave function
    gwf   = gwl_gwf(lamda, gatm, hbsn, meta)
    for icfg in range(gatm.ncfg):
        gwf[icfg] /= np.sqrt(mgmm[icfg])

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

    gatm.qre = gatm.symm.symmetrize(gatm.qre)
    gatm.elm = gatm.symm.symmetrize(gatm.elm)

    print ' gatm.dedr :'
    print gatm.dedr
    print ' gatm.elm :'
    print gatm.elm
    print ' gatm.qre :'
    print gatm.qre
    print ' gatm.eimp :'
    print gatm.eimp
    print ' gatm.nloc :'
    print gatm.nloc

    return

def gwl_gennloc(lamda, gatm, hbsn, meta):

    from scipy import linalg

    hnew = np.copy(hbsn)
    for icfg in range(gatm.ncfg):
        ceff = 0.0
        for iorb in range(gatm.norb):
            ceff += -lamda[iorb] * meta[iorb,icfg]
        hnew[icfg,icfg] += ceff

    eigs, evec = linalg.eigh(hnew)

    nnew = np.zeros(gatm.norb, dtype=np.float)
    for iorb in range(gatm.norb):
        for icfg in range(gatm.ncfg):
            nnew[iorb] += (meta[iorb,icfg]*evec[icfg,0]**2.0).real

    # make new local occupation symmetrized
    # nnew = gatm.symm.symmetrize(nnew)

    return nnew-gatm.nloc

def gwl_gwf(lamda, gatm, hbsn, meta):

    from scipy import linalg

    hnew = np.copy(hbsn)
    for icfg in range(gatm.ncfg):
        ceff = 0.0
        for iorb in range(gatm.norb):
            ceff += -lamda[iorb] * meta[iorb,icfg]
        hnew[icfg,icfg] += ceff

    eigs, evec = linalg.eigh(hnew)

    return evec[:,0]

