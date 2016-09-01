
import numpy as np
import sys
from scipy import linalg

def gwl_ksum(gatm):

    from gwl_tools     import chkherm, chkrnd, chknloc, prjblh, prjloc, genfdis, searchmu
    from gwl_atom      import corr_atom
    from gwl_symm      import symmetry
    from gwl_constants import isym

    natm = len(gatm)
    norb = np.sum( [gatm[iatm].norb for iatm in range(natm)] )

    fatm = corr_atom(gatm[0].nkpt,gatm[0].nbnd,norb,gatm[0].ntot)

    fatm.eigs = np.copy( gatm[0].eigs )
    fatm.smat = np.copy( gatm[0].smat )
    for iatm in range(1,natm):
        fatm.smat = np.append( fatm.smat, gatm[iatm].smat, axis=1)

    fatm.elm  = []
    for iatm in range(natm):
        fatm.elm  = np.append( fatm.elm,  gatm[iatm].elm )
    fatm.qre  = []
    for iatm in range(natm):
        fatm.qre  = np.append( fatm.qre,  gatm[iatm].qre )
    fatm.eimp = []
    for iatm in range(natm):
        fatm.eimp = np.append( fatm.eimp, gatm[iatm].eimp )
    fatm.udc  = []
    for iatm in range(natm):
        fatm.udc  = np.append( fatm.udc,  gatm[iatm].udc )

    # initialize symmetry
    sym = []
    for iatm in range(natm):
        sym.append(np.array(isym[iatm]+np.repeat(np.sum([gatm[iatm].norb for iatm in range(iatm)]),gatm[iatm].norb),dtype=np.int))
    sym = np.array(sym).reshape(-1)
    fatm.osym = symmetry(np.array(sym))

    # eigen vectors
    etmp = np.zeros((fatm.nkpt, fatm.nbnd),dtype=np.float)
    vtmp = np.zeros((fatm.nkpt, fatm.nbnd, fatm.nbnd), dtype=np.complex)

    for ikpt in range(fatm.nkpt):

        # 1-P+Q
        imat  = np.identity(fatm.nbnd,dtype=np.complex)
        pmat  = np.dot( fatm.smat[ikpt,:,:].transpose().conj(), fatm.smat[ikpt,:,:] )
        qmat  = prjblh( np.diag(fatm.qre), fatm.smat[ikpt,:,:] )
        qmat  = imat - pmat + qmat

        # H_eff & Hop term
        heff  = prjblh( np.diag(fatm.eimp), fatm.smat[ikpt,:,:] )
        hhop  = np.diag(fatm.eigs[ikpt,:]) - heff
        heff += prjblh( np.diag(fatm.elm-fatm.udc), fatm.smat[ikpt,:,:] )
        heff += np.dot( qmat.transpose().conj(), np.dot(hhop,qmat) )

        # Check Heff is Hermite matrix, or not
        if not(chkherm(heff)): sys.exit(" Heff in gwl_ksum is not hermite !")

        # evaluate new eigen value & eigen vector
        etmp[ikpt,:], vtmp[ikpt,:,:] = linalg.eigh(heff)

    fatm.mu = searchmu(etmp,fatm.ntot,fatm.kwt)
    print " Chemical potential in gwl_ksum :", ("%10.5f") %(fatm.mu)

    # Generate new fermi distribution
    fdis = genfdis(etmp-fatm.mu)

    # Generate new local occupation
    nmat = np.zeros((fatm.norb,fatm.norb),dtype=np.complex)
    for ikpt in range(fatm.nkpt):
        tmat = np.dot(fatm.smat[ikpt,:,:],vtmp[ikpt,:,:])
        nmat += prjloc( np.diag(fdis[ikpt,:]), tmat ) * fatm.kwt[ikpt]
    if not(chkrnd(nmat)) : sys.exit(" Local density matrix is not REAL & DIAGONAL in gwl_ksum !\n")
    fatm.nloc = np.diag(nmat).real
    # make local occupation symmetrize
    fatm.nloc = chknloc(fatm.nloc)
    fatm.nloc = fatm.osym.symmetrize(fatm.nloc)
    del nmat

    # get dedr
    emat = np.zeros((fatm.norb,fatm.norb),dtype=np.complex)
    for ikpt in range(fatm.nkpt):

        # 1-P+Q
        imat = np.identity(fatm.nbnd,dtype=np.complex)
        pmat = np.dot( fatm.smat[ikpt,:,:].transpose().conj(), fatm.smat[ikpt,:,:] )
        qmat = prjblh( np.diag(fatm.qre), fatm.smat[ikpt,:,:] )
        qmat = imat - pmat + qmat

        # H_eff & Hop term
        heff  = prjblh( np.diag(fatm.eimp), fatm.smat[ikpt,:,:] )
        hhop  = np.diag(fatm.eigs[ikpt,:]) - heff


        dmat  = np.dot( np.dot(vtmp[ikpt,:,:],np.diag(fdis[ikpt,:])),vtmp[ikpt,:,:].transpose().conj() )
        pmat  = np.dot( np.dot(dmat,qmat), hhop )
        emat += np.dot( np.dot(fatm.smat[ikpt,:,:],pmat),fatm.smat[ikpt,:,:].transpose().conj() )*fatm.kwt[ikpt]
        # <G|f|G><G|0><0|(1-P+Q)|0><0|Ht|0><0|P|0><0|G><G|f|G>
        # Old style
        # for iorb in range(gatm.norb):
        #     for jorb in range(gatm.norb):
        #         pmat = np.outer( gatm.smat[ikpt,iorb,:].conj(), gatm.smat[ikpt,jorb,:] )
        #         hmat = np.dot( np.dot(qmat,hhop), pmat )
        #         dmat = np.dot( vtmp[ikpt,:,:], np.diag(np.sqrt(fdis[ikpt,:])) )
        #         emat[iorb,jorb] += np.dot(dmat.transpose().conj(),np.dot(hmat,dmat)).trace()*gatm.kwt[ikpt]

    if not(chkrnd(emat)) : sys.exit(" dEdR should be REAL & DIAGONAL in gwl_ksum !\n")
    fatm.dedr = np.diag(emat).real

    for iatm in range(natm):
        start = np.sum([gatm[ii].norb for ii in range(iatm)],dtype=np.int)
        stop  = start + gatm[iatm].norb
        gatm[iatm].dedr = fatm.dedr[start:stop]
        # make dEdR symmetrize
        # gatm[iatm].dedr = gatm[iatm].symm.symmetrize(gatm[iatm].dedr)

    del emat

    return gatm

