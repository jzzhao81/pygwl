
import numpy as np
import sys
from scipy import linalg

def gwl_ksum(gatm):

    from gwl_tools import chkherm, chkrnd, chknloc

    eigs = np.zeros((gatm.nkpt,gatm.nbnd),dtype=np.float)
    evec = np.zeros((gatm.nkpt,gatm.nbnd,gatm.nbnd), dtype=np.complex)
    fdis = np.zeros((gatm.nkpt,gatm.nbnd),dtype=np.float)

    for ikpt in range(gatm.nkpt):

        # 1-P+Q
        imat  = np.identity(gatm.nbnd,dtype=np.complex)
        pmat  = np.dot( gatm.smat[ikpt,:,:].transpose().conj(), gatm.smat[ikpt,:,:] )
        qmat  = gatm.prjblh( np.diag(gatm.qre), gatm.smat[ikpt,:,:] )
        qmat  = imat - pmat + qmat

        # H_eff & Hop term
        heff  = gatm.prjblh( np.diag(gatm.eimp), gatm.smat[ikpt,:,:] )
        hhop  = np.diag(gatm.eigs[ikpt,:]) - heff
        heff += gatm.prjblh( np.diag(gatm.elm-gatm.udc), gatm.smat[ikpt,:,:] )
        heff += np.dot( qmat.transpose().conj(), np.dot(hhop,qmat) )

        # Check Heff is Hermite matrix, or not
        if not(chkherm(heff)): sys.exit(" Heff in gwl_ksum is not hermite !")

        # evaluate new eigen value & eigen vector
        eigs[ikpt,:], evec[ikpt,:,:] = linalg.eigh(heff)

    gatm.mu = gatm.searchmu(eigs)
    print " Chemical potential in gwl_ksum :", ("%10.5f") %(gatm.mu)
    # Generate new fermi distribution
    fdis = gatm.genfdis(eigs-gatm.mu)

    # Generate new local occupation
    nmat = np.zeros((gatm.norb,gatm.norb),dtype=np.complex)
    for ikpt in range(gatm.nkpt):
        tmat = np.dot(gatm.smat[ikpt,:,:],evec[ikpt,:,:])
        nmat += gatm.prjloc( np.diag(fdis[ikpt,:]), tmat ) * gatm.kwt[ikpt]
    if not(chkrnd(nmat)) : sys.exit(" Local density matrix is not REAL & DIAGONAL in gwl_ksum !\n")
    gatm.nloc = np.diag(nmat).real
    # make local occupation symmetrize
    gatm.nloc = chknloc(gatm.nloc)
    gatm.nloc = gatm.symm.symmetrize(gatm.nloc)
    del nmat

    # get dedr
    emat = np.zeros((gatm.norb,gatm.norb),dtype=np.complex)
    for ikpt in range(gatm.nkpt):

        # 1-P+Q
        imat = np.identity(gatm.nbnd,dtype=np.complex)
        pmat = np.dot( gatm.smat[ikpt,:,:].transpose().conj(), gatm.smat[ikpt,:,:] )
        qmat = gatm.prjblh( np.diag(gatm.qre), gatm.smat[ikpt,:,:] )
        qmat = imat - pmat + qmat

        # H_eff & Hop term
        heff  = gatm.prjblh( np.diag(gatm.eimp), gatm.smat[ikpt,:,:] )
        hhop  = np.diag(gatm.eigs[ikpt,:]) - heff

        dmat  = np.dot( np.dot(evec[ikpt,:,:],np.diag(fdis[ikpt,:])),evec[ikpt,:,:].transpose().conj() )
        pmat  = np.dot( np.dot(dmat,qmat), hhop )
        emat += np.dot( np.dot(gatm.smat[ikpt,:,:],pmat),gatm.smat[ikpt,:,:].transpose().conj() )*gatm.kwt[ikpt]
        # <G|f|G><G|0><0|(1-P+Q)|0><0|Ht|0><0|P|0><0|G><G|f|G>
        # Old style
        # for iorb in range(gatm.norb):
        #     for jorb in range(gatm.norb):
        #         pmat = np.outer( gatm.smat[ikpt,iorb,:].conj(), gatm.smat[ikpt,jorb,:] )
        #         hmat = np.dot( np.dot(qmat,hhop), pmat )
        #         dmat = np.dot( evec[ikpt,:,:], np.diag(np.sqrt(fdis[ikpt,:])) )
        #         emat[iorb,jorb] += np.dot(dmat.transpose().conj(),np.dot(hmat,dmat)).trace()*gatm.kwt[ikpt]

    if not(chkrnd(emat)) : sys.exit(" dEdR should be REAL & DIAGONAL in gwl_ksum !\n")
    gatm.dedr = np.diag(emat).real
    # make dEdR symmetrize
    # gatm.dedr = gatm.symm.symmetrize(gatm.dedr)
    del emat

    return

