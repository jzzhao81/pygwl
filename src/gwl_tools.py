
import numpy as np

# If the given matrix is not real & diagonal, return False
def chkrnd(amat):
    return True
    # if (np.absolute(amat-np.diag(np.diag(amat)))>1e-2).any() : return False
    # if (np.diag(amat).imag > 1e-2).any() : return False
    # return True

# If the given matrix is not Hermite, return False
def chkherm(amat):
    if ((amat.transpose().conj()-amat)>1e-2).any() : return False
    return True

# If the nloc is too small(n<1e-5) or too large((1-n)<1e-5), correct it
def chknloc(nloc):
    norb = nloc.shape[0]
    nnew = np.zeros(norb,dtype=np.float)
    for iorb in range(norb):
        if nloc[iorb] < 1e-5 : nnew[iorb] = 1e-5; continue
        if (1.0-nloc[iorb]) < 1e-5 : nnew[iorb] = 1.0-1e-5 ; continue
        nnew[iorb] = nloc[iorb]
    return nnew

def prjloc(imat, smat):
    omat = np.dot( np.dot(smat, imat), smat.transpose().conj() )
    return omat

def prjblh(imat, smat):
    omat = np.dot( smat.transpose().conj(), np.dot(imat, smat) )
    return omat

def genfdis(eigs):
    from gwl_constants import smear
    nkpt = eigs.shape[0]; nbnd = eigs.shape[1]
    fdis = np.zeros((nkpt,nbnd),dtype=np.float)
    for ikpt in range(nkpt):
        for ibnd in range(nbnd):
            fdis[ikpt,ibnd] = disvalue(smear.keys()[0],eigs[ikpt,ibnd],smear.values()[0])
    return fdis

def disvalue(method,ene,value):
    if   method == 'fermi' :
        rslt = fermidis(ene,value)
    elif method == 'gauss' :
        rslt = gaussdis(ene,value)
    elif method == 'mp' :
        rslt = mpdis(ene,value)
    else :
        sys.exit(" Error method in disvalue ", method)
    return rslt

def fermidis(ene,beta):
    if ene <= 0.0 :
        dist = 1.0/(np.exp(ene*beta)+1.0)
    else :
        dist = np.exp(-1.0*ene*beta)/(1.0+np.exp(-1.0*ene*beta))
    return dist

def mpfun(ene,mporder):
    from scipy.special import eval_hermite
    dsum = 0.0
    for nn in range(mporder):
        an = (-1.0)**float(nn)/(np.math.factorial(nn)*4.0**float(nn)*np.sqrt(np.pi))
        dsum += an*eval_hermite(2*nn,ene)*np.exp(-ene**2.0)
    return dsum

def mpdis(ene,mporder):
    from scipy import integrate
    fun = lambda x : mpfun(x,mporder)
    dist = integrate.romberg(fun,-5.0,ene)
    return 1.0-dist

def gaussdis(ene,fsgm):
    from scipy.special import ndtr
    return 1.0 - ndtr(ene/fsgm)

def searchmu(eigs, ntot, kwt):
    from scipy.optimize import brentq
    f = lambda x : genntot(eigs-x, kwt)-ntot
    mu,r  = brentq(f,eigs.min(),eigs.max(),maxiter=200,full_output=True)
    if not(r.converged) : print " Failed to search mu ! "
    return mu

def genntot(eigs, kwt):
    nkpt = eigs.shape[0]; nbnd = eigs.shape[1]
    ntot = 0.0
    fdis = genfdis(eigs)
    for ikpt in range(nkpt):
        for ibnd in range(nbnd):
            ntot += kwt[ikpt]*fdis[ikpt,ibnd]
    return ntot
