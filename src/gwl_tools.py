
import numpy as np

# If the given matrix is not real & diagonal, return False
def chkrnd(amat):
    if (np.absolute(amat-np.diag(np.diag(amat)))>1e-2).any() : return False
    if (np.diag(amat).imag > 1e-2).any() : return False
    return True

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