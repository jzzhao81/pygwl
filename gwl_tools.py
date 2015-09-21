
import numpy as np

def chkrnd(amat):
    if (np.absolute(amat-np.diag(np.diag(amat)))>1e-1).any() : return False
    if (np.diag(amat).imag > 1e-1).any() : return False
    return True

def chkherm(amat):
    if ((amat.transpose().conj()-amat)>1e-1).any() : return False
    return True
