
import numpy as np
from datetime import datetime

def gwl_interface():

    from gwl_read import read_dft, gen_bethe_kwt
    from gwl_symm import symmetry
    from gwl_constants import isym
    from gwl_tools import chknloc

    np.set_printoptions(precision=3,linewidth=160,suppress=True,\
        formatter={'float': '{: 0.5f}'.format})

    print 
    print " Pygwl begin @ ", datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print 

    error = 0

    # read eigenvalue and eigenvector
    gatm = read_dft('output.enk', 'output.ovlp')
    # gatm.kwt = gen_bethe_kwt(gatm.eigs)
    print " Read DFT input DONE !"

    # initialize symmetry
    gatm.symm = symmetry(np.array(isym))
    
    # search chemical potential
    gatm.mu = gatm.searchmu(gatm.eigs)
    print " Chemical potential from DFT :", ("%10.5f") %(gatm.mu)
    gatm.eigs -= gatm.mu

    # generate impurity energy level
    gatm.eimp = gatm.geneimp(gatm.eigs, gatm.smat)
    gatm.eimp = gatm.symm.symmetrize(gatm.eimp)
    print " Impurity level from DFT :"
    print gatm.eimp
    # generate local density matrix
    gatm.nloc = gatm.gennloc(gatm.eigs, gatm.smat)
    gatm.nloc = chknloc(gatm.nloc)
    gatm.nloc = gatm.symm.symmetrize(gatm.nloc)
    print " Local particle number from DFT :"
    print gatm.nloc
    print 
    
    # atom eigenstate & eigen value
    gatm.eigenstate()

    # Enter Gutzwiller main loop
    gwl_mainloop(gatm)

    # Dump Gutzwiller results
    gwl_dumprslt(gatm)

    return error


def gwl_mainloop(gatm):

    import sys
    from scipy import optimize
    from gwl_constants import mixer

    ini  = np.append(gatm.elm,gatm.qre)

    if   mixer.keys()[0] == 'broyden1' :
        rslt = optimize.root(gatm.outerloop,ini,method='broyden1',\
        options={'maxiter':mixer.values()[0][0],'ftol':mixer.values()[0][1],\
        'jac_options':{'reduction_method':'simple'}})
    elif mixer.keys()[0] == 'linearmixing' :
        rslt = optimize.root(gatm.outerloop,ini,method='linearmixing',
        options={'maxiter':mixer.values()[0][0],'ftol':mixer.values()[0][1]})
    elif mixer.keys()[0] == 'lm' :
        rslt = optimize.root(gatm.outerloop,ini,method='lm',
        options={'maxiter':mixer.values()[0][0],'ftol':mixer.values()[0][1]})

    if not(rslt.success) :
        print
        print rslt
        print  " Pygwl main does not converge !\n Try another mixer method !\n"

    gatm.elm = rslt.x[:gatm.norb]
    gatm.qre = rslt.x[gatm.norb:]

    return

def gwl_dumprslt(gatm):

    print ' Final results : '
    print ' Elm :'
    print gatm.elm
    print ' Qre :'
    print gatm.qre
    print ' Nloc :'
    print gatm.nloc
    print ' Z :'
    print (gatm.qre)**2.0
    print 
    print ' Pygwl finished @ ', datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print
    print


