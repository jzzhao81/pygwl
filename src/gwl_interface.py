
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

    error = 0

    # read eigenvalue and eigenvector
    gatm = read_dft('output.enk', 'output.ovlp')
    # gatm.kwt = gen_bethe_kwt(gatm.eigs)
    print " Read DFT input DONE !"
    natm = len(gatm)

    # Loop for atoms
    for iatm in range(natm):
        # initialize symmetry
        gatm[iatm].osym = symmetry(np.array(isym[iatm]))

    for iatm in range(natm):

        # generate impurity energy level
        gatm[iatm].eimp = gatm[iatm].geneimp(gatm[iatm].eigs, gatm[iatm].smat)
        gatm[iatm].eimp = gatm[iatm].osym.symmetrize(gatm[iatm].eimp)
        # generate local density matrix
        gatm[iatm].nloc = gatm[iatm].gennloc(gatm[iatm].eigs, gatm[iatm].smat)
        gatm[iatm].nloc = chknloc(gatm[iatm].nloc)
        gatm[iatm].nloc = gatm[iatm].osym.symmetrize(gatm[iatm].nloc)
        # atom eigenstate & eigen value
        gatm[iatm].eigenstate()
        # determine degenerate atomic configuration
        gatm[iatm].degenerate()

        # Double counting
        gatm[iatm].udc  = gatm[iatm].genudc(gatm[iatm].nloc)
        gatm[iatm].elm  = np.copy(gatm[iatm].udc)


    print " Impurity level from DFT :"
    for iatm in range(natm): print gatm[iatm].eimp
    print " Local particle number from DFT :", ("%10.5f") %(np.sum(gatm[iatm].nloc))
    for iatm in range(natm): print gatm[iatm].nloc
    print " Double Counting :"
    for iatm in range(natm): print gatm[iatm].udc

    # Enter Gutzwiller main loop
    gatm = gwl_mainloop(gatm)

    # Dump Gutzwiller results
    gwl_dumprslt(gatm)

    return error


def gwl_mainloop(gatm):

    import sys
    from scipy import optimize
    from gwl_constants import mixer

    # number of atoms
    natm = len(gatm)

    ini  = []
    for iatm in range(natm):
        ini.append(np.append(gatm[iatm].elm,gatm[iatm].qre))
    ini = np.array(ini)
    print

    outr = lambda enq : gwl_outerloop(enq, gatm)
    if   mixer.keys()[0] == 'broyden1' :
        rslt = optimize.root(outr,ini,method='broyden1',\
        options={'maxiter':mixer.values()[0][0],'ftol':mixer.values()[0][1],\
        'jac_options':{'reduction_method':'simple'}})
    elif mixer.keys()[0] == 'linearmixing' :
        rslt = optimize.root(outr,ini,method='linearmixing',
        options={'maxiter':mixer.values()[0][0],'ftol':mixer.values()[0][1]})
    elif mixer.keys()[0] == 'lm' :
        rslt = optimize.root(outr,ini,method='lm',
        options={'maxiter':mixer.values()[0][0],'ftol':mixer.values()[0][1]})
    else :
        sys.exit(' Mixer error :', mixer.keys()[0] )

    if not(rslt.success) :
        print
        print rslt
        print  " Pygwl main does not converge !\n Try another mixer method !\n"


    for iatm in range(natm):
        gatm[iatm].elm = rslt.x[iatm][:gatm[iatm].norb]
        gatm[iatm].qre = rslt.x[iatm][gatm[iatm].norb:]

    return gatm

def gwl_outerloop(inp, gatm):

    from gwl_ksum import gwl_ksum
    from gwl_core import gwl_core

    # number of atoms from input
    natm = len(gatm)

    # Elm & Qvec from last loop
    eold = []; qold = []

    print
    print " Iter  :", gatm[0].iter
    for iatm in range(natm):
        gatm[iatm].elm = np.copy(inp[iatm][:gatm[iatm].norb])
        gatm[iatm].qre = np.copy(inp[iatm][gatm[iatm].norb:])
        eold.append( np.copy(gatm[iatm].elm) )
        qold.append( np.copy(gatm[iatm].qre) )

    # k summation
    gatm = gwl_ksum(gatm)

    # inner loop
    diff = []
    for iatm in range(natm):

        print ' Inner loop for atom :', iatm+1
        gatm[iatm].gwl_core()

        diff = np.append(diff, np.append(gatm[iatm].elm-eold[iatm],gatm[iatm].qre-qold[iatm]))

        gatm[iatm].iter += 1
        print " diff :"
        print gatm[iatm].elm-eold[iatm]
        print gatm[iatm].qre-qold[iatm]

    return diff

def gwl_dumprslt(gatm):

    # number of correlated atoms
    natm = len(gatm)

    print ' Final results : '
    print ' Elm :'
    for iatm in range(natm):
        print gatm[iatm].elm
    print ' Qre :'
    for iatm in range(natm):
        print gatm[iatm].qre
    print ' Nloc :'
    for iatm in range(natm):
        print gatm[iatm].nloc
    print ' Z :'
    for iatm in range(natm):
        print (gatm[iatm].qre)**2.0
    print
    print ' Pygwl finished @ ', datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print
    print


