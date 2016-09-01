#!/usr/bin/python
# -*- coding: utf-8 -*-
from optparse import OptionParser
import numpy as np


# Datatype for rpoint & kpoint
ptype = np.dtype({
        'names':['name','vec','wt','dist'],
        'formats':['S10','3f','f','f']})

# status bar
def view_bar(num=1, sum=100, bar_word="="):

    import sys,os

    rate = float(num) / float(sum)
    rate_num = int(rate * 100)
    print '\r %d%% :' %(rate_num),
    for i in range(0, rate_num/2):
        os.write(1, bar_word)
    sys.stdout.flush()

def command_argument(): 
    # transpose command argument
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage,version="%prog 0.1")

    parser.add_option("-w","--wfile",action="store",dest="wfile",type="string", \
                          help="read Wannier Function from FILENAME", \
                          default="wannier90_hr.dat")
    parser.add_option("-k","--kflie",action="store",dest="kfile",type="string", \
                          help="read K-Path from FILENAME", default="wann.klist_band")
    parser.add_option("-p","--psoc",action="store",dest="plam",type="float",\
                          help="SOC for d orbitals", default="0.")
    parser.add_option("-d","--dsoc",action="store",dest="dlam",type="float",\
                          help="SOC for d orbitals", default="0.")
    parser.add_option("-c","--udc",action="store",dest="udc",type="float",\
                         help="Double counting", default="0.")
    parser.add_option("-z","--nzlay",action="store",dest="zlay",type="int",\
                          help="Number of layers in z direction", default="0")
    parser.add_option("-e","--elm",action="store",dest="felm",type="string",\
                          help="GW Elm results", default="output.Elm")
    parser.add_option("-r","--rmtrx",action="store",dest="frmt",type="string",\
                          help="GW R-Matrix results", default="output.Rmtrx")
    parser.add_option("-m","--mu",action="store",dest="mu",type="float",\
                          help="Chemical potential", default="0.0")
    parser.add_option("-n","--renorm",action="store_true",dest="renm",\
                          help="Renormalization")
    # parser.add_option("-no","--norb",dest="norb",type="int", \
    #                       help="How many kinds of orbitals", default=2)

    (options, args) = parser.parse_args()

    return options

def getham_w90(wfile):

    import string

    # Read Wannier Functions
    #================================================================================#
    wan_file = open(wfile, 'r')

    title = wan_file.readline()
    nwan = string.atoi(wan_file.readline())
    nrpt = string.atoi(wan_file.readline())

    # defile arrays
    rpt = np.zeros(nrpt,dtype=ptype)

    nline = nrpt/15
    if nrpt%15 != 0:
        nline += 1

    for i in range(0, nline):
        line = wan_file.readline()
        rpt[i*15:i*15+15]["wt"]=map(float, line.split())

    ham = np.zeros((nrpt,nwan,nwan),dtype=np.complex)
    irpt = 0
    onsite = 0
    for line in wan_file:
        rpt[irpt]['vec'][0:3] = line.split()[0:3]
        if ( rpt[irpt]['vec'][0] == 0 and rpt[irpt]['vec'][1] ==0 and rpt[irpt]['vec'][2] == 0 ): onsite = irpt
        # ham.append( string.atof(line.split()[5]) + \
        #                     1j*string.atof(line.split()[6]) )
        iwan = string.atoi(line.split()[3]) - 1
        jwan = string.atoi(line.split()[4]) - 1
        ham[irpt,iwan,jwan] = string.atof(line.split()[5]) + 1j*string.atof(line.split()[6])
        if (string.atoi(line.split()[3]) == nwan) and (string.atoi(line.split()[4]) == nwan) :
            irpt += 1

    # close wann file
    wan_file.close()
    print " Reading Wannier functions done !"

    # for iatm in range(4):
    #     for iwan in range(5):
    #         ham[onsite,iatm*12+iwan,iatm*12+iwan]     -= 0.04
    # for iatm in range(4):
    #     for iwan in range(7):
    #         ham[onsite,5+iatm*12+iwan,5+iatm*12+iwan] += 0.05
    # for iatm in range(4):
    #     for iwan in range(5):
    #         ham[onsite,48+iatm*5+iwan,48+iatm*5+iwan] -= 0.03

    return nrpt, onsite, rpt, nwan, ham


def defkpt():

    nkpt = 5
    kpt  = np.zeros(nkpt,dtype=ptype)
    kpt[0]['vec'][:] = [0.000, 0.000, 0.000]
    kpt[2]['vec'][:] = [0.500, 0.000, 0.000]
    kpt[2]['vec'][:] = [0.000, 0.500, 0.000]
    kpt[1]['vec'][:] = [0.000, 0.000, 0.500]
    kpt[3]['vec'][:] = [0.500, 0.000, 0.500]
    kpt[4]['vec'][:] = [0.500, 0.500, 0.500]

    return nkpt, kpt

def genkpt():

    from scipy import linalg

    dist = 0.004

    data = np.loadtxt('hkpt.ini')
    lvec = np.array(data[0:3,:],dtype=np.float).reshape( 3,3)
    hkpt = np.array(data[3: ,:],dtype=np.float).reshape(-1,3)

    kpts = []
    for ihkp in range(hkpt.shape[0]-1):

        kpt1 = np.dot(hkpt[ihkp,:],lvec)
        kpt2 = np.dot(hkpt[ihkp+1,:],lvec)

        dlt1 = dist * ( kpt2-kpt1 ) / linalg.norm(kpt2-kpt1)

        icount = 1
        dlt2 = 0.0
        kpt = kpt1
        while dlt2 < linalg.norm(kpt2-kpt1):
            kpts.append( list(np.dot(kpt, linalg.inv(lvec))) )
            kpt = icount * dlt1 + kpt1
            dlt2 = linalg.norm(kpt - kpt1)
            icount += 1

    # add the last kpoint to the list
    kpts.append(list(hkpt[-1,:]))

    kpts = np.array(kpts,dtype=np.float).reshape(-1,3)
    nkpt = kpts.shape[0]

    kpt = np.zeros(nkpt,dtype=ptype)

    dist = 0.0
    for ikpt in range(nkpt):

        if ikpt == 0 :
            dist = 0.0
        else:
            dist += linalg.norm( np.dot( (kpts[ikpt,:] - kpts[ikpt-1,:]), lvec ) )
            # dist += linalg.norm( kpts[ikpt,:] - kpts[ikpt-1,:] )

        kpt[ikpt]['vec'][:] = kpts[ikpt,:]
        kpt[ikpt]['dist']   = dist


    klist_file = open("klist.dat","w")
    for ikpt in range(nkpt):
        print >> klist_file, ("%5d%10.5f%10.5f%10.5f%10.5f") \
        %(ikpt+1,kpt[ikpt]['vec'][0],kpt[ikpt]['vec'][1],kpt[ikpt]['vec'][2],kpt[ikpt]['dist'])
    klist_file.close()

    return kpt


def genkpts():

    nkpx = 101
    nkpy = 101
    nkpt = nkpx * nkpy

    hkp = np.zeros(2, dtype=ptype)
    dlt = np.zeros(2, dtype=np.float)
    hkp[0]['vec'][:] = [-0.1,-0.1, 0.0]
    hkp[1]['vec'][:] = [ 0.1, 0.1, 0.0]

    dlt[0] = ( hkp[1]['vec'][0] - hkp[0]['vec'][0] ) / (nkpx-1)
    dlt[1] = ( hkp[1]['vec'][1] - hkp[0]['vec'][1] ) / (nkpy-1)

    kpt = np.zeros(nkpt,dtype=ptype)
    ikpt = 0

    eta = 1.0*np.pi/3.0

    for ikps in range(0, nkpy):
        for jkps in range(0, nkpx):

            # kpt[ikpt]['vec'][0] = hkp[0]['vec'][0] + dlt[0] * jkps
            # kpt[ikpt]['vec'][1] = hkp[0]['vec'][1] + dlt[1] * ikps

            xx = hkp[0]['vec'][0] + dlt[0] * jkps
            yy = hkp[0]['vec'][1] + dlt[1] * ikps

            kpt[ikpt]['vec'][0] = xx - yy / np.tan(eta)
            kpt[ikpt]['vec'][1] = yy / np.sin(eta)

            if ikpt > 0:
                kpt[ikpt]['dist'] = kpt[ikpt-1]['dist'] + np.sqrt( \
                ( kpt[ikpt]['vec'][0] - kpt[ikpt-1]['vec'][0] )**2 + \
                ( kpt[ikpt]['vec'][1] - kpt[ikpt-1]['vec'][1] )**2 + \
                ( kpt[ikpt]['vec'][2] - kpt[ikpt-1]['vec'][2] )**2 )

            ikpt += 1

    return nkpt, kpt

def gen_fullkpt():

    nkpx = 10
    nkpy = 10
    nkpz = 10
    nkpt = nkpx * nkpy * nkpz

    hkp = np.zeros(2, dtype=ptype)
    dlt = np.zeros(3, dtype=np.float)
    hkp[0]['vec'][:] = [ 0.0, 0.0, 0.0]
    hkp[1]['vec'][:] = [ 1.0, 1.0, 1.0]

    dlt[0] = ( hkp[1]['vec'][0] - hkp[0]['vec'][0] ) / (nkpx)
    dlt[1] = ( hkp[1]['vec'][1] - hkp[0]['vec'][1] ) / (nkpy)
    dlt[2] = ( hkp[1]['vec'][2] - hkp[0]['vec'][2] ) / (nkpz)

    kpt = np.zeros(nkpt,dtype=ptype)
    ikpt = 0

    for ikps in range(nkpy):
        for jkps in range(nkpx):
            for kkps in range(nkpz):

                kpt[ikpt]['vec'][0] = hkp[0]['vec'][0] + dlt[0] * ikps
                kpt[ikpt]['vec'][1] = hkp[0]['vec'][1] + dlt[1] * jkps
                kpt[ikpt]['vec'][2] = hkp[0]['vec'][2] + dlt[2] * kkps

                if ikpt > 0: 
                    kpt[ikpt]['dist'] = kpt[ikpt-1]['dist'] + np.sqrt( \
                    ( kpt[ikpt]['vec'][0] - kpt[ikpt-1]['vec'][0] )**2 + \
                    ( kpt[ikpt]['vec'][1] - kpt[ikpt-1]['vec'][1] )**2 + \
                    ( kpt[ikpt]['vec'][2] - kpt[ikpt-1]['vec'][2] )**2 )

                ikpt += 1

    kfile = open('klist.dat','w')
    for ikpt in range(nkpt):
        print >> kfile, "%5d%10.5f%10.5f%10.5f%10.5f" %(ikpt+1,kpt[ikpt]['vec'][0],kpt[ikpt]['vec'][1],\
            kpt[ikpt]['vec'][2],kpt[ikpt]['dist'])
    kfile.close()


    return kpt

def change_basis(nwan,nrpt,onsite,rpt,ham):
    hamr = np.zeros((nrpt,2*nwan,2*nwan),dtype=np.complex)
    for irpt in range(nrpt):
        hamr[irpt,0:nwan*2:2,0:nwan*2:2] = ham[irpt,:,:]/rpt[irpt]['wt']
        hamr[irpt,1:nwan*2:2,1:nwan*2:2] = ham[irpt,:,:]/rpt[irpt]['wt']
        if irpt == onsite : print np.diag(hamr[irpt,:,:]).real

    return hamr

# For SOC
# def change_basis(nwan,nrpt,onsite,rpt,ham):

#     # import spharm
#     from tran import tmat_c2r
#     from soc import atom_hsoc
#     # from cgmat  import clebsch_gordan
#     from scipy import linalg

#     np.set_printoptions(precision=3,suppress=True,linewidth=160)

#     ced = 0.05
#     cef = 0.09
#     nid = 0.01
#     snp = 0.01

#     # nwan2 = nwan
#     nwan2 = nwan * 2

#     # Tranform ham to include spin
#     hamr = np.zeros((nrpt,nwan2,nwan2),dtype=np.complex)

#     pc2r = tmat_c2r('p',True)
#     dc2r = tmat_c2r('d',True)
#     fc2r = tmat_c2r('f',True)

#     socmat = np.zeros((nwan2,nwan2),dtype=np.complex)
#     # Ce d & f
#     for iatm in range(4):
#         socmat[iatm*24:iatm*24+10, iatm*24:iatm*24+10] = np.dot(linalg.inv(dc2r), np.dot(atom_hsoc('d', ced), dc2r))
#         socmat[iatm*24+10:(iatm+1)*24, iatm*24+10:(iatm+1)*24] = np.dot(linalg.inv(fc2r), np.dot(atom_hsoc('f',cef), fc2r))
#     # Ni d
#     for iatm in range(4):
#         socmat[96+iatm*10:96+(iatm+1)*10, 96+iatm*10:96+(iatm+1)*10] = np.dot(linalg.inv(dc2r), np.dot(atom_hsoc('d', nid), dc2r))
#     # Sn p
#     for iatm in range(4):
#         socmat[136+iatm*6:136+(iatm+1)*6, 136+iatm*6:136+(iatm+1)*6] = np.dot(linalg.inv(pc2r), np.dot(atom_hsoc('p', snp), pc2r))

#     for irpt in range(0, nrpt):
#         hamr[irpt,0:nwan2:2,0:nwan2:2] = ham[irpt,:,:] / rpt[irpt]['wt']
#         hamr[irpt,1:nwan2:2,1:nwan2:2] = ham[irpt,:,:] / rpt[irpt]['wt']

#         if irpt == onsite : 
#             hamr[irpt,:,:] = hamr[irpt,:,:] + socmat

#     return hamr


def pfm_hamk(rpt, kpt, hamr, mu):

    from scipy import linalg
    import wann_tran

    nrpt = hamr.shape[0]
    nwan = hamr.shape[1]
    nkpt = kpt.shape[0]

    eig = np.zeros((nkpt,nwan),dtype=np.float)
    vec = np.zeros((nkpt,nwan,nwan),dtype=np.complex)

    for ikpt in range(0,nkpt):

        view_bar(ikpt,nkpt)

        # Ser Hamiltonian
        hamk = wann_tran.trans_hamr(kpt[ikpt]['vec'][:],rpt[:]['vec'][:],hamr)

        # Diagonal Hamiltonian
        eig[ikpt,:],vec[ikpt,:,:] = linalg.eigh(hamk)
        eig[ikpt,:] -= mu

    print 

    return eig, vec

def dump_fatband(kpt, eig, vec):

    nkpt = vec.shape[0]
    nbnd = vec.shape[2]
    norb = vec.shape[1]
    prob = np.zeros((nkpt,nbnd),dtype=np.float)

    for ikpt in range(nkpt):
        prob[ikpt,:] = np.diag(np.dot(vec[ikpt,:10,:].transpose().conj(),vec[ikpt,:10,:])).real

    feig = open("wanbnd.dat","w")
    for ibnd in range(0, nbnd):
        for ikpt in range(0, nkpt):
            print >> feig, "%10.5f   %10.5f   %10.5f" \
            %(kpt[ikpt]['dist'], eig[ikpt,ibnd],prob[ikpt,ibnd])
        print >> feig
        print >> feig
    feig.close()

def dump_output( eig, smtrx ):

    nkpt = smtrx.shape[0]
    nbnd = smtrx.shape[2]
    norb = 10
    natm = 1
    ntot = 20.0

    output = open('output.enk','w')
    print >> output, "%10d%-40s" %(nkpt, ' : number of k-points')
    print >> output, "%10d%-40s" %(nbnd, ' : number of bands')
    print >> output, "%10d%-40s" %(norb, ' : number of correlated orbitals')
    print >> output, "%10d%-40s" %(natm, ' : number of correlated atoms')
    print >> output, "%10.5f%-40s" %(ntot, ' : number of total electrons')
    for ikpt in range(nkpt):
        print >> output, "%10s%10d" %(' # ikpt ', ikpt+1)
        for ibnd in range(nbnd):
            print >> output, "%5d%20.10f" %(ibnd+1, eig[ikpt,ibnd])
        print >> output
    output.close()

    output = open('output.ovlp','w')
    print >> output, "%10d%-40s" %(nkpt, ' : number of k-points')
    print >> output, "%10d%-40s" %(nbnd, ' : number of bands')
    print >> output, "%10d%-40s" %(norb, ' : number of correlated orbitals')
    print >> output, "%10d%-40s" %(natm, ' : number of correlated atoms')
    print >> output, "%10.4f%-40s" %(ntot, ' : number of total electrons')
    for ikpt in range(nkpt):
        print >> output, "%10s%10d" %(' # ikpt ', ikpt+1)
        for iorb in range(norb):
            for ibnd in range(nbnd):
                print >> output, "%5d%5d%20.10f%20.10f" %(iorb+1, ibnd+1, \
                  smtrx[ikpt,iorb,ibnd].real, smtrx[ikpt,iorb,ibnd].imag)
        print >> output
    output.close()

def dump_dmft(eig,smtrx):

    nkpt = smtrx.shape[0]
    nbnd = smtrx.shape[2]
    norb = smtrx.shape[1]
    natm = 1
    ntot = 7.0

    # Print eigen values 
    feig = open("eig.dat","w")
    print >> feig, "%5d%5d%5d%3d%3d%50.30s" %(nkpt,1,norb,3,2,"# nkpt, nsymop, norb, L, nspin")
    print >> feig, "%5d%5d%5d%16.9f%40.30s" %(1,nbnd,nbnd,ntot,"# nemin, nemax, nbands, ntotal")
    print >> feig, "%16.9f%29.10s" %(0.0,"# Mu")
    print >> feig
    for ikpt in range(0, nkpt):
        print >> feig, "%5d%20.12f%31.20s" %(ikpt+1, 1.0/np.float(nkpt)," # ikpt, k-weight")
        for ibnd in range(0, nbnd):
            print >> feig, "%5d%20.12f" %(ibnd+1, eig[ikpt,ibnd])
        print >> feig
        print >> feig
    feig.close()
    
    fprj = open("udmft.dat","w")
    print >> fprj, "%5d%5d%5d%3d%3d%50.32s" %(nkpt,1,norb,3,2,"# nkpt, nsymop, norb, L, nspin")
    print >> fprj, "%5d%5d%5d%16.9f%40.30s" %(1,nbnd,nbnd,ntot,"# nemin, nemax, nbands, ntotal")
    print >> fprj
    for ikpt in range(0,nkpt):
        print >> fprj, "%5d%20.12f%31.20s" %(ikpt+1, 1.0/np.float(nkpt),"# ikpt, kweight")
        print >> fprj, "%5d%44.20s" %(1,"# isymop")
        for iorb in range(0,norb):
            for ibnd in range(0, nbnd):
                print >> fprj, "%5d%5d%20.12f%20.12f" %(ibnd+1,iorb+1, \
                    smtrx[ikpt,iorb,ibnd].real, -smtrx[ikpt,iorb,ibnd].imag )
        print >> fprj
        print >> fprj
    fprj.close()



if __name__ == "__main__" :

    options = command_argument()

    nrpt, onsite, rpt, nwan, ham = getham_w90(options.wfile)
    kpt = genkpt()
    # nkpt, kpt = defkpt()
    # nkpt, kpt = genkpts()
    # kpt = gen_fullkpt()

    hamr = change_basis(nwan,nrpt,onsite,rpt,ham)

    eig, vec   = pfm_hamk(rpt, kpt, hamr, options.mu)
    dump_fatband( kpt, eig, vec)
    # dump_output(eig,vec)

    # omega, rho       = pfm_srho(rpt, kpt, hamr, options.mu)
    # dump_rho( kpt, options.mu, omega, rho )

    # eig, smtrx       = pfm_hams(options.zlay, rpt, kpt, hamr, options.mu)
    # dump_fatband( kpt, eig, smtrx )

    # rho, spin        = pfm_hams_fs(rpt,kpt,hamr,options.mu)
    # dump_fs(kpt, rho, spin)

    # def_parity(kpt, eig,smtrx)



