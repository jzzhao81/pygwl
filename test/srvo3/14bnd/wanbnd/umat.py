# -*- coding: utf-8 -*-
"""
Created on Thu Nov 06 18:24:49 2014

@author: 建洲
"""

import numpy as np

np.set_printoptions(precision=4,suppress=True,linewidth=160)

class corr_atom:

    def __init__(self, nkpt=1, nbnd=1, norb=1, lmom=0, nspn=2, ntot=0.0):

        self.nkpt = nkpt
        self.nbnd = nbnd
        self.norb = norb
        self.ntot = ntot
        self.lmom = lmom
        self.nspn = nspn

        self.smat = np.zeros((self.nkpt,self.nbnd,self.norb), dtype=np.complex)
        self.nloc = np.zeros((self.norb), dtype=np.float)
        self.eimp = np.zeros((self.norb,self.norb), dtype=np.complex)
        self.eigs = np.zeros((self.nkpt,self.nbnd), dtype=np.float)
        self.kwt  = np.zeros(self.nkpt, dtype=np.float)

    def geneimp(self):

        self.eimp = 0.0
        for ikpt in range(self.nkpt):
            ekk = np.zeros((self.nbnd,self.nbnd),dtype=np.complex)
            np.fill_diagonal( ekk, self.eigs[ikpt,:] )
            self.eimp += ( prjloc( ekk, self.smat[ikpt,:,:] ) * self.kwt[ikpt] )

        return

    def gennloc(self):
        self.nloc = 0.0
        for ikpt in range(self.nkpt):
            fermi = np.zeros((self.nbnd,self.nbnd), dtype=np.complex)
            for ibnd in range(self.nbnd):
                if self.eigs[ikpt,ibnd] < 0.0 :
                    fermi[ibnd,ibnd] = 1.0 + 1j*0.0
                else:
                    fermi[ibnd,ibnd] = 0.0 + 1j*0.0
            self.nloc += np.diag( prjloc( fermi, self.smat[ikpt,:,:] ) * self.kwt[ikpt] ).real

        return
    
def read_dft(natm):

    efile = open('eig_1.dat','r')
    nkpt, nsymp, norb, lmom, nspn = map(int, efile.readline().split()[:5])
    nbnd, ntot = map( float , efile.readline().split()[2:4] )
    nbnd = int(nbnd)
    mu = float(efile.readline().split()[0])

    efile.readline()

    kwt  = np.zeros(nkpt, dtype=np.float)
    eigs = np.zeros((nkpt,nbnd),dtype=np.float)

    for ikpt in range(0,nkpt):
        kwt[ikpt] = float(efile.readline().split()[1])
        for ibnd in range(0,nbnd):
            eigs[ikpt,ibnd] = float(efile.readline().split()[1]) - mu
        efile.readline()
        efile.readline()
    efile.close()

    atom = []
    for iatm in range(natm):

        atom.append(corr_atom(nkpt,nbnd,norb,lmom,nspn,ntot))

        name = 'udmft_'+str(iatm+1)+'.dat'
        print ' Reading : ', name
        sfile = open(name,'r')

        for iline in range(3):
            sfile.readline()

        for ikpt in range(nkpt):
            for iline in range(2):
                sfile.readline()
            for iorb in range(norb):
                for ibnd in range(nbnd):
                    data = map(float, sfile.readline().split()[2:4])
                    atom[iatm].smat[ikpt,ibnd,iorb] = data[0] + 1j*data[1]
            sfile.readline()
            sfile.readline()
   
        sfile.close()

        atom[iatm].eigs[:,:] = eigs
        atom[iatm].kwt  = kwt
        atom[iatm].geneimp()
        print np.diag(atom[iatm].eimp).real
        print
        atom[iatm].gennloc()
        print np.sum(atom[iatm].nloc)
        print atom[iatm].nloc
        print

        name = 'eimp_'+str(iatm+1)+'.dat'
        output = open(name,'w')
        for iorb in range(norb):
            for jorb in range(norb):
                print >> output, "%5d%5d%20.10f%20.10f" %( iorb+1, jorb+1, \
                    atom[iatm].eimp[iorb,jorb].real, atom[iatm].eimp[iorb,jorb].imag )
        output.close()


    print ' Reading LDA Input Done ! '
    print

    return atom

def prjloc(imat, smat):

    norb = smat.shape[1]

    # print "Bloch to Local"
    omat = np.zeros((norb,norb),dtype=np.complex)
    omat = np.dot( smat.transpose().conj(), np.dot(imat, smat) )

    return omat

def prjblh(imat, smat):

    nbnd = smat.shape[0]

    # print "Local to Bloch"
    omat = np.zeros((nbnd,nbnd),dtype=np.complex)
    omat = np.dot( np.dot(smat, imat), smat.transpose().conj() )

    return omat
    
def natural_basis(atom,Hzee):
    
    from scipy import linalg
    
    natm = len(atom)
    
    for iatm in range(natm):

        Heff = atom[iatm].eimp + Hzee
        eigs, lvec = linalg.eigh(Heff)
        hloc = np.diag(eigs)
        hcef = atom[iatm].eimp - hloc

        for ikpt in range(atom[iatm].nkpt):
            
            Hblh = np.zeros((atom[iatm].nbnd,atom[iatm].nbnd),dtype=np.complex)
            Hblh = prjblh(hcef,atom[iatm].smat[ikpt,:,:])
            ekk  = np.diag(atom[iatm].eigs[ikpt,:])
            Hblh = ekk - Hblh
            
            atom[iatm].eigs[ikpt,:], hvec = linalg.eigh(Hblh)
            atom[iatm].smat[ikpt,:,:] = np.dot( hvec.transpose().conj(), atom[iatm].smat[ikpt,:,:] )
        
        atom[iatm].geneimp()
        atom[iatm].gennloc()

        print 'Atom :', iatm+1
        print 'Eimp (New)'
        print np.diag(atom[iatm].eimp).real
        print 'Nloc (New)'
        print np.sum(atom[iatm].nloc)
        print atom[iatm].nloc
        print                 
        
    return atom

def combs( atom ):

    natm = len(atom)
    nkpt = atom[0].nkpt
    nbnd = atom[0].nbnd
    lmom = atom[0].lmom
    nspn = atom[0].nspn
    norb = 0
    for iatm in range(natm):
        norb += atom[iatm].norb
    ntot = atom[0].ntot

    atmt = corr_atom(nkpt,nbnd,norb,lmom,nspn,ntot)

    atmt.eigs[:,:] = atom[0].eigs
    atmt.kwt = atom[0].kwt

    start = 0
    end   = 0
    for iatm in range(natm):
        end += atom[iatm].norb
        atmt.smat[:,:,start:end] = atom[iatm].smat
        atmt.eimp[start:end,start:end] = atom[iatm].eimp
        start = end

    atmt.gennloc()

    return atmt
 
def dump_data(natm,atom):

    print ' Dumping data into files ! '

    output = open('eigs_new.dat','w')
    print >> output, "%10d%-40s" %(atom.nkpt, ' : number of k-points')
    print >> output, "%10d%-40s" %(atom.nbnd, ' : number of bands')
    print >> output, "%10d%-40s" %(atom.norb/natm, ' : number of correlated orbitals')
    print >> output, "%10d%-40s" %(natm, ' : number of correlated atoms')
    print >> output, "%10.5f%-40s" %(atom.ntot, ' : number of total electrons')
    for ikpt in range(atom.nkpt):
        print >> output, "%10s%10d" %(' # ikpt ', ikpt+1)
        for ibnd in range(atom.nbnd):
            print >> output, "%10d%20.10f" %(ibnd+1, atom.eigs[ikpt,ibnd])
        print >> output
    output.close()

    output = open('smat_new.dat','w')
    print >> output, "%10d%-40s" %(atom.nkpt, ' : number of k-points')
    print >> output, "%10d%-40s" %(atom.nbnd, ' : number of bands')
    print >> output, "%10d%-40s" %(atom.norb/natm, ' : number of correlated orbitals')
    print >> output, "%10d%-40s" %(natm, ' : number of correlated atoms')
    print >> output, "%10.4f%-40s" %(atom.ntot, ' : number of correlated atoms')
    for ikpt in range(atom.nkpt):
        print >> output, "%10s%10d" %(' # ikpt ', ikpt+1)
        for ibnd in range(atom.nbnd):
            for iorb in range(atom.norb):
                print >> output, "%10d%10d%20.10f%20.10f" %(ibnd+1, iorb+1, \
                    atom.smat[ikpt,ibnd,iorb].real, -1.0*atom.smat[ikpt,ibnd,iorb].imag)
        print >> output
    output.close()

    output = open('dft.eimp.in','w')
    print >> output, "%50s" %('=================================================')
    print >> output, "%50s" %('>>>           impurity energy level           <<<')
    print >> output, "%50s" %('=================================================')
    print >> output, "%20.10f" %(0.0)
    print >> output, "%50s" %('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    for iorb in range(atom.norb/natm):
        for jorb in range(atom.norb/natm):
            print >> output, "%5d%5d%20.10f%20.10f" %(iorb+1, jorb+1, \
                atom.eimp[iorb,jorb].real, 0.0)
    output.close()

def main():
    
    from mag import magnetic_field

    natm = 4
    mag  = np.array((0.0, 0.0, 10.0),dtype=np.float)
    
    atom = read_dft( natm )
    Hzee = magnetic_field(atom[0].lmom,atom[0].nspn,mag)
    
    atom_natural = natural_basis(atom,Hzee)

    atmt = combs(atom)
    
    dump_data(natm,atmt)

if __name__ == "__main__":

    main()

