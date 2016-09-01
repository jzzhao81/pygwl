
import sys
import numpy as np

class symmetry:

    def __init__(self, data=[]):

        self.data = data
        self.ntot = len(self.data)
        self.nsym = len(self.data)
        self.flag = []
        
        self.countsym()

    def defsym(self, data):
        eps = 1e-5
        self.data = []
        for indx,nume in enumerate(data):
            if (np.absolute(nume-data[:indx])<eps).any() :
                self.data.append(np.where(np.absolute(nume-data[:indx])<eps)[0][0])
            else : 
                self.data.append(indx)
        self.ntot = len(self.data)
        self.countsym()
        return

    def countsym(self):

        # Check number of different orbitals
        self.nsym = 0
        for idat, data in enumerate(self.data):
            match = False
            for ii in range(idat):
                if self.data[ii] == data : 
                    match = True
                    break
            if match == False :
                self.flag.append(data)
                self.nsym += 1

        self.indx = [[] for i in range(self.nsym)]
        self.npsy = np.zeros(self.nsym,dtype=int)

        for isym in range(self.nsym):
            for idat, data in enumerate(self.data):
                if data == self.flag[isym] :
                    self.indx[isym].append(idat)
                    self.npsy[isym] += 1

        return

    def symmetrize(self, inp):

        if inp.shape[0] != self.ntot :
            sys.exit(" Input data size is wrong !\n")

        smat = np.zeros(self.nsym, dtype=np.float)
        for isym in range(self.nsym):
            for idat in range(self.ntot):
                if self.data[idat] == self.flag[isym] : smat[isym] += inp[idat]
            smat[isym] /= np.float(self.npsy[isym])

        out = np.zeros(self.ntot,dtype=np.float)
        for isym in range(self.nsym):
            for idat in range(self.npsy[isym]):
                out[self.indx[isym][idat]] = smat[isym]
        return out


def main():

    symm = symmetry()
    symm.defsym(np.array([-1.0,-1.0, 0.0, 0.0, 0.0, 1.0, 1.0,1.0,1.0]))

    symm.countsym()
    print symm.nsym, symm.indx, symm.flag, symm.npsy

if __name__ == "__main__" :

    main()



