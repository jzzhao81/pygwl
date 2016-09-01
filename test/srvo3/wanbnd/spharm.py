import numpy as np

np.set_printoptions(precision=3,suppress=True,linewidth=160)

def spharm_p():

	# real basis
	# |px,up>, |py,up>, |pz,up>
	# |px,dn>, |py,dn>, |pz,dn>


	nbnd = 3
	nspn = 2
	norb = nbnd * nspn

	pmat = np.zeros((norb,norb), dtype=np.complex)

	pmat[0,0] = +np.sqrt(0.5)
	pmat[4,0] = -np.sqrt(0.5)
	pmat[0,1] = +1j*np.sqrt(0.5)
	pmat[4,1] = +1j*np.sqrt(0.5)
	pmat[2,2] = 1.0

	pmat[0+1,0+nbnd] = +np.sqrt(0.5)
	pmat[4+1,0+nbnd] = -np.sqrt(0.5)
	pmat[0+1,1+nbnd] = +1j*np.sqrt(0.5)
	pmat[4+1,1+nbnd] = +1j*np.sqrt(0.5)
	pmat[2+1,2+nbnd] = +1.0

	# print pmat 

	return pmat

def spharm_d():

	# real basis: 
	# |dxy,up>, |dxz,up>, |dyz,up>, |d(x^2-y^x),up>, |dz^2,up>
	# |dxy,dn>, |dxz,dn>, |dyz,dn>, |d(x^2-y^x),dn>, |dz^2,dn>

	nbnd = 5
	nspn = 2
	norb = nbnd * nspn

	dmat = np.zeros((norb,norb), dtype=np.complex)

	dmat[0,0] = +1j*np.sqrt(0.5)
	dmat[8,0] = -1j*np.sqrt(0.5)
	dmat[2,1] = +np.sqrt(0.5)
	dmat[6,1] = -np.sqrt(0.5)
	dmat[2,2] = +1j*np.sqrt(0.5)
	dmat[6,2] = +1j*np.sqrt(0.5)
	dmat[0,3] = +np.sqrt(0.5)
	dmat[8,3] = +np.sqrt(0.5)
	dmat[4,4] = +1.0

	dmat[0+1,0+nbnd] = +1j*np.sqrt(0.5)
	dmat[8+1,0+nbnd] = -1j*np.sqrt(0.5)
	dmat[2+1,1+nbnd] = +np.sqrt(0.5)
	dmat[6+1,1+nbnd] = -np.sqrt(0.5)
	dmat[2+1,2+nbnd] = +1j*np.sqrt(0.5)
	dmat[6+1,2+nbnd] = +1j*np.sqrt(0.5)
	dmat[0+1,3+nbnd] = +np.sqrt(0.5)
	dmat[8+1,3+nbnd] = +np.sqrt(0.5)
	dmat[4+1,4+nbnd] = +1.0

	# print dmat

	return dmat 

def spharm_f():

	# real basis
	# |fxz^2,up>,|fyz^2,up>,|fz^3,up>,|fx(x^2-3y^2),up>,|fy(3x^2-y^2),up>,|fz(x^2-y^2),up>,|fxyz,up>
	# |fxz^2,dn>,|fyz^2,dn>,|fz^3,dn>,|fx(x^2-3y^2),dn>,|fy(3x^2-y^2),dn>,|fz(x^2-y^2),dn>,|fxyz,dn>

	nbnd = 7
	nspn = 2
	norb = nbnd * nspn

	fmat = np.zeros((norb,norb),dtype=np.complex)

	fmat[ 4, 0] = +np.sqrt(0.5)
	fmat[ 8, 0] = -np.sqrt(0.5)
	fmat[ 4, 1] = +1j*np.sqrt(0.5)
	fmat[ 8, 1] = +1j*np.sqrt(0.5)
	fmat[ 6, 2] = +1.0
	fmat[ 0, 3] = +np.sqrt(0.5)
	fmat[12, 3] = -np.sqrt(0.5)
	fmat[ 0, 4] = +1j*np.sqrt(0.5)
	fmat[12, 4] = +1j*np.sqrt(0.5)
	fmat[ 2, 5] = +np.sqrt(0.5)
	fmat[10, 5] = +np.sqrt(0.5)
	fmat[ 2, 6] = +1j*np.sqrt(0.5)
	fmat[10, 6] = -1j*np.sqrt(0.5)

	fmat[ 4+1, 0+nbnd] = +np.sqrt(0.5)
	fmat[ 8+1, 0+nbnd] = -np.sqrt(0.5)
	fmat[ 4+1, 1+nbnd] = +1j*np.sqrt(0.5)
	fmat[ 8+1, 1+nbnd] = +1j*np.sqrt(0.5)
	fmat[ 6+1, 2+nbnd] = +1.0
	fmat[ 0+1, 3+nbnd] = +np.sqrt(0.5)
	fmat[12+1, 3+nbnd] = -np.sqrt(0.5)
	fmat[ 0+1, 4+nbnd] = +1j*np.sqrt(0.5)
	fmat[12+1, 4+nbnd] = +1j*np.sqrt(0.5)
	fmat[ 2+1, 5+nbnd] = +np.sqrt(0.5)
	fmat[10+1, 5+nbnd] = +np.sqrt(0.5)
	fmat[ 2+1, 6+nbnd] = +1j*np.sqrt(0.5)
	fmat[10+1, 6+nbnd] = -1j*np.sqrt(0.5)

	# print fmat.real
	# print
	# print fmat.imag

	return fmat

if __name__ == "__main__" : 

	spharm_f()

