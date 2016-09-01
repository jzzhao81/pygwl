import numpy as np
from scipy import linalg

def surho(freq, h00, h01):

	Lmax = 20
	eta  = 0.002

	nbnd = h00.shape[0]

	omega = np.zeros((nbnd,nbnd),dtype=np.complex)
	Tmati = np.zeros((nbnd,nbnd),dtype=np.complex)
	Tmatj = np.zeros((nbnd,nbnd),dtype=np.complex)
	tii   = np.zeros((nbnd,nbnd),dtype=np.complex)
	tjj   = np.zeros((nbnd,nbnd),dtype=np.complex)
	tio   = np.zeros((nbnd,nbnd),dtype=np.complex)
	tin   = np.zeros((nbnd,nbnd),dtype=np.complex)
	tjo   = np.zeros((nbnd,nbnd),dtype=np.complex)
	tjn   = np.zeros((nbnd,nbnd),dtype=np.complex)
	g00   = np.zeros((nbnd,nbnd),dtype=np.complex)

	# I
	umat  = np.zeros((nbnd,nbnd),dtype=np.complex)
	np.fill_diagonal(umat,1.0)

	np.fill_diagonal(omega, freq+1j*eta)
	g00   = linalg.inv(omega-h00)

	tio   = np.dot(g00,h01.transpose().conj())
	tjo   = np.dot(g00,h01)

	tii   = tio
	tjj   = tjo
	Tmati = tio
	Tmatj = tjo

	for iloop in range(Lmax):

		temp = np.zeros((nbnd,nbnd),dtype=np.complex)
		temp = linalg.inv(umat-np.dot(tio,tjo)-np.dot(tjo,tio))
		tin  = np.dot( temp, np.dot(tio,tio) )
		tjn  = np.dot( temp, np.dot(tjo,tjo) )

		if np.sum(np.absolute(np.dot(tjj,tin))) < 1.0e-8 : break

		Tmati = Tmati + np.dot( tjj, tin )
		Tmatj = Tmatj + np.dot( tii, tjn )

		tii = np.dot( tii, tin )
		tjj = np.dot( tjj, tjn )

		tio = tin
		tjo = tjn

		if iloop >= Lmax-1 :
			print 'Error in surface :'
			print 'H00'
			print h00
			print 'H01'
			print h01
			exit()

	g00 = linalg.inv( omega-h00-np.dot(h01,Tmati) )

	# Surface modify
	g00 = linalg.inv( omega-h00-np.dot(np.dot(h01,g00),h01.transpose().conj()) )


	# generate spinor
	spinor = genspinor(nbnd)
	spin   = np.zeros(3,dtype=np.float)
	for ii in range(3):
		spin[ii] = -np.trace(np.dot(g00,spinor[ii,:,:])).imag/np.pi

	rho = -np.trace(g00).imag/np.pi

	return rho, spin


def PauliMatrix():
    
    sgmx = np.zeros((2,2),dtype=np.complex)
    sgmy = np.zeros((2,2),dtype=np.complex)
    sgmz = np.zeros((2,2),dtype=np.complex)
    
    sgmx[0,1] =  1
    sgmx[1,0] =  1
    
    sgmy[0,1] = -1j
    sgmy[1,0] =  1j
    
    sgmz[0,0] =  1
    sgmz[1,1] = -1
    
    return sgmx, sgmy, sgmz


def genSxyz(ll,ns):
    
    sgmx,sgmy,sgmz = PauliMatrix()
    
    nm = 2*ll+1
    
    SS = np.zeros((3,nm*ns,nm*ns),dtype=np.complex)
    
    for mi in range(nm):
        SS[0,ns*mi:ns*(mi+1), ns*mi:ns*(mi+1)] = sgmx
        SS[1,ns*mi:ns*(mi+1), ns*mi:ns*(mi+1)] = sgmy
        SS[2,ns*mi:ns*(mi+1), ns*mi:ns*(mi+1)] = sgmz
    
    return SS

def genspinor(nwan):

	spinor = np.zeros((3,nwan,nwan), dtype=np.complex)

	spinor[:,0:10,0:10] = genSxyz(2,2)
	spinor[:,10:16,10:16] = genSxyz(1,2)
	spinor[:,16:22,16:22] = genSxyz(1,2)

	# sss = np.zeros((nwan,nwan),dtype=np.float)
	# sss[ 0,    0] = 1.0
	# sss[ 1, 5+ 0] = 1.0
	# sss[ 2,    1] = 1.0
	# sss[ 3, 5+ 1] = 1.0
	# sss[ 4,    2] = 1.0
	# sss[ 5, 5+ 2] = 1.0
	# sss[ 6,    3] = 1.0
	# sss[ 7, 5+ 3] = 1.0
	# sss[ 8,    4] = 1.0
	# sss[ 9, 5+ 4] = 1.0
	# sss[10,   10] = 1.0
	# sss[11, 3+10] = 1.0
	# sss[12,   11] = 1.0
	# sss[13, 3+11] = 1.0
	# sss[14,   12] = 1.0
	# sss[15, 3+12] = 1.0
	# sss[16,   16] = 1.0
	# sss[17, 3+16] = 1.0
	# sss[18,   17] = 1.0
	# sss[19, 3+17] = 1.0
	# sss[20,   18] = 1.0
	# sss[21, 3+18] = 1.0

	# for i in range(3):
	# 	spinor[i,:,:] = np.dot(sss.transpose(), np.dot(spinor[i,:,:], sss))

	return spinor

if __name__ =='__main__':

	genspinor(22)





