
def spinor():

	import numpy as np
	import wanntb.rotate as rot

	natm = 1
	norb = 22

	xyz = np.zeros((natm,3,3), dtype=np.float)
	angle = np.zeros((natm,3), dtype=np.float)
	dmat = np.zeros((natm,2,2), dtype=np.complex)

	xyz[0,2,:] = +0.43779135,+0.75827691,+0.48306817
	xyz[0,0,:] = +0.81919251,-0.11501778,-0.56186701

	for iatm in range(natm):
		xyz[iatm,:,:] = rot.zx_to_rmat(xyz[iatm,2,:], xyz[iatm,0,:]) 

	for iatm in range(natm):
		angle[iatm,:] = rot.rmat_to_euler(xyz[iatm,:,:])

	for iatm in range(natm):
		dmat[iatm,:,:] = rot.dmat_spinor(angle[iatm,0], angle[iatm,1], angle[iatm,2])

	tmat = np.eye(norb, dtype=np.complex)

	for iatm in range(natm):
		tmat[iatm*6+0:iatm*6+ 2, iatm*6+0:iatm*6+ 2] = dmat[iatm,:,:]
		tmat[iatm*6+2:iatm*6+ 4, iatm*6+2:iatm*6+ 4] = dmat[iatm,:,:]
		tmat[iatm*6+4:iatm*6+ 6, iatm*6+4:iatm*6+ 6] = dmat[iatm,:,:]
		tmat[iatm*6+6:iatm*6+ 8, iatm*6+6:iatm*6+ 8] = dmat[iatm,:,:]
		tmat[iatm*6+8:iatm*6+10, iatm*6+8:iatm*6+10] = dmat[iatm,:,:]

	return tmat

