import numpy as np
from scipy.sparse import coo_matrix

class Euler:
	def __init__(self, mesh):
		S = np.zeros((len(mesh.c), len(mesh.f)), dtype=float)
		for i, c in enumerate(mesh.c):
			for j, f in enumerate(c.f):
				S[i, f] = c.fo[j] / c.v
		self.S = coo_matrix(S)

	def __call__(self, F, p, dt):
		R = self.S @ F
		p -= dt*R
		return p, R



# def ab2(F, mesh, p, dt, R0):
# 	R = np.zeros_like(p)
# 	for i, c in enumerate(mesh.c):
# 		R[i] = np.sum([F[f]*c.fo[j] for j, f in enumerate(c.f)]) / c.v
# 		p[i] -= dt*0.5*(3.*R[i] - R0[i])
# 	return p, R