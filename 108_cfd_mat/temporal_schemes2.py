import numpy as np

class Euler:
	def __init__(self, mesh):
		self.S = np.zeros((len(mesh.c), len(mesh.f)))
		for i, c in enumerate(mesh.c):
			for j, f in enumerate(c.f):
				self.S[i, f] = c.fo[j] / c.v

	def __call__(self, F, p, dt):
		R = np.dot(self.S, F)
		p -= dt*R
		return p, R



# def ab2(F, mesh, p, dt, R0):
# 	R = np.zeros_like(p)
# 	for i, c in enumerate(mesh.c):
# 		R[i] = np.sum([F[f]*c.fo[j] for j, f in enumerate(c.f)]) / c.v
# 		p[i] -= dt*0.5*(3.*R[i] - R0[i])
# 	return p, R