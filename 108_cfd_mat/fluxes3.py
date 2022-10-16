import numpy as np
from scipy.sparse import coo_matrix

class Upwind:
	
	def __init__(self, mesh, vel):
		U = np.zeros((len(mesh.f), len(mesh.c)), dtype=float)
		for i in mesh.fi:
			f = mesh.f[i]
			c0, c1 = f.cells
			u = np.dot(vel(f.c.x, f.c.y), f.n)
			U[i, c0] = u*(u > 0)*f.a
			U[i, c1] = u*(u < 0)*f.a
		self.U = coo_matrix(U)

	def __call__(self, p):
		return self.U @ p

# def central(mesh, vel, p):
# 	F = np.zeros_like(mesh.f)
# 	for i in mesh.fi:
# 		f = mesh.f[i]
# 		u = np.dot(vel(f.c.x,f.c.y), f.n)
# 		F[i] = u*0.5*(p[f.cells[0]] + p[f.cells[1]])
# 		F[i] *= f.a
# 	return F

# def flux_visc(mesh, F, vel, p, TAU):
# 	for i in mesh.fi:
# 		f = mesh.f[i]
# 		c0, c1 = mesh.c[f.cells[0]].c, mesh.c[f.cells[1]].c
# 		F[i] -= TAU*(p[f.cells[1]] - p[f.cells[0]])*f.a / np.linalg.norm(np.array([c1.x, c1.y]) - np.array([c0.x, c0.y]))
# 	return F