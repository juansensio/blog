import numpy as np

def euler(F, mesh, p, dt):
	R = np.zeros_like(p)
	for i, c in enumerate(mesh.c):
		R[i] = np.sum([F[f.ix]*c.fo[j] for j, f in enumerate(c.f)]) / c.volume()
		p[i] -= dt*R[i]
	return p, R

def ab2(F, mesh, p, dt, R0):
	R = np.zeros_like(p)
	for i, c in enumerate(mesh.c):
		R[i] = np.sum([F[f.ix]*c.fo[j] for j, f in enumerate(c.f)]) / c.volume()
		p[i] -= dt*0.5*(3.*R[i] - R0[i])
	return p, R