import numpy as np

def upwind(mesh, vel, p):
	F = np.zeros_like(mesh.f)
	for i in mesh.fi:
		f = mesh.f[i]
		u = np.dot(vel(f.c.x, f.c.y), f.n)
		F[i] = u*(u > 0)*p[f.cells[0]] + u*(u < 0)*p[f.cells[1]]
		F[i] *= f.area()
	return F

def central(mesh, vel, p):
	F = np.zeros_like(mesh.f)
	for i in mesh.fi:
		f = mesh.f[i]
		u = np.dot(vel(f.c.x,f.c.y), f.n)
		F[i] = u*0.5*(p[f.cells[0]] + p[f.cells[1]])
		F[i] *= f.area()
	return F