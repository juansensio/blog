from mesh import Face 
from dataclasses import dataclass, field
import numpy as np 

@dataclass
class DirichletBC:
	f: list[Face]
	p: list[float]
	
	def flux_conv(self, mesh, F, vel, p):
		for f, p in zip(self.f, self.p):
			fb = mesh.f[f]
			u = np.dot(vel(fb.c.x, fb.c.y), fb.n)
			F[f] = u*p*fb.a
		return F

	def flux_visc(self, mesh, F, vel, p, TAU):
		for f, _p in zip(self.f, self.p):
			fb = mesh.f[f]
			c0, c1 = mesh.c[fb.cells[0]].c, fb.c
			F[f] -= TAU*(_p - p[fb.cells[0]])*fb.a / np.linalg.norm(np.array([c1.x, c1.y]) - np.array([c0.x, c0.y]))
		return F
	

@dataclass
class NeumannBC:
	f: list[Face]
	
	def flux_conv(self, mesh, F, vel, p):
		for f in self.f:
			fb = mesh.f[f]
			u = np.dot(vel(fb.c.x, fb.c.y), fb.n)
			F[f] = u*p[fb.cells[0]]*fb.a
		return F

	def flux_visc(self, mesh, F, vel, p, TAU):
		for f in self.f:
			F[f] -= 0.
		return F