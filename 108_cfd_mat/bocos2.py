from mesh import Face 
from dataclasses import dataclass, field
import numpy as np 

@dataclass
class DirichletBC2:
	f: list[Face]
	p: list[float]
	F: np.array = field(init=False)

	def setup(self, mesh, vel):
		self.F = np.zeros_like(mesh.f)
		for i, p in zip(self.f, self.p):
			f = mesh.f[i]
			u = np.dot(vel(f.c.x, f.c.y), f.n)
			self.F[i] = u*p*f.a
	
	def flux_conv(self, p):
		return self.F

	def flux_visc(self, mesh, F, vel, p, TAU):
		for f, _p in zip(self.f, self.p):
			fb = mesh.f[f]
			c0, c1 = mesh.c[fb.cells[0]].c, fb.c
			F[f] -= TAU*(_p - p[fb.cells[0]])*fb.a / np.linalg.norm(np.array([c1.x, c1.y]) - np.array([c0.x, c0.y]))
		return F
	

@dataclass
class NeumannBC2:
	f: list[Face]
	C: np.array = field(init=False)

	def setup(self, mesh, vel):
		self.C = np.zeros((len(mesh.f), len(mesh.c)))
		for i in self.f:
			f = mesh.f[i]
			c0 = f.cells[0]
			u = np.dot(vel(f.c.x, f.c.y), f.n)
			self.C[i, c0] = u*f.a
	
	def flux_conv(self, p):
		return np.dot(self.C, p)

	def flux_visc(self, mesh, F, vel, p, TAU):
		for f in self.f:
			F[f] -= 0.
		return F