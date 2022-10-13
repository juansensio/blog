from dataclasses import dataclass, field
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class Point:
	x: float
	y: float

@dataclass
class Vertex:
	p: Point

@dataclass
class Face:
	v: list[Vertex]
	c: Point
	cells: list[int] = field(default_factory=list, init=False)
	b: int = 0 
	n: np.array = field(init=False)

	def __post_init__(self):
		assert len(self.v) == 2
		self.n = np.array([self.v[1].p.y - self.v[0].p.y, self.v[0].p.x - self.v[1].p.x]) 
		self.n /= np.linalg.norm(self.n)

	def area(self) -> float:
		return ((self.v[1].p.x - self.v[0].p.x)**2 + (self.v[1].p.y - self.v[0].p.y)**2)**0.5

	def order_cells(self, c):
		if len(self.cells) == 2:
			c1, c2 = self.cells
			v = np.array([c[c2].c.x - c[c1].c.x, c[c2].c.y - c[c1].c.y])
			if np.dot(v, self.n) < 0:
				self.cells = [c2, c1]

@dataclass
class Cell:
	f: list[Face]
	c: Point = field(init=False)
	fo: list[bool] = field(default_factory=list, init=False)

	def __init__(self, f, ix):
		self.f = f
		cx = np.mean([f.c.x for f in self.f]) 
		cy = np.mean([f.c.y for f in self.f]) 
		self.c = Point(cx, cy)
		assert len(self.f) == 4
		for _f in self.f:
			if len(_f.cells) >= 2:
				raise Exception()	
			_f.cells.append(ix)
			
	def volume(self) -> float:
		return self.f[0].area() * self.f[1].area() 

	def order_faces(self):
		self.fo = []
		for f in self.f:
			v = np.array([f.c.x - self.c.x, f.c.y - self.c.y])
			self.fo.append(np.dot(v, f.n) > 0)

@dataclass
class Mesh:
	Nx: int
	Ny: int
	Lx: list[float] 
	Ly: list[float] 

	v: list[Vertex] = field(init=False)
	f: list[Face] = field(init=False)
	c: list[Cell] = field(init=False)
	fi: list[int] = field(default_factory=list, init=False)
	fb: list[int] = field(default_factory=list, init=False)

	def __post_init__(self):
		vx = np.linspace(self.Lx[0], self.Lx[1], self.Nx+1)
		vy = np.linspace(self.Ly[0], self.Ly[1], self.Ny+1)
		self.v = [Vertex(Point(x, y)) for y in vy for x in vx]
		fh = [Face([self.v[i + (self.Nx+1)*j], self.v[i+1+ (self.Nx+1)*j]], Point((vx[i]+vx[i+1])/2, vy[j])) for i in range(self.Nx) for j in range(self.Ny + 1)]
		fv = [Face([self.v[i + j*(self.Nx+1)], self.v[i+ (self.Nx+1)+j*(self.Nx+1)]], Point(vx[i], (vy[j]+vy[j+1])/2)) for j in range(self.Ny) for i in range(self.Nx+1)]
		self.f = fv + fh
		self.c = [Cell([fv[i + j*(self.Nx+1)], fh[i*(self.Ny+1)+1+j], fv[i + j*(self.Nx+1) + 1], fh[i*(self.Ny+1)+j]], i + self.Nx*j) for j in range(self.Ny) for i in range(self.Nx)]
		for i, f in enumerate(self.f):
			f.order_cells(self.c)
			if len(f.cells) == 2:
				self.fi.append(i)
			else:
				self.fb.append(i)
		for c in self.c:
			c.order_faces()
		

	def plot(self):
		plt.scatter([v.p.x for v in self.v], [v.p.y for v in self.v])
		for f in self.f:
			plt.plot([v.p.x for v in f.v], [v.p.y for v in f.v], 'r-')
			plt.plot(f.c.x, f.c.y, 'Xr')
		for c in self.c:
			plt.plot(c.c.x, c.c.y, '^g')
		plt.xlabel('x')
		plt.ylabel('y', rotation=0)
		plt.show()		

	def plot_field(self, p):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(
			[[self.c[i + j*self.Nx].c.x for i in range(self.Nx)] for j in range(self.Ny)], 
			[[self.c[i + j*self.Nx].c.y for i in range(self.Nx)] for j in range(self.Ny)], 
			p.reshape(self.Ny, self.Nx),
			cmap='viridis'
		)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		plt.show()


if __name__ == '__main__':
    mesh = Mesh(1, 1, [0, 1], [0, 1])
    assert len(mesh.v) == 4
    assert len(mesh.f) == 4
    assert len(mesh.c) == 1
    assert mesh.c[0].volume() == 1.
    assert mesh.f[0].area() == 1.
    assert mesh.f[1].area() == 1.
    assert mesh.f[2].area() == 1.
    assert mesh.f[3].area() == 1.
    mesh = Mesh(2, 1, [0, 1], [0, 1])
    assert len(mesh.v) == 6
    assert len(mesh.f) == 7
    assert len(mesh.c) == 2
    assert mesh.c[0].volume() == 0.5
    assert mesh.c[1].volume() == 0.5
    assert mesh.f[0].area() == 1.
    assert mesh.f[1].area() == 1.
    assert mesh.f[2].area() == 1.
    assert mesh.f[3].area() == 0.5
    assert mesh.f[4].area() == 0.5
    assert mesh.f[5].area() == 0.5
    assert mesh.f[6].area() == 0.5
    mesh = Mesh(1, 2, [0, 1], [0, 1])
    assert len(mesh.v) == 6
    assert len(mesh.f) == 7
    assert len(mesh.c) == 2
    assert mesh.c[0].volume() == 0.5
    assert mesh.c[1].volume() == 0.5
    assert mesh.f[0].area() == 0.5
    assert mesh.f[1].area() == 0.5
    assert mesh.f[2].area() == 0.5
    assert mesh.f[3].area() == 0.5
    assert mesh.f[4].area() == 1.
    assert mesh.f[5].area() == 1.
    assert mesh.f[6].area() == 1.
    mesh = Mesh(2, 2, [0, 1], [0, 1])
    assert len(mesh.v) == 9
    assert len(mesh.f) == 12
    assert len(mesh.c) == 4
    assert mesh.c[0].volume() == 0.25
    assert mesh.c[1].volume() == 0.25
    assert mesh.c[2].volume() == 0.25
    assert mesh.c[3].volume() == 0.25
    assert mesh.f[0].area() == 0.5
    assert mesh.f[6].area() == 0.5
    mesh = Mesh(2, 2, [-1, 1], [0, 10])
    assert len(mesh.v) == 9
    assert len(mesh.f) == 12
    assert len(mesh.c) == 4
    assert mesh.c[0].volume() == 5.
    assert mesh.c[1].volume() == 5.
    assert mesh.c[2].volume() == 5.
    assert mesh.c[3].volume() == 5.
    assert mesh.f[0].area() == 5.
    assert mesh.f[6].area() == 1.
