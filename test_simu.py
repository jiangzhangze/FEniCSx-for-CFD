from simulation import simulation
from area import *

points = probes(x_min=0.4, y_min=0.05, length=0.1, num=10)
simu = simulation(c_x=0.2, c_y=0.2, o_x=0.3, o_y=0.2, r=0.05, r2=0)
mesh, ft = simu.generate_mesh()
u, p = simu.compute(mesh=mesh, ft=ft, points=points)
print(u)