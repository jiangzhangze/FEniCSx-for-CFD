import gmsh
import os
import numpy as np
import sys
gmsh.initialize()

L = 2.2
H = 0.41
c_x = c_y =0.2
o_x = 0.3
o_y = 0.2
r = 0.05
r2 = 0.01
gdim = 2
model_rank = 0

rectangle = gmsh.model.occ.addRectangle(0,0,0, L, H, tag=1)
obstacle1 = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
obstacle2 = gmsh.model.occ.addDisk(o_x, o_y, 0, r2, r2)
pre_fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle1)])
fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle2)])
gmsh.model.occ.synchronize()
fluid_marker = 1

volumes = gmsh.model.getEntities(dim=gdim)
assert(len(volumes) == 1)
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []

boundaries = gmsh.model.getBoundary(volumes, oriented=False)
for boundary in boundaries:
    center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
    if np.allclose(center_of_mass, [0, H/2, 0]):
            inflow.append(boundary[1])
    elif np.allclose(center_of_mass, [L, H/2, 0]):
            outflow.append(boundary[1])
    elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
            walls.append(boundary[1])
    else:
            obstacle.append(boundary[1])
gmsh.model.addPhysicalGroup(1, walls, wall_marker)
gmsh.model.setPhysicalName(1, wall_marker, "Walls")
gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

res_min = r / 3
distance_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
threshold_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
gmsh.model.mesh.generate(gdim)
gmsh.model.mesh.setOrder(2)
gmsh.model.mesh.optimize("Netgen")

gmsh.write("cylinders.msh")

'''
#visualize
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
'''