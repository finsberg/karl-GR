import gmsh
import numpy as np

gmsh.initialize()

# 1. Create the outer disk (radius 2)
outer_disk_tag = gmsh.model.occ.addDisk(0, 0, 0, 2, 2)


# 2. Create the inner disk to cut from disk above (radius 1)
inner_disk_tag = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)

R_o = 2.0
R_i = 1.0
# 3. Cut the inner disk from the outer disk
cut_result, _ = gmsh.model.occ.cut([(2, outer_disk_tag)], [(2, inner_disk_tag)])

gmsh.model.occ.synchronize()
annulus_tag = cut_result[0][1]
annulus_boundaries = gmsh.model.getBoundary([(2, annulus_tag)])
ridges = gmsh.model.getEntities(dim=1)
gdim = 2
# 4. Add the physical group using the new annulus_tag
gmsh.model.addPhysicalGroup(gdim, [annulus_tag], 1)
gmsh.model.setPhysicalName(gdim, 1, "Annulus_Surface")

gmsh.model.addPhysicalGroup(1, [ridges[1][1]], 10)
gmsh.model.setPhysicalName(1, 10, "Outer_Boundary")

gmsh.model.addPhysicalGroup(1, [ridges[0][1]], 20)
gmsh.model.setPhysicalName(1, 20, "Inner_Boundary")

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.025)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.025)
gmsh.model.mesh.generate(gdim)

# Uncomment these lines to visualize the mesh
gmsh.write("annulus.msh")

# check that the physical groups were created correctly
physical_groups = gmsh.model.getPhysicalGroups()
for dim, tag in physical_groups:
    name = gmsh.model.getPhysicalName(dim, tag)
    print(f"Physical group - Dimension: {dim}, Tag: {tag}, Name: {name}")

gmsh.finalize()