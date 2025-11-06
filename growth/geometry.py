import shutil
from pathlib import Path
from typing import NamedTuple

from mpi4py import MPI

import basix
import cardiac_geometries as cg
import dolfinx
import numpy as np


def get_cube_fibers(mesh, fiber_el):
    fiber_space = dolfinx.fem.functionspace(mesh, fiber_el)

    fiber_x = dolfinx.fem.Function(fiber_space, name="fiber_r")
    fiber_y = dolfinx.fem.Function(fiber_space, name="fiber_ang")
    fiber_z = dolfinx.fem.Function(fiber_space, name="fiber_z")

    fiber_x_expr = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    fiber_x.interpolate(
        dolfinx.fem.Expression(fiber_x_expr, fiber_space.element.interpolation_points()),
    )
    fiber_y_expr = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    fiber_y.interpolate(
        dolfinx.fem.Expression(fiber_y_expr, fiber_space.element.interpolation_points()),
    )
    fiber_z_expr = dolfinx.fem.Constant(mesh, (0.0, 0.0, 1.0))
    fiber_z.interpolate(
        dolfinx.fem.Expression(fiber_z_expr, fiber_space.element.interpolation_points()),
    )
    return fiber_x, fiber_y, fiber_z


class BaseGeometry(NamedTuple):
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    f: dolfinx.fem.Function
    s: dolfinx.fem.Function
    n: dolfinx.fem.Function
    markers: dict[str, tuple[int, int]]

    def save(self, output_dir: Path) -> None:
        """Save the geometry to a file."""
        # Save the mesh
        with dolfinx.io.XDMFFile(self.mesh.comm, output_dir / "mesh.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_meshtags(self.facet_tags, self.mesh.geometry)

        # Save the fiber fields
        save_fibers(self, output_dir)


class CubeGeometry(BaseGeometry):
    @property
    def x_range(self) -> tuple[float, float]:
        comm = self.mesh.comm
        min_value = comm.allreduce(self.mesh.geometry.x[:, 0].min(), op=MPI.MIN)
        max_value = comm.allreduce(self.mesh.geometry.x[:, 0].max(), op=MPI.MAX)
        return min_value, max_value

    @property
    def y_range(self) -> tuple[float, float]:
        comm = self.mesh.comm
        min_value = comm.allreduce(self.mesh.geometry.x[:, 1].min(), op=MPI.MIN)
        max_value = comm.allreduce(self.mesh.geometry.x[:, 1].max(), op=MPI.MAX)
        return min_value, max_value

    @property
    def z_range(self) -> tuple[float, float]:
        comm = self.mesh.comm
        min_value = comm.allreduce(self.mesh.geometry.x[:, 2].min(), op=MPI.MIN)
        max_value = comm.allreduce(self.mesh.geometry.x[:, 2].max(), op=MPI.MAX)
        return min_value, max_value


class CylinderGeometry(BaseGeometry):
    @property
    def z_range(self) -> tuple[float, float]:
        comm = self.mesh.comm
        min_value = comm.allreduce(self.mesh.geometry.x[:, 2].min(), op=MPI.MIN)
        max_value = comm.allreduce(self.mesh.geometry.x[:, 2].max(), op=MPI.MAX)
        return min_value, max_value

    @property
    def r_range(self) -> tuple[float, float]:
        # Get the minimum and maximum values of the radial coordinate
        comm = self.mesh.comm
        r = self.mesh.geometry.x[:, 0] ** 2 + self.mesh.geometry.x[:, 1] ** 2
        min_value = comm.allreduce(r.min(), op=MPI.MIN)
        max_value = comm.allreduce(r.max(), op=MPI.MAX)
        return np.sqrt(min_value), np.sqrt(max_value)

    @property
    def theta_range(self) -> tuple[float, float]:
        # Get the minimum and maximum values of the angular coordinate
        comm = self.mesh.comm
        theta = np.arctan2(self.mesh.geometry.x[:, 1], self.mesh.geometry.x[:, 0])
        min_value = comm.allreduce(theta.min(), op=MPI.MIN)
        max_value = comm.allreduce(theta.max(), op=MPI.MAX)
        return min_value, max_value


def save_fibers(geo: BaseGeometry, output_dir: Path) -> None:
    shutil.rmtree(output_dir / "fibers.bp", ignore_errors=True)
    with dolfinx.io.VTXWriter(
        MPI.COMM_WORLD,
        output_dir / "fibers.bp",
        [geo.f, geo.s, geo.n],
        engine="BP4",
    ) as vtx:
        vtx.write(0.0)


def load_cylinder_geometry(
    comm: MPI.Intracomm,
    inner_radius: float = 10.0,
    outer_radius: float = 20.0,
    height: float = 40.0,
    char_length: float = 10.0,
    outdir: Path = Path("cylinder"),
    fiber_angle_endo: float = -60,
    fiber_angle_epi: float = 60,
) -> CylinderGeometry:
    import cardiac_geometries as cg

    geo = cg.mesh.cylinder(
        comm=comm,
        r_inner=inner_radius,
        r_outer=outer_radius,
        height=height,
        char_length=char_length,
        outdir=outdir,
        create_fibers=True,
        fiber_angle_endo=fiber_angle_endo,
        fiber_angle_epi=fiber_angle_epi,
    )

    return CylinderGeometry(
        mesh=geo.mesh,
        facet_tags=geo.ffun,
        f=geo.f0,
        s=geo.s0,
        n=geo.n0,
        markers=geo.markers,
    )


class EllipsoidGeometry(BaseGeometry):
    @property
    def z_range(self) -> tuple[float, float]:
        comm = self.mesh.comm
        min_value = comm.allreduce(self.mesh.geometry.x[:, 2].min(), op=MPI.MIN)
        max_value = comm.allreduce(self.mesh.geometry.x[:, 2].max(), op=MPI.MAX)
        return min_value, max_value

    @property
    def r_range(self) -> tuple[float, float]:
        # Get the minimum and maximum values of the radial coordinate
        comm = self.mesh.comm
        r = self.mesh.geometry.x[:, 0] ** 2 + self.mesh.geometry.x[:, 1] ** 2
        min_value = comm.allreduce(r.min(), op=MPI.MIN)
        max_value = comm.allreduce(r.max(), op=MPI.MAX)
        return np.sqrt(min_value), np.sqrt(max_value)

    @property
    def theta_range(self) -> tuple[float, float]:
        # Get the minimum and maximum values of the angular coordinate
        comm = self.mesh.comm
        theta = np.arctan2(self.mesh.geometry.x[:, 1], self.mesh.geometry.x[:, 0])
        min_value = comm.allreduce(theta.min(), op=MPI.MIN)
        max_value = comm.allreduce(theta.max(), op=MPI.MAX)
        return min_value, max_value


def load_ellipsoid_geometry(output_dir) -> EllipsoidGeometry:
    geo = cg.mesh.lv_ellipsoid(
        output_dir,
        create_fibers=True,
    )

    return EllipsoidGeometry(
        mesh=geo.mesh,
        facet_tags=geo.ffun,
        f=geo.f0,
        s=geo.s0,
        n=geo.n0,
        markers=geo.markers,
    )


def load_cube_geometry(comm: MPI.Intracomm) -> CubeGeometry:
    x_min, x_max, Nx = 0, 1, 4
    y_min, y_max, Ny = 0, 1, 4
    z_min, z_max, Nz = 0, 1, 4

    mesh = dolfinx.mesh.create_box(
        comm,
        [np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max])],
        [Nx, Ny, Nz],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )

    bcs_func = dict(
        XMIN=lambda x: np.isclose(x[0], x_min),
        XMAX=lambda x: np.isclose(x[0], x_max),
        YMIN=lambda x: np.isclose(x[1], y_min),
        YMAX=lambda x: np.isclose(x[1], y_max),
        ZMIN=lambda x: np.isclose(x[2], z_min),
        ZMAX=lambda x: np.isclose(x[2], z_max),
    )
    points = []
    values = []
    markers = {}
    for i, (name, f) in enumerate(bcs_func.items(), start=1):
        facet_points = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, f)
        points.append(facet_points)
        values.append(np.full_like(facet_points, i))
        markers[name] = (i, 2)

    facet_tags = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        np.hstack(points),
        np.hstack(values).astype(np.int32),
    )

    # Create fiber field
    fiber_el = basix.ufl.element(
        family="Lagrange",
        cell=str(mesh.ufl_cell()),
        degree=1,
        shape=(mesh.geometry.dim,),
        discontinuous=True,
    )
    fiber_x, fiber_y, fiber_z = get_cube_fibers(mesh, fiber_el)

    return CubeGeometry(
        mesh=mesh,
        facet_tags=facet_tags,
        f=fiber_x,
        s=fiber_y,
        n=fiber_z,
        markers=markers,
    )
