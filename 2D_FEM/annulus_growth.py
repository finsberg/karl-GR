import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import ufl
import basix
import numpy as np
from collections import defaultdict
from pathlib import Path
import scifem

# 1. Initialize MPI communicator
comm = MPI.COMM_WORLD

output_dir = Path(
    "simple_growth",
)  # Specify the folder where all the data should be saved
output_dir.mkdir(parents=True, exist_ok=True)  # Make the folder if it doesn't exist

# 2. Define filename
filename = "annulus.msh"

# 3. Read the mesh
#    - Pass 'comm' as the second argument
#    - Specify 'rank=0' (so only one process reads the file)
#    - CRITICAL: Specify 'gdim=2' for your 2D mesh

mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    filename, comm, rank=0, gdim=2
)

''' CREATE FIBERS '''
P_fiber = basix.ufl.element(
    family="CG",  # Type of functions (Lagrange polynomials)
    cell=str(mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
    degree=2,  # Polynomial degree of functions #TODO find out why this only works for degree 1
    shape=(mesh.geometry.dim,),  # Dimension of functions (2D vector in this case)
)

fiber_space = dolfinx.fem.functionspace(mesh, P_fiber)

x, y, z = fiber_space.tabulate_dof_coordinates().T
r = np.sqrt(x**2 + y**2)
e_r = np.array([x / r, y / r])
e_theta = np.array([-e_r[1], e_r[0]])  # 90 degree rotation of e_r

line_points = np.array([[r * np.cos(theta), r * np.sin(theta)]
                        for r in np.linspace(1.0, 2.0, 10)
                        for theta in np.linspace(0, 2 * np.pi, 36)], dtype=np.float64)

f0 = dolfinx.fem.Function(fiber_space, name="f0")
r0 = dolfinx.fem.Function(fiber_space, name="r0")

f0.x.array[:] = np.array(e_theta).T.reshape(-1)
r0.x.array[:] = np.array(e_r).T.reshape(-1)

breakpoint()


''' BOUNDARY CONDITONS '''
import numpy as np

def inner_BC(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.isclose(r, 1.0)

def outer_BC(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.isclose(r, 2.0)

fdim = mesh.topology.dim - 1
inner_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, inner_BC)
outer_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, outer_BC)

''' DATA STORAGE '''
#region
functions: dict[str, dolfinx.fem.Function] = {}
history: dict[str, list[float]] = {}
line_data: dict[str, dolfinx.fem.Function] = {}
line_history: dict[str, list[np.ndarray]] = {}

def register_function(name: str, function: dolfinx.fem.Function):
    function.name = name

    functions[name] = function
    if name not in history:
        history[name] = []

def register_line_data(name: str, function: dolfinx.fem.Function, set_point=None):
    """Register a function to collect data along the line points."""
    line_data[name] = function
    if name not in line_history:
        line_history[name] = []
#endregion    

''' DEFINE FUNCTION SPACES AND TRIAL FUNCTIONS '''
#region
QUAD_DEGREE = 8  # The number of points used in the quadrature scheme.

# Create a second order Lagrange function space for displacement
# This is basically the the function space to represent the tangent space without the base space
P_u = basix.ufl.element(
    family="Lagrange",  # Type of functions (Lagrange polynomials)
    cell=str(mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
    degree=2,  # Polynomial degree of functions
    shape=(mesh.geometry.dim,),  # Dimension of functions (3D vector in this case)
)

# This is basically the union of the tangent space and the base space
u_space = dolfinx.fem.functionspace(mesh, P_u)

# Create a first order Lagrange function space for pressure
P_p = basix.ufl.element(
    family="Lagrange",  # Type of functions (Lagrange polynomials)
    cell=str(mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
    degree=1,  # Polynomial degree of functions
    shape=(),  # Dimension of functions (scalar in this case)
)

# This creates a union of the base space and the tangent space
p_space = dolfinx.fem.functionspace(mesh, P_p)

u = dolfinx.fem.Function(u_space, name="u")
v = ufl.TestFunction(u_space)
du = ufl.TrialFunction(u_space)
p = dolfinx.fem.Function(p_space, name="p")
q = ufl.TestFunction(p_space)
dp = ufl.TrialFunction(p_space)

scalar_element = basix.ufl.element(
    family="CG",
    cell=str(mesh.ufl_cell()),
    degree=5,
    shape=(),
    discontinuous=True,
)
scalar_space = dolfinx.fem.functionspace(mesh, scalar_element)

stress_ff = dolfinx.fem.Function(scalar_space, name="stress_ff")
stress_tt = dolfinx.fem.Function(scalar_space, name="stress_tt")
stress_nn = dolfinx.fem.Function(scalar_space, name="stress_nn")
J = dolfinx.fem.Function(scalar_space, name="J")
u_mag = dolfinx.fem.Function(scalar_space, name="displacement_magnitude")

g_2 = dolfinx.fem.Function(scalar_space, name="g_2")

# REGISTER FUNCTIONS FOR DATA STORAGE
register_function("u", u)  # saves values for displacement
register_function("p", p)  # saves values for pressure
register_function("stress_ff", stress_ff)  # saves values for stress
register_function("stress_tt", stress_tt)  # saves values for stress
register_function("stress_nn", stress_nn)  # saves values for stress

register_line_data("stress_ff", stress_ff)
register_line_data("stress_nn", stress_nn)
register_line_data("stress_tt", stress_tt)
register_line_data("g2", g_2)
register_line_data("p", p)
# register_line_data("displacement_magnitude", u_mag)

# Group functions by their function space element hash
sorted_functions = defaultdict(list)
for f in functions.values():
    f_hash = f.function_space.element.basix_element.hash()
    sorted_functions[f_hash].append(f)

writers = []
n = 1
for funcs in sorted_functions.values():
    if len(funcs) == 1:
        # If we have only one function, use its name
        filename = output_dir / f"{funcs[0].name}.bp"
    else:
        filename = output_dir / f"variables_{n}.bp"
        n += 1

    writers.append(
        dolfinx.io.VTXWriter(
            comm,
            filename,
            funcs,
            engine="BP4",
        ),
    )

# for name, function in line_data.items():
#     values = scifem.evaluate_function(function, line_points)
#     line_history[name].append(values)

#endregion

''' INITIAL CONDITIONS '''
#region
# Define parameters for analytical solutions
R_o = 2.0  # Outer radius
R_i = 1.0  # Inner radius in reference configuration (same as geo inner_radius)
g_2.x.array[:] = 1.1
c = -0.05  # Constant equal to Neumann boundary condition (0.05)
dt = 0.025  # Time step size
set_point = 0.5  # Set point for growth
#endregion


''' KINEMATICS '''
#region
F = ufl.variable(ufl.grad(u) + ufl.Identity(2))
G = ufl.outer(r0, r0) + g_2 * ufl.outer(f0, f0)
A = ufl.variable(F * ufl.inv(G))
#endregion

P_g = basix.ufl.element(
    family="Lagrange",  # Type of functions (Lagrange polynomials)
    cell=str(mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
    degree=2,  # Polynomial degree of functions
    shape=(mesh.geometry.dim, mesh.geometry.dim),  # Dimension of functions (3D vector in this case)
)

# This is basically the union of the tangent space and the base space
G_space = dolfinx.fem.functionspace(mesh, P_g)
G_func = dolfinx.fem.Function(G_space, name="G")
G_expression = dolfinx.fem.Expression(G, G_space.element.interpolation_points())
G_func.interpolate(G_expression)

# scifem.evaluate_function(G_func, mesh.geometry.x[0:1, 0:2])

''' BOUNDARY CONDITIONS '''
#region
N = ufl.FacetNormal(mesh)  # Normal vector on the boundary of the mesh

ds = ufl.Measure(
    "ds",  # ???
    domain=mesh,  # Domain of the measure
    subdomain_data=facet_tags,  # Boundary we are interested in
    metadata={"quadrature_degree": QUAD_DEGREE},  # Quadrature degree for the measure
)

# Pressure on the inside (Neumann)
traction = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(-c))
# Pressure value on the inside of the cylinder
# Neumann boundary condition on the inside surface (pulling back the surface element)
neumann = ufl.inner(v, traction * ufl.det(F) * ufl.inv(F).T * N) * ds(20)

# Robin on the outside
N = ufl.FacetNormal(mesh)
spring = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.00001))
robin_value = ufl.inner(spring * u, N)
robin = ufl.inner(robin_value * v, ufl.det(F) * ufl.inv(F).T * N) * ds(10)

dx = ufl.dx(metadata={"quadrature_degree": QUAD_DEGREE})
#endregion

''' MATERIAL MODEL '''
#region
mu = 1.0  # Shear modulus
C = A.T * A  # Right Cauchy-Green deformation tensor
I1 = ufl.tr(C)  # First invariant of the right Cauchy-Green tensor
psi = (mu / 2) * (I1 - 3)  # Neo-Hookean strain energy function # / 2.0
stress = ufl.diff(psi, F)
#endregion

''' WEAK FORMULATION AND SOLVER '''
#region
pressure_term = p * (ufl.det(A) - 1) * dx
elasticity_term = ufl.inner(stress, ufl.grad(v)) * dx
cauchy = (stress + p * ufl.inv(F.T)) * F.T / ufl.det(F)  # This is cauchy stress because stress = dPsi/dF * dF / dG = dPsi/dF * inv(G)

J_expr = dolfinx.fem.Expression(ufl.det(A), scalar_space.element.interpolation_points())

F0 = (
    elasticity_term + ufl.derivative(pressure_term, u, v) + neumann + robin
)  #  ufl.derivative(rigid_form, u, v) +ufl.derivative(psi * dx, u, v)

F1 = ufl.derivative(psi * dx, p, q) + ufl.derivative(
    pressure_term,
    p,
    q,
)

R = [F0, F1]  # , F2]
dR = [
    [ufl.derivative(F0, u, du), ufl.derivative(F0, p, dp)],
    [ufl.derivative(F1, u, du), ufl.derivative(F1, p, dp)],
]

petsc_options = {
    "ksp_type": "preonly",  # direct solver
    "pc_type": "lu",  # LU preconditioner
    "pc_factor_mat_solver_type": "mumps",  # paralellization
    "ksp_monitor": None,  # see output during solve
}
solver = scifem.NewtonSolver(
    R,
    dR,
    [u, p],
    bcs=[],
    max_iterations=25,
    petsc_options=petsc_options,
)

solver.solve()

# for name, function in line_data.items():
#     values = scifem.evaluate_function(function, line_points)
#     line_history[name].append(values)

for writer in writers:
    writer.write(1)