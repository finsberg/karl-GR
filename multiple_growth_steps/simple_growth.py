import logging
from collections import defaultdict
from pathlib import Path

from mpi4py import MPI

import basix
import dolfinx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import ufl

import scifem
from growth.geometry import load_cylinder_geometry

logging.basicConfig(level=logging.INFO)

comm = MPI.COMM_WORLD  # ???

output_dir = Path(
    "simple_growth",
)  # Specify the folder where all the data should be saved
output_dir.mkdir(parents=True, exist_ok=True)  # Make the folder if it doesn't exist

''' CREATE GEOMETRY '''
#region
geo = load_cylinder_geometry(
    comm,
    char_length=0.125,
    height=0.5,
    inner_radius=1,
    outer_radius=2,
    fiber_angle_endo=0,
    fiber_angle_epi=0,
    # fiber_space="DG_5",
)  # Load cylinder geometry from Henrik's code
geo.save(output_dir)

z = (geo.z_range[0] + geo.z_range[1]) / 2
tol = 0.0
line_points = np.array(
    [[r * np.cos(0), r * np.sin(0), z] for r in np.linspace(geo.r_range[0], geo.r_range[1], 16)],
)
#endregion

''' PLOTTING '''
def plot_line(r_i, line_history, line_points, analytical_solutions=None):
    """Plot current values of all variables along the line defined by line_points.

    Args:
        analytical_solutions: Dict with variable names as keys and analytical functions as values.
                             Each function should take line_points as input and return values.
    """

    # Get all variable names from line history
    variables = list(line_history.keys())
    num_vars = len(variables)

    # Calculate grid dimensions - always use 1 column
    ncols = 1
    nrows = num_vars  # Each variable gets its own row

    # Evenly spaced grid from 0 to 1 in length on line_poin
    distances = line_points[:, 0]

    # Create subplots grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True)

    # Handle case where there's only one variable
    if num_vars == 1:
        axes = [axes]
    else:
        axes = axes.reshape(-1)  # Flatten to 1D array for single column

    for idx, var_name in enumerate(variables):
        ax = axes[idx]
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="y")

        # Plot numerical solution
        if len(line_history[var_name]) > 0:
            for t_idx, time_step_values in enumerate(line_history[var_name][:]):
                color = plt.cm.viridis(t_idx / max(1, len(line_history[var_name]) - 1))
                # timestamp = time[t_idx + 2] if t_idx + 2 < len(time) else t_idx + 2

                ax.plot(
                    distances,
                    time_step_values,
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Numerical",  # t = {timestamp:.3f}",
                )

        # Plot analytical solution if provided
        if analytical_solutions and var_name in analytical_solutions:
            print("var_name", var_name)
            analytical_values = analytical_solutions[var_name](line_points, r_i)
            ax.plot(
                distances,
                analytical_values,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="Analytical",
            )

        ax.set_ylabel(var_name)
        ax.set_title(f"{var_name} (t = {time[-1]:.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    # Set x-label for bottom plot
    axes[-1].set_xlabel("Distance along line")

    plt.tight_layout()

    output_path = output_dir / "variables_line_spatial.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


''' DEFINE FUNCTION SPACES AND TRIAL FUNCTIONS '''
#region
QUAD_DEGREE = 8  # The number of points used in the quadrature scheme.

# Create a second order Lagrange function space for displacement
# This is basically the the function space to represent the tangent space without the base space
P_u = basix.ufl.element(
    family="Lagrange",  # Type of functions (Lagrange polynomials)
    cell=str(geo.mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
    degree=2,  # Polynomial degree of functions
    shape=(geo.mesh.geometry.dim,),  # Dimension of functions (3D vector in this case)
)

# This is basically the union of the tangent space and the base space
u_space = dolfinx.fem.functionspace(geo.mesh, P_u)

# Create a first order Lagrange function space for pressure
P_p = basix.ufl.element(
    family="Lagrange",  # Type of functions (Lagrange polynomials)
    cell=str(geo.mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
    degree=1,  # Polynomial degree of functions
    shape=(),  # Dimension of functions (scalar in this case)
)

# This creates a union of the base space and the tangent space
p_space = dolfinx.fem.functionspace(geo.mesh, P_p)

u = dolfinx.fem.Function(u_space, name="u")
v = ufl.TestFunction(u_space)
du = ufl.TrialFunction(u_space)
p = dolfinx.fem.Function(p_space, name="p")
q = ufl.TestFunction(p_space)
dp = ufl.TrialFunction(p_space)

scalar_element = basix.ufl.element(
    family="DG",
    cell=str(geo.mesh.ufl_cell()),
    degree=5,
    shape=(),
    discontinuous=True,
)
scalar_growth_space = dolfinx.fem.functionspace(geo.mesh, scalar_element)

stress_ff = dolfinx.fem.Function(scalar_growth_space, name="stress_ff")
stress_tt = dolfinx.fem.Function(scalar_growth_space, name="stress_tt")
stress_nn = dolfinx.fem.Function(scalar_growth_space, name="stress_nn")
J = dolfinx.fem.Function(scalar_growth_space, name="J")
u_mag = dolfinx.fem.Function(scalar_growth_space, name="displacement_magnitude")

#endregion

''' INITIAL CONDITIONS '''
#region
# Define parameters for analytical solutions
R_o = 2.0  # Outer radius
R_i = 1.0  # Inner radius in reference configuration (same as geo inner_radius)
g_1 = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(1.6))  # Growth in fiber direction
g_2 = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(1.2))  # Growth in cross-fiber direction
c = -0.05  # Constant equal to Neumann boundary condition (0.05)
#endregion

''' KINEMATICS '''
#region
F = ufl.variable(ufl.Identity(3) + ufl.grad(u))
G = g_1 * ufl.outer(geo.n, geo.n) + g_2 * ufl.outer(geo.f, geo.f) + ufl.outer(geo.s, geo.s)
A = ufl.variable(F * ufl.inv(G))
#endregion

''' BOUNDARY CONDITIONS '''
#region
N = ufl.FacetNormal(geo.mesh)  # Normal vector on the boundary of the mesh

ds = ufl.Measure(
    "ds",  # ???
    domain=geo.mesh,  # Domain of the measure
    subdomain_data=geo.facet_tags,  # Boundary we are interested in
    metadata={"quadrature_degree": QUAD_DEGREE},  # Quadrature degree for the measure
)

# Pressure on the inside (Neumann)
traction = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(-c))
# Pressure value on the inside of the cylinder
# Neumann boundary condition on the inside surface (pulling back the surface element)
neumann = ufl.inner(v, traction * N) * ds(  # ufl.det(F) * ufl.inv(F).T *
    geo.markers["INSIDE"][0],
)

# Robin on the outside
N = ufl.FacetNormal(geo.mesh)
spring = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.00001))
robin_value = ufl.inner(spring * u, N)
robin = ufl.inner(robin_value * v, N) * ds(  # ufl.det(F) * ufl.inv(F).T *
    geo.markers["OUTSIDE"][0],
)
# Dirichlet boundary conditions
fdim = 2
bottom_facets = geo.facet_tags.find(geo.markers["BOTTOM"][0])
bottom_dofs = dolfinx.fem.locate_dofs_topological(u_space.sub(2), fdim, bottom_facets)

top_facets = geo.facet_tags.find(geo.markers["TOP"][0])
top_dofs = dolfinx.fem.locate_dofs_topological(u_space.sub(2), fdim, top_facets)

# Clamped boundary conditions on the bottom and top surfaces
bc_bottom = dolfinx.fem.dirichletbc(0.0, bottom_dofs, u_space.sub(2))
bc_top = dolfinx.fem.dirichletbc(0.0, top_dofs, u_space.sub(2))
bcs = [bc_bottom, bc_top]

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
cauchy = stress + p * ufl.inv(F.T)
# stress = stress + p * ufl.inv(A).T * ufl.det(A)   # PK1 stress
stress_ff_expr = dolfinx.fem.Expression(  # hoop stress
    ufl.inner(cauchy * geo.f, geo.f),
    scalar_growth_space.element.interpolation_points(),
)
stress_tt_expr = dolfinx.fem.Expression(
    ufl.inner(cauchy * geo.s, geo.s),
    scalar_growth_space.element.interpolation_points(),
)
stress_nn_expr = dolfinx.fem.Expression(
    ufl.inner(cauchy * geo.n, geo.n),
    scalar_growth_space.element.interpolation_points(),
)
J_expr = dolfinx.fem.Expression(ufl.det(A), scalar_growth_space.element.interpolation_points())
u_mag_expr = dolfinx.fem.Expression(
    ufl.sqrt(ufl.dot(u, u)),
    scalar_growth_space.element.interpolation_points(),
)

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
    bcs=bcs,
    max_iterations=25,
    petsc_options=petsc_options,
)
#endregion

solver.solve()

with dolfinx.io.VTXWriter(
    comm,
    output_dir / "solution.bp",
    [u],
    engine="BP4",
) as writer:
    writer.write(0.0)

stress_ff.interpolate(stress_ff_expr)
stress_tt.interpolate(stress_tt_expr)
stress_nn.interpolate(stress_nn_expr)
J.interpolate(J_expr)
u_mag.interpolate(u_mag_expr)

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


register_function("u", u)  # saves values for displacement
register_function("p", p)  # saves values for pressure
register_function("stress_ff", stress_ff)  # saves values for stress
register_function("stress_tt", stress_tt)  # saves values for stress
register_function("stress_nn", stress_nn)  # saves values for stress

register_line_data("stress_ff", stress_ff)
register_line_data("stress_nn", stress_nn)
# register_line_data("stress_tt", stress_tt)
register_line_data("p", p)
register_line_data("|A|", J)
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

for name, function in line_data.items():
    values = scifem.evaluate_function(function, line_points)
    line_history[name].append(values)

for writer in writers:
    writer.write(1)

time = [1]
# print inner radius of deformation configuration, use u to get the current inner radius
new_geo_inner = scifem.evaluate_function(u, np.array([line_points[0]]))
new_geo_outer = scifem.evaluate_function(u, np.array([line_points[-1]]))
inner_rad = R_i + np.sqrt(new_geo_inner[0, 0] ** 2 + new_geo_inner[0, 1] ** 2)
outer_rad = R_o + np.sqrt(new_geo_outer[-1, 0] ** 2 + new_geo_outer[-1, 1] ** 2)
print(f"Inner radius in deformation configuration: {inner_rad}")
print(f"Outer radius in deformation configuration: {outer_rad}")
# print(f"Outer radius in deformation configuration: {outer_radius:.3f}")
plot_line(r_i=1.101, line_history=line_history, line_points=line_points)
