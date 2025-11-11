import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import ufl
import basix
import numpy as np
from collections import defaultdict
from pathlib import Path
import scifem
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

''' PLOTTING '''
def plot_line(r_i, line_history, line_points, time, analytical_solutions=None):
    """Plot current values of all variables along the line defined by line_points.

    Args:
        analytical_solutions: Dict with variable names as keys and analytical functions as values.
                             Each function should take line_points as input and return values.
    """

    # Get all variable names from line history
    variables = list(line_history.keys())
    num_vars = len(variables)

    # Determine indices for 10 evenly spaced time points
    if variables:
        num_time_steps = len(line_history[variables[0]])
        num_plots = min(20, num_time_steps)  # Don't try to plot more than available
        if num_plots > 0:
            indices_to_plot = np.linspace(0, num_time_steps - 1, num_plots, dtype=int)
        else:
            indices_to_plot = []
    else:
        indices_to_plot = []

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
            for t_idx in indices_to_plot:
                time_step_values = line_history[var_name][t_idx]
                color = plt.cm.viridis(t_idx / max(1, len(line_history[var_name]) - 1))
                timestamp = time[t_idx] if t_idx < len(time) else t_idx

                # Only add labels for the first subplot to avoid duplicate legend entries
                label = f"t = {timestamp:.3f}" if idx == 0 else None

                ax.plot(
                    distances,
                    time_step_values,
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.7,
                    label=label,
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
    

    # Set x-label for bottom plot
    axes[-1].set_xlabel("Distance along line")
    fig.legend(fontsize="small", loc="center right")

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)

    output_path = output_dir / "variables_line_spatial.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

''' CREATE FIBERS '''
P_fiber = basix.ufl.element(
    family="CG",  # Type of functions (Lagrange polynomials)
    cell=str(mesh.ufl_cell()),  # Type of cell (e.g., triangle, square, etc.)
    degree=1,  # Polynomial degree of functions
    shape=(mesh.geometry.dim,),  # Dimension of functions (2D vector in this case)
)

fiber_space = dolfinx.fem.functionspace(mesh, P_fiber)

x, y, z = fiber_space.tabulate_dof_coordinates().T
r = np.sqrt(x**2 + y**2)
e_r = np.array([x / r, y / r])
e_theta = np.array([-e_r[1], e_r[0]])  # 90 degree rotation of e_r


# points on a radial line from inner to outer radius
line_points = np.array([[r * np.cos(0), r * np.sin(0)]
                        for r in np.linspace(1.0, 2.0, 50)], dtype=np.float64)

# line_points = np.array([[r * np.cos(0), r * np.sin(0)]
#                         for r in np.linspace(1.0, 2.0, 10)
#                         for theta in np.linspace(0, 2 * np.pi, 36)], dtype=np.float64)

f0 = dolfinx.fem.Function(fiber_space, name="f0")
r0 = dolfinx.fem.Function(fiber_space, name="r0")

f0.x.array[:] = np.array(e_theta).T.reshape(-1)
r0.x.array[:] = np.array(e_r).T.reshape(-1)

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
stress_nn = dolfinx.fem.Function(scalar_space, name="stress_nn")
J = dolfinx.fem.Function(scalar_space, name="J")
u_mag = dolfinx.fem.Function(scalar_space, name="displacement_magnitude")

g_2 = dolfinx.fem.Function(scalar_space, name="g_2")

# REGISTER FUNCTIONS FOR DATA STORAGE
register_function("u", u)  # saves values for displacement
register_function("p", p)  # saves values for pressure
register_function("Hoop Stress", stress_ff)  # saves values for stress
register_function("Radial Stress", stress_nn)  # saves values for stress

register_line_data("Hoop Stress", stress_ff)
register_line_data("Radial Stress", stress_nn)
register_line_data("Cumulative Hoop Growth", g_2)
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
g_2.x.array[:] = 1.0
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
inner_neumann = ufl.inner(v, traction * ufl.det(F) * ufl.inv(F).T * N) * ds(20)

# Robin on the outside
N = ufl.FacetNormal(mesh)
spring = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0001))
robin_value = ufl.inner(spring * u, N)
outer_robin = ufl.inner(robin_value * v, ufl.det(F) * ufl.inv(F).T * N) * ds(10)


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

stress_ff_expr = dolfinx.fem.Expression(  # hoop stress
    ufl.inner(cauchy * f0, f0),
    scalar_space.element.interpolation_points(),
)

stress_nn_expr = dolfinx.fem.Expression(
    ufl.inner(cauchy * r0, r0),
    scalar_space.element.interpolation_points(),
)

J_expr = dolfinx.fem.Expression(ufl.det(A), scalar_space.element.interpolation_points())

g2_expr = dolfinx.fem.Expression(
    g_2 * (dt * (stress_ff - set_point) / set_point + 1),
    scalar_space.element.interpolation_points(),
)

F0 = (
    elasticity_term
    + ufl.derivative(pressure_term, u, v)
    + inner_neumann  # + outer_robin
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
    "mat_mumps_icntl_24": 1,  # Zero pivot detection
    "mat_mumps_icntl_25": 0,  # Which nullspace to extract
    "mat_mumps_icntl_4": 1,  # Verbosity
    "mat_mumps_icntl_2": 1,  # std out blaaah
    "mat_mumps_cntl_3": 1e-6,  # Threshold factor
}
solver = scifem.NewtonSolver(
    R,
    dR,
    [u, p],
    bcs=[],
    max_iterations=25,
    petsc_options=petsc_options,
)

n = 10 #2**9
for i in range(n):

    print(f"Starting growth step {i} of {n}")

    solver.solve()

    stress_ff.interpolate(stress_ff_expr)
    stress_nn.interpolate(stress_nn_expr)
    J.interpolate(J_expr)
    # u_mag.interpolate(u_mag_expr)

    g_2.interpolate(g2_expr)
    
    for name, function in line_data.items():
        values = scifem.evaluate_function(function, line_points)
        line_history[name].append(values)

    for writer in writers:
        writer.write(i)
    

time = [dt * i for i in range(n)]

# print inner radius of deformation configuration, use u to get the current inner radius
new_geo_inner = scifem.evaluate_function(u, np.array([line_points[0]]))
new_geo_outer = scifem.evaluate_function(u, np.array([line_points[-1]]))
inner_rad = R_i + np.sqrt(new_geo_inner[0, 0] ** 2 + new_geo_inner[0, 1] ** 2)
outer_rad = R_o + np.sqrt(new_geo_outer[-1, 0] ** 2 + new_geo_outer[-1, 1] ** 2)
print(f"Inner radius in deformation configuration: {inner_rad}")
print(f"Outer radius in deformation configuration: {outer_rad}")
# print(f"Outer radius in deformation configuration: {outer_radius:.3f}")
plot_line(r_i=1.19, line_history=line_history, time=time, line_points=line_points)