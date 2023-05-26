import sys
import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import pybie2d
from ipde.embedded_boundary_tr import EmbeddedBoundary
from ipde.ebdy_collection_tr import EmbeddedBoundaryCollection
from ipde.embedded_function import EmbeddedFunction, BoundaryFunction
from ipde.solvers.multi_boundary.poisson import PoissonSolver
from qfs.laplace_qfs import Laplace_QFS
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply

import warnings
warnings.filterwarnings('ignore')

# number of points discretizing the boundary
try:
	nb = int(sys.argv[1])
except:
	nb = 200

print("\nSolving poisson example with ", nb, "boundary points.\n")

################################################################################
# functions defining solution, forces, etc for test problem

solution_func = lambda x, y: np.exp(np.sin(x))*np.sin(2*y) + np.log(0.1 + np.cos(y)**2)
uxx_func = lambda x, y: (np.cos(x)**2 - np.sin(x))*np.exp(np.sin(x))*np.sin(2*y)
uyy_func = lambda x, y: -4*np.exp(np.sin(x))*np.sin(2*y) - 1/(0.1 + np.cos(y)**2)**2*(2*np.cos(y)*np.sin(y))**2 - 1/(0.1 + np.cos(y)**2)*(2*np.cos(y)**2 - 2*np.sin(y)**2)
force_func = lambda x, y: uxx_func(x, y) + uyy_func(x, y)

################################################################################
# Setup Poisson Solver

# construct boundary
bdy = GSB(c=star(nb, a=0.15, f=5))
# construct embedded boundary object
ebdy = EmbeddedBoundary(bdy, True)
# place into a collection
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# generate a background grid, register it, construct bump function
grid = ebdyc.generate_grid(ready_bump=True)
# generate a poisson solver
solver = PoissonSolver(ebdyc)

################################################################################
# Setup Laplace Solver

# simple second-kind scheme for singular density
A = Laplace_Layer_Singular_Form(bdy, ifdipole=True) - 0.5*np.eye(bdy.N)
A_LU = sp.linalg.lu_factor(A)
# generate close evaluation quadratures
qfs = Laplace_QFS(bdy, interior=True, slp=False, dlp=True)

################################################################################
# Solve Poisson problem

# we assume we have sufficient knowledge of f
f = EmbeddedFunction(ebdyc, function=force_func)
# and of the boundary condition
bc = BoundaryFunction(ebdyc, function=solution_func)
# here, we just construct them from analytic functions

# solve for a particular solution to the equation
ue = solver(f, tol=1e-12, verbose=True)
# this does not satisfy the boundary condition.  get the boundary values
bv = solver.get_boundary_values(ue)
# solve for the density that defines a homogeneous solution that fixes the boundary conditions
tau = sp.linalg.lu_solve(A_LU, bc-bv)
# get an effective density which allows us to evaluate this everywhere in the domain
sigma = qfs([tau,])
# evaluate this onto everywhere it needs to go
ue += Laplace_Layer_Apply(qfs.source, ebdyc.grid_and_radial_pts, charge=sigma)

################################################################################
# Analyze the solution to this problem

# analytic solution
ua = EmbeddedFunction(ebdyc, function=solution_func)

# compute the error
err = np.abs(ue - ua)
print("Maximum Error is : {:0.3e}".format(err.max()))

# plot the error
fig, ax = plt.subplots()
clf = err.plot(ax)
fig.colorbar(clf)
plt.show(block=True)
