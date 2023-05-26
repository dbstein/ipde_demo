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

# number of points discretizing the inner boundary
nb2 = 200
nb1 = 2*nb2

################################################################################
# functions defining solution, forces, etc for test problem

solution_func = lambda x, y: np.exp(np.sin(x))*np.sin(2*y) + np.log(0.1 + np.cos(y)**2)
uxx_func = lambda x, y: (np.cos(x)**2 - np.sin(x))*np.exp(np.sin(x))*np.sin(2*y)
uyy_func = lambda x, y: -4*np.exp(np.sin(x))*np.sin(2*y) - 1/(0.1 + np.cos(y)**2)**2*(2*np.cos(y)*np.sin(y))**2 - 1/(0.1 + np.cos(y)**2)*(2*np.cos(y)**2 - 2*np.sin(y)**2)
force_func = lambda x, y: uxx_func(x, y) + uyy_func(x, y)

################################################################################
# Setup Poisson Solver

# construct boundary
bdy1 = GSB(c=star(nb1, r=2.5, a=0.2, f=7))
bdy2 = GSB(c=star(nb2, a=0.15, f=5))
# construct embedded boundary object
ebdy1 = EmbeddedBoundary(bdy1, True)
ebdy2 = EmbeddedBoundary(bdy2, False)
# place into a collection
ebdyc = EmbeddedBoundaryCollection([ebdy1, ebdy2])
# generate a background grid, register it, construct bump function
grid = ebdyc.generate_grid(ready_bump=True)
# generate a poisson solver
solver = PoissonSolver(ebdyc)

################################################################################
# Setup Laplace Solver

# simple second-kind scheme for singular density
A11 = Laplace_Layer_Singular_Form(bdy1, ifdipole=True) - 0.5*np.eye(bdy1.N)
A22 = Laplace_Layer_Singular_Form(bdy2, ifcharge=True, ifdipole=True) + 0.5*np.eye(bdy2.N)
A12 = Laplace_Layer_Form(bdy1, bdy2, ifdipole=True)
A21 = Laplace_Layer_Form(bdy2, bdy1, ifcharge=True, ifdipole=True)
A = np.bmat([[A11, A21], [A12, A22]])
A_LU = sp.linalg.lu_factor(A)
# generate close evaluation quadratures
qfs1 = Laplace_QFS(bdy1, interior=True, slp=False, dlp=True)
qfs2 = Laplace_QFS(bdy2, interior=False, slp=True, dlp=True)

################################################################################
# Solve Poisson problem

# we assume we have sufficient knowledge of f
f = EmbeddedFunction(ebdyc, function=force_func)
# and of the boundary condition
bc = BoundaryFunction(ebdyc, function=solution_func)
# here, we just construct them from analytic functions

# solve for a particular solution to the equation
ue = solver(f, tol=1e-12)
# this does not satisfy the boundary condition.  get the boundary values
bv = solver.get_boundary_values(ue)
# solve for the density that defines a homogeneous solution that fixes the boundary conditions
tau = sp.linalg.lu_solve(A_LU, bc-bv)
# separate into lists associated with each boundary
taul = ebdyc.v2l(tau)
# get an effective density which allows us to evaluate this everywhere in the domain
sigma1 = qfs1([taul[0],])
sigma2 = qfs2([taul[1],taul[1]])
# evaluate this onto everywhere it needs to go
ue += Laplace_Layer_Apply(qfs1.source, ebdyc.grid_and_radial_pts, charge=sigma1)
ue += Laplace_Layer_Apply(qfs2.source, ebdyc.grid_and_radial_pts, charge=sigma2)
# the above two Layer potentials could be reduced to 1 FMM call, for speed

################################################################################
# Analyze the solution to this problem

# analytic solution
ua = EmbeddedFunction(ebdyc, function=solution_func)

# compute the error
err = np.abs(ue - ua)
print("Error is : {:0.3e}".format(err.max()))

# plot the error
fig, ax = plt.subplots()
clf = err.plot(ax)
fig.colorbar(clf)
