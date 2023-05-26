# ipde_demo
Simplified Demo of Inhomogeneous PDE Solvers

## Installation
Assuming you have conda installed in a standard way, installation should be as easy as running:
```bash
source install.sh NAME
```
Where NAME is the name you choose for a new virtual environment that will be created for running the demos. Upon installation, this script will run example1.py with a coarse discretization.

## Examples
There are two examples contained here, which show off a few of the capabilities. The first solves a Poisson problem on a simple interior domain, run as:
```bash
conda activate NAME
python example1.py N
```
Where NAME is the name of the virtual environment you created during installation and N is the number of points used to discretize the boundary. As convergence is exponential, error decreases rapidly as N is increased, and 500 points are all that is required to get to nearly machine precision. Note that the code relies on numba acceleration, meaining it is not fast on the first execution, but will be much faster on repeated executions, once jit compilation is completed. The second (example2.py) solves the same problem on a more complicated domain with an internal occlusion, and is run in the same way. To see how the problem is constructed and solved, I recommend running the code interactively, instead, in ipython, for example.

