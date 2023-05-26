conda create -n $1
conda activate $1
conda install -y numpy scipy numba numexpr matplotlib ipython shapely pip
pip install pygeos
pip install finufft
pip install fast_interp/
pip install function_generator/
pip install ipde/
pip install near_finder/
pip install pybie2d/
pip install qfs/

python example1.py 100
