# Simulation code

Code underlying the manuscript "Principles underlying the complex dynamics of temperature entrainment by a circadian clock"  

# Usage

In order to run simulations faster, we implemented ordinary differential equations in FORTRAN and connected the FORTRAN code with the Python simulation scripts via F2PY. In order to successfully run our Python scripts please run '''f2py -c ForcedHongModel.f90 -m ForcedHongModel''' within your shell beforehand.


# Requirement
Python
* numpy
* scipy
* astropy
* matplotlib
* seaborn
* pickle
