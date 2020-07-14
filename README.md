# GP3: A Sampling-based Analysis Framework for Gaussian Processes  
This code accompanies the  work "GP3: A Sampling-based Analysis Framework for Gaussian Processes " by Armin Lederer, Markus Kessler and Sandra Hirche accepted to the 21st IFAC World Congress 2020. The code is as is and is not maintained. The code has been developed with Python 3.6.3 and requires a GPU supporting CUDA as well as the following Python packages: numpy, numba and matplotlib. 

## Overview of Python scripts:

#### General (reusable, non-problem specific) GPU calculation scripts:
See the function headers inside the specific scripts for detailed information about required inputs and returns of each function.

- **SquaredExp_GPU.py**: GPU parallelized functions to evaluate the decrease condition of a given Gaussian Process (GP) mean function trained on a cost function of a dynamical system with the squared exponential kernel.
  
  Based on the decrease region and GP mean additionally estimate the region of attraction of the system. 
- **Matern32.py**: GPU parallelized functions to evaluate the decrease condition of a given Gaussian Process (GP) mean function trained on a cost function of a dynamical system with the matern32 kernel.

  Based on the decrease region and GP mean additionally estimate the region of attraction of the system. 
- **Matern52.py**: GPU parallelized functions to evaluate the decrease condition of a given Gaussian Process (GP) mean function trained on a cost function of a dynamical system with the matern52 kernel.
  
  Based on the decrease region and GP mean additionally estimate the region of attraction of the system. 
 
 
   
#### Example demonstration scripts (problem specific):
- **main_example.py**: This is the main function for running the example shown in the publication mentioned at the top. Import data from the from the 'MA32bus_example.npz' file, which is used to store the GP trained on a cost function for the example system. Calculate the decrease region and region of attraction using the general scripts listed above.

  This script should give an idea how the general GPU scripts above can be used in a specific problem setting.

  It also involves an example refinement process for multiple sampling resolutions.
  Again, see the function headers and commentary in the code for detailed information about the usage.

- **parallel_ode_solver_example.py**: CPU parallelized calculation of single step solutions of the system differential equation from the specific demo example required for the decrease condition evaluation. This must be adapted for any different dynamical system.




