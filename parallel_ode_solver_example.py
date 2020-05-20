import numpy
from scipy.integrate import ode
import math
import multiprocessing
from multiprocessing import Pool


# Dynamical system parameters
m1 = 12
d1 = 20
theta1 = math.asin(0.05)
a12 = 10


def system_ode(t, x):
    """System differential equation

    Args:
        t(float): Simulation time.
        x: System states: x[0] = state x_1; x[1] = state x_2.

    Returns:
        (ndarray): The differential equation for the dynamics of the system states [x_1_dot, x_2_dot]"""
    return numpy.array([x[1], 1/m1*(-d1*x[1] - a12*(math.sin(theta1+x[0])) - math.sin(theta1))])


def solveodepar(gridx, gridy):
    """Main function calculating ode solutions on analytic grid CPU parallelized

    Args:
        gridx (ndarray): Coordinates of interval centers on the first axis
        gridy (ndarray): Coordinates of interval centers on the second axis

    Returns:
        res (ndarray):
            Solution of system differential equation at all interval centers. Shape = (gridx.size, gridy.size).
        """

    gridx2, gridx1 = numpy.meshgrid(gridy, gridx)           # create 2D grid
    gridx1 = gridx1.flatten()                               # flatten grid
    gridx2 = gridx2.flatten()                               #
    grid = numpy.array([gridx1, gridx2]).transpose(1, 0)    # flattened array of grid coordinates

    threads = multiprocessing.cpu_count()                   # nr. of available CPU cores
    pool = Pool(threads)                                    # create threads
    xnext = numpy.array(pool.map(simulate, grid))               # run solution of ode in parallel
    xnext = xnext.reshape((gridx.shape[0], gridy.shape[0], 2))  # reshape flat results to 2D grid
    return xnext


def simulate(state):
    """Perform one simulation of ode
    Args:
        state (tuple): Coordinates of a system state
    Returns:
        sol (tuple): Next system state after ode integration
    """
    t0, t1 = 0, 0.01                        # start time, end time (only one step)
    solver = ode(system_ode)                # create ode instance
    solver.set_integrator("dopri5")         # set integrator method
    solver.set_initial_value(state, t0)     # set initial value
    sol = solver.integrate(t1)              # calculate one solution of ode
    return sol
