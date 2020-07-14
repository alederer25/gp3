import numpy
import matplotlib
import matplotlib.pyplot as plt
from parallel_ode_solver_example import solveodepar

# # # Global parameters # # #

# Plot results for testing/demonstration purposes. This reduces the sampling density since matplotlib cannot handle
# possibly billions of points and reduces the required runtime.
# If set to 1 note that the results are not theoretically justified, but no difference in the plot is observable
# due to the limited plot resolution. Please use this feature carefully and only for quick testing/demonstration!
DO_PLOT = 1

# Kernel selection (select the kernel used in the Gaussian Process):
KERNEL = "ardmatern32"
# KERNEL = "ardmatern52"
# KERNEL = "ardsquaredexponential"

# Initialize grid constants
"""The analyzed region will range from [xmin, xmax] on the first axis and [ymin,ymax] on the second axis"""
xmin = -5
xmax = 5
ymin = -5
ymax = 5

"""b is the grid constant. The analyzed region will be covered by squares ranging from [xcenter-b, xcenter+b] on the
first axis and [ycenter-b, ycenter+b] on the second axis. This parameter can be iteratively reduced in a refinement
process using the parameters below. The calculation on squares where the decrease condition was already fulfilled in
an earlier iteration is skipped in further refinement iterations. This can improve the performance depending on
the specific problem."""
bstart = numpy.float64(1e-06)   # this is the starting b in the refinement process
bmin = numpy.float64(1e-06)     # this is the minimum size b. Upon reaching this value refinement is stopped.
# bmin = bstart                 # set this to DISABLE refinement (one full calculation with fixed size)
refinementfactor = int(10)      # In each refinement iteration a square is divided into refinementfactor^2 subsquares

epsilon = 0.01                      # threshold used to determine the decrease region of the cost function
dynlipconstant = numpy.float64(20)  # fixed Lipschitz constant of the dynamics on the analyzed region


def main_run():
    """Main function running the example. Determines the decrease region in a refinement process and the
    the region of attraction (ROA) afterwards.
    Note:
        If not otherwise specified, all mentioned numpy.ndarrays are of the type numpy.float64 for maximum precision.

    Returns:
        gridx (ndarray): Coordinates of interval centers on the first axis
        gridy (ndarray): Coordinates of interval centers on the second axis
        drcondfulfilled (ndarray of bool):
            Fulfillment of the decrease condition in each square interval indicated by TRUE or non fulfilled condition
            indicated by FALSE. Shape = (gridx.size, gridy.size).
        gpmean (ndarray):
            Values of the GP mean function at all square interval centers. Shape = (gridx.size, gridy.size).
        vmin (float):
            Minimum value of GP mean function outside of the decrease region.
            Filter gpmean by all values lower than vmin to obtain ROA."""

    intervalwidth = bstart          # set starting size of square analysis intervals
    refinementiterations = 0        # refinement iteration number
    while intervalwidth >= bmin:    # stop condition (if minimum interval size reached)
        print("\nRefinement iteration: ", refinementiterations + 1)
        print("Analytic grid x-range: [%d, %d], y-range: [%d, %d] and grid constant b: %e"
              % (xmin, xmax, ymin, ymax, intervalwidth))
        if DO_PLOT == 0:
            gridx = numpy.arange(xmin, xmax, intervalwidth * 2).astype(numpy.float64)  # division of both axes into
            gridy = numpy.arange(ymin, ymax, intervalwidth * 2).astype(numpy.float64)  # intervals with the size 2b
        elif DO_PLOT == 1:
            # ONLY FOR QUICK TEST USE (AND WITHOUT REFINEMENT):
            reduced_density = 0.01/intervalwidth  # only 100 points per axis unit BUT same b (theoretically unjustified!)
            gridx = numpy.arange(xmin, xmax, intervalwidth * reduced_density).astype(numpy.float64)  # division of both axes into
            gridy = numpy.arange(ymin, ymax, intervalwidth * reduced_density).astype(numpy.float64)  # intervals with the size 2b
        print("Resulting grid size:", (gridx.shape[0], gridy.shape[0]))

        if refinementiterations == 0:
            # initialize bool array indicating regions where decrease condition is fulfilled as FALSE
            # After the first iteration calculation at indexes where this array is TRUE is skipped
            drcondfulfilled = numpy.zeros((gridx.shape[0], gridy.shape[0]), dtype=bool)
        else:
            drcondfulfilled = numpy.repeat(drcondfulfilled, refinementfactor, axis=0)   # repeat values to match refined
            drcondfulfilled = numpy.repeat(drcondfulfilled, refinementfactor, axis=1)   # grid size
            print("Refinement done")

        # Solve system ode for x_{k+1} (required for decrease condition) CPU parallel
        print("Solving dynamical system ode at all grid points for use in decrease condition...")
        xnext = solveodepar(gridx, gridy)
        print("Done.\n")

        # calculate decrease region by GPGPU
        print("Start decrease region calculation on GPU:")
        result = decrease_region(gridx, gridy, intervalwidth, xnext, dynlipconstant, drcondfulfilled, xvec, lengthscale,
                                 sigmaf, lambdavec)
        # add regions where decrease condition is fulfilled (TRUE)
        drcondfulfilled += numpy.less(result, -epsilon)

        intervalwidth = intervalwidth / refinementfactor    # refine grid by reducing intervalwidth
        refinementiterations += 1                           # next iteration

    # Create mask for GP mean array to find the minimum GP mean value outside decrease region for ROA estimation
    drmask = numpy.copy(drcondfulfilled)
    # Ignore area around origin: This is necessary since the method determining the decrease region does not work
    # in very close proximity to the origin. This is not a problem since stability of the system around the origin can
    # often be simply shown e.g. with a quadratic Lyapunov function. Here, we add all indexes of the grid in a small
    # area around the origin to the decrease region mask to avoid the hole at the origin:
    xindexes = numpy.nonzero(numpy.less(numpy.abs(gridx), 0.15))[0]     # x-indexes of grid that are set to 1
    yindexes = numpy.nonzero(numpy.less(numpy.abs(gridy), 0.15))[0]     # y-indexes of grid that are set to 1
    drmask[numpy.ix_(xindexes, yindexes)] = 1                           # apply to mask

    # Calculate GP mean function values at all square interval centers to determine ROA
    print("\nStart region of attraction estimation on GPU:")
    gpmean, vmin = estimate_region_of_attraction(gridx, gridy, xvec, lengthscale, sigmaf, lambdavec, drmask)

    return gridx, gridy, drcondfulfilled, gpmean, vmin


if __name__ == "__main__":
    # # # Copy data from the example Gaussian Process (GP) trained on a finite horizon cost function
    print("Importing trained GP data.")
    GPdata = numpy.load("MA32bus_example.npz")
    xvec = GPdata['traindata'].astype(numpy.float64)    # vector of training data: Shape = (n, 2).
    n = xvec.shape[0]                                   # number of training data points
    print("Number of GP training data points:", n)
    lambdavec = GPdata['lambdavector'].astype(numpy.float64)
    # lambdavec: weight vector of GP mean prediction formula: lambda = (K + sigma_n*I)^{-1}. Shape = (n).
    sigmaf = GPdata['sigmaf']                           # sigmaf: signal standard deviation of kernel
    sigmaf = numpy.float64(sigmaf.item())
    lengthscale = GPdata['lengthscales'].astype(numpy.float64)  # separate lengthscales of ARD-kernel. Shape = (2)

    # Import analysis script for the selected kernel
    if KERNEL == "ardmatern32":
        from Matern32_GPU import decrease_region, estimate_region_of_attraction
    elif KERNEL == "ardmatern52":
        from Matern52_GPU import decrease_region, estimate_region_of_attraction
    elif KERNEL == "ardsquaredexponential":
        from SquaredExp_GPU import decrease_region, estimate_region_of_attraction
    else:
        raise ValueError("No supported kernel selected.")

    # # # Run calculation of decrease region and ROA
    gridx1, gridx2, ret, mean, minmean = main_run()

    # # # plot Decrease Region
    gridx2, gridx1 = numpy.meshgrid(gridx2, gridx1)     # 2D grid for plotting
    fig, ax = plt.subplots()
    # hull of decrease region
    CSf = ax.contourf(gridx1, gridx2, ret, levels=[0.5, 1], colors=['yellowgreen'])
    # fill area of decrease region with color
    CS = ax.contour(gridx1, gridx2, ret, linewidths=1, levels=[0.5], linestyles='solid', antialiased=True)

    # Format plot
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title('Decrease region of learned cost')
    fig.savefig('Decrease_Region.png', dpi=fig.dpi)

    # # # Plot region of attraction
    fig, ax = plt.subplots()
    # calculate 10 contour levels to be used in the contour plot
    minlvl = numpy.min(mean)
    lvls = numpy.arange(minlvl, minmean, minmean / 11)

    # fill area of region of attraction with color
    CSf = ax.contourf(gridx1, gridx2, mean, levels=[minlvl, minmean], colors=['yellowgreen'])
    # plot contours of levels of ROA
    CS = ax.contour(gridx1, gridx2, mean, levels=lvls, linewidths=1, linestyles='solid',
                    cmap=plt.cm.get_cmap('plasma_r', 100), antialiased=True)
    # plot hull of ROA
    CS2 = ax.contour(gridx1, gridx2, mean, linewidths=1, levels=[minmean], linestyles='solid', antialiased=True)
    # add colorbar
    norm = matplotlib.colors.Normalize(vmin=CS.cvalues.min(), vmax=CS.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CS.cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ticks=CS.levels)
    cbar.set_label('Level of learned cost function')

    # Format plot
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title('Estimated region of attraction of system')

    plt.show()
    fig.savefig('Region_of_Attraction.png', dpi=fig.dpi)




