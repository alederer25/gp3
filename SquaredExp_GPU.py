import numba
from numba import cuda
import math
import numpy
import time

MAX_VRAM_OCCUPATION = 0.9   # maximum percentage of available VRAM size that should be occupied at once. Must be <= 1.


def decrease_region(gridx, gridy, b, xnext, dynlipconst, drcondfulfilled, traindata, lengthscale, sigmaf, lambdavec):
    """Determine the decrease region of a trained GP on a given analytic grid on the GPU.
    Args:
        gridx (ndarray): Coordinates of interval centers on the first axis (1D array)
        gridy (ndarray): Coordinates of interval centers on the second axis (1D array)
        b: Grid constant; distance between two elements (interval centers) in gridx and gridy should be 2*b
        xnext (ndarray): Solutions of system ode at all grid points defined by gridx, gridy.
            Shape = (gridx.size, gridy.size, 2). xnext[:,:,0] contains states x_1; xnext[:,:,1] contains states x_2
        dynlipconst (float64): Fixed Lipschitz constant of the system dynamics on the analyzed region
        drcondfulfilled (ndarray of bool): Indicates indexes of the grid where decrease condition is already
            fulfilled (TRUE) when a refinement process is used. Otherwise supposed to be filled by zeros (FALSE).
            Shape = (gridx.size, gridy.size)
        traindata (ndarray): N training data points of cost function used for GP training. Shape = (N, 2).
        lengthscale (tuple): Kernel parameter: separate length scales of ARD kernel used for GP training.
            First element l_1, second element l_2. If non-ARD kernel is used both elements should be set equal.
        sigmaf (float64): Kernel parameter: signal standard deviation
        lambdavec (ndarray): Weights vector used to make predictions from the trained GP model.
            lambda = (K + sigma_n*I)^{-1}. Shape = (N).

    Returns:
        decreaseval (ndarray): Values of left side of the decrease condition inside all square
            intervals defined by gridx, gridy."""

    # Initialize result array
    decreaseval = numpy.empty((gridx.shape[0], gridy.shape[0]))

    # Required / available VRAM size calculations
    constantvram = traindata.nbytes + lengthscale.nbytes + lambdavec.nbytes + b.nbytes + dynlipconst.nbytes \
                   + sigmaf.nbytes
    availablevram = cuda.current_context().get_memory_info()[0] - constantvram
    requiredvram = gridx.nbytes + gridy.nbytes + xnext.nbytes + drcondfulfilled.nbytes + decreaseval.nbytes

    # If the size of all arrays together exceeds the available VRAM, division each axis by divisionfactor later.
    # This results in a division in divisionfactor^2 subblocks.
    divisionfactor = 1
    while requiredvram / divisionfactor**2 > 0.75 * availablevram:
        divisionfactor += 1     # determine how many times data needs to be divided to fit into VRAM

    @cuda.jit(device=True)
    def se_kernel(x1, x2, l):
        """CUDA device function: Returns covariance between points calculated by squared exponential kernel
        Args:
            x1 (tuple): Coordinates of a single point on the analyzed region
            x2 (tuple): Coordinates of a single point on the analyzed region
            l (tuple): Kernel parameter: separate length scales of ARD kernel used for GP training.

        Returns:
            cov (float64): Covariance between x1 and x2 calculated by the Matern 3/2 kernel"""
        dist = (x1[0]-x2[0]) ** 2 / l[0] ** 2 + (x1[1] - x2[1]) ** 2 / l[1] ** 2
        cov = sigmaf**2 * math.exp(-dist / 2)
        return cov

    @cuda.jit(device=True)
    def se_kernel_derive(delta, ls):
        """CUDA device function: Calculate derivative value of the squared exponential kernel in dimension m
        Args:
            delta (float64): Distance between two points the covariance should be calculated between
            ls (float64): Kernel parameter: Length scale of ARD kernel used for GP training in the dimension m
        Returns:
            (float64): Value of the derivative of the squared exponential kernel."""
        return -(delta / (ls ** 2)) * math.exp(-(delta ** 2) / (2 * ls ** 2))

    @cuda.jit(device=True)
    def fixed_se_kernel(xtrain, xcenter, kappa, ls):
        """CUDA device function: Returns bounds of the derivative of the squared exponential kernel in dimension != m
        Args:
            xtrain (float64): Coordinate of a training data point in dimension != m
            xcenter (float64): Coordinate of a center of a 1D-interval in dimension != m
            kappa (cuda.local.array): Array reference to write results back into
            ls (float64): Kernel parameter: Length scale of ARD kernel used for GP training in dimension != m.
        Returns:
            kappa (cuda.local.array): Bounds of the derivative of the squared exponential kernel in dimension != m.
                The first element is an upperbound and the second element is a lowerbound inside the interval
                with center xcenter."""
        norm = abs(xtrain - xcenter)
        if norm == 0:   # special case: training point lies on the interval center (avoid division by 0)
            kappa[0] = 1.0                                      # upper bound
            kappa[1] = math.exp(-(b ** 2) / (2 * ls ** 2))      # lower bound
        else:           # normal case: training point inside or outside interval
            tauover = min(b, norm) * (xtrain - xcenter) / norm
            tauunder = b * (xcenter - xtrain) / norm
            kappa[0] = math.exp(-(xtrain - xcenter - tauover) ** 2 / (2 * ls ** 2))     # upper bound
            kappa[1] = math.exp(-(xtrain - xcenter - tauunder) ** 2 / (2 * ls ** 2))    # lower bound
        return kappa

    @cuda.jit('void(float64[:], float64[:], float64[:,:], boolean[:,:],float64[:,:,:])', nopython=True, parallel=True)
    def calc_decrease_region(devgridx, devgridy, devresultgrid, decreaseregion, devxnext):
        """CUDA device function: Calculate local Lipschitz constants and evaluate decrease condition of the
        learned GP mean function over the analyzed region using these constants.
        Args:
            devgridx (DeviceNDArray): Device array of gridx of outer scope inside VRAM.
            devgridy (DeviceNDArray): Device array of gridy of outer scope inside VRAM.
            devresultgrid (DeviceNDArray): Array reference to write results into. Shape = (devgridx.size, devgridy.size)
            decreaseregion (DeviceNDArray): Device array of drcondfulfilled of outer scope inside VRAM.
            devxnext (DeviceNDArray): Device array of xnext of outer scope inside VRAM.
        Returns:
            devresultgrid (DeviceNDArray): Values of left side of the decrease condition inside all square
            intervals defined by devgridx, devgridy.
            Note:
                This CUDA device function does not have a direct return. Instead, it writes the results of the
                calculation to devresultgrid inside the VRAM. To return the array, it needs to be transferred back to
                the hosts RAM by calling copy_to_host() with its reference in the outer scope."""

        # Initialize computational CUDA grid
        i, j = cuda.grid(2)
        lambdavector = cuda.const.array_like(lambdavec)
        xtrain = cuda.const.array_like(traindata)
        l = cuda.const.array_like(lengthscale)

        if (i < devresultgrid.shape[0]) and (j < devresultgrid.shape[1]):
            # skip calculation if decrease condition is already fulfilled from earlier iterations
            if decreaseregion[i, j] == 0:
                # Allocate local device array for the coordinate tuple of a single point i,j on the grid
                x = cuda.local.array(shape=2, dtype=numba.float64)
                x[0] = devgridx[i]
                x[1] = devgridy[j]
                # Allocate local device array to store calculation of upper and lower bounds of GP mean function
                upperbound = cuda.local.array(shape=2, dtype=numba.float64)
                lowerbound = cuda.local.array(shape=2, dtype=numba.float64)
                upperbound[:] = 0
                lowerbound[:] = 0
                # Initialize value to calculate GP mean value at a single point i,j on the grid
                meanv = 0
                # Initialize value to calculate GP mean value at a single point i,j on devxnext
                meanvf = 0

                for k in range(0, xtrain.shape[0]):     # loop over all training data points
                    # Calculate GP mean at grid point
                    meanv += se_kernel(x, xtrain[k], l) * lambdavector[k]
                    # Calculate GP mean at devxnext point
                    meanvf += se_kernel(devxnext[i][j], xtrain[k], l) * lambdavector[k]

                    for m in range(0, xtrain.shape[1]):     # loop over dimensions of training data (=2)
                        # Allocate loc device array to store calculation of upper and lower bounds of
                        # kernel in dimension != m
                        fixeddimkernel = cuda.local.array(shape=2, dtype=numba.float64)
                        fixeddimkernel = fixed_se_kernel(xtrain[k][m - 1], x[m - 1], fixeddimkernel, l[m - 1])

                        # Calculate upper and lower bounds of the GP mean derivative according to the theory:
                        delta = xtrain[k][m] - x[m]
                        if abs(delta) > b + l[m]:
                            if lambdavector[k] > 0:
                                if delta > 0:
                                    upperbound[m] += -se_kernel_derive(abs(delta) - b, l[m]) * fixeddimkernel[0] * (
                                                sigmaf ** 2) * lambdavector[k]
                                    lowerbound[m] += -se_kernel_derive(abs(delta) + b, l[m]) * fixeddimkernel[1] * (
                                                sigmaf ** 2) * lambdavector[k]
                                else:
                                    upperbound[m] += se_kernel_derive(abs(delta) + b, l[m]) * fixeddimkernel[1] * (
                                                sigmaf ** 2) * lambdavector[k]
                                    lowerbound[m] += se_kernel_derive(abs(delta) - b, l[m]) * fixeddimkernel[0] * (
                                                sigmaf ** 2) * lambdavector[k]
                            else:
                                if delta > 0:
                                    lowerbound[m] += -se_kernel_derive(abs(delta) - b, l[m]) * fixeddimkernel[0] * (
                                                sigmaf ** 2) * lambdavector[k]
                                    upperbound[m] += -se_kernel_derive(abs(delta) + b, l[m]) * fixeddimkernel[1] * (
                                                sigmaf ** 2) * lambdavector[k]
                                else:
                                    lowerbound[m] += se_kernel_derive(abs(delta) + b, l[m]) * fixeddimkernel[1] * (
                                                sigmaf ** 2) * lambdavector[k]
                                    upperbound[m] += se_kernel_derive(abs(delta) - b, l[m]) * fixeddimkernel[0] * (
                                                sigmaf ** 2) * lambdavector[k]

                        elif abs(delta) < l[m] - b:
                            if lambdavector[k] > 0:
                                if delta > 0:
                                    upperbound[m] += -se_kernel_derive(abs(delta) + b, l[m]) * fixeddimkernel[0] * (
                                                sigmaf ** 2) * lambdavector[k]
                                    lowerbound[m] += -se_kernel_derive(abs(delta) - b, l[m]) * fixeddimkernel[1] * (
                                                sigmaf ** 2) * lambdavector[k]
                                else:
                                    upperbound[m] += se_kernel_derive(abs(delta) - b, l[m]) * fixeddimkernel[1] * (
                                                sigmaf ** 2) * lambdavector[k]
                                    lowerbound[m] += se_kernel_derive(abs(delta) + b, l[m]) * fixeddimkernel[0] * (
                                                sigmaf ** 2) * lambdavector[k]
                            else:
                                if delta > 0:
                                    lowerbound[m] += -se_kernel_derive(abs(delta) + b, l[m]) * fixeddimkernel[0] * (
                                                sigmaf ** 2) * lambdavector[k]
                                    upperbound[m] += -se_kernel_derive(abs(delta) - b, l[m]) * fixeddimkernel[1] * (
                                                sigmaf ** 2) * lambdavector[k]
                                else:
                                    lowerbound[m] += se_kernel_derive(abs(delta) - b, l[m]) * fixeddimkernel[1] * (
                                                sigmaf ** 2) * lambdavector[k]
                                    upperbound[m] += se_kernel_derive(abs(delta) + b, l[m]) * fixeddimkernel[0] * (
                                                sigmaf ** 2) * lambdavector[k]

                        elif l[m] - b <= abs(delta) <= l[m] + b:
                            if lambdavector[k] > 0:
                                if delta > 0:
                                    upperbound[m] += -se_kernel_derive(l[m], l[m]) * fixeddimkernel[0] * (sigmaf ** 2) * \
                                                     lambdavector[k]
                                    lowerbound[m] += min(-se_kernel_derive(abs(delta) + b, l[m]),
                                                         -se_kernel_derive(abs(delta) - b, l[m])) * fixeddimkernel[1] * \
                                                     lambdavector[k] * (sigmaf ** 2)
                                else:
                                    upperbound[m] += -min(-se_kernel_derive(abs(delta) + b, l[m]),
                                                          -se_kernel_derive(abs(delta) - b, l[m])) * fixeddimkernel[1] * \
                                                     lambdavector[k] * (sigmaf ** 2)
                                    lowerbound[m] += se_kernel_derive(l[m], l[m]) * fixeddimkernel[0] * (sigmaf ** 2) * \
                                                     lambdavector[k]
                            else:
                                if delta > 0:
                                    lowerbound[m] += -se_kernel_derive(l[m], l[m]) * fixeddimkernel[0] * (sigmaf ** 2) * \
                                                     lambdavector[k]
                                    upperbound[m] += min(-se_kernel_derive(abs(delta) + b, l[m]),
                                                         -se_kernel_derive(abs(delta) - b, l[m])) * fixeddimkernel[1] * \
                                                     lambdavector[k] * (sigmaf ** 2)
                                else:
                                    lowerbound[m] += -min(-se_kernel_derive(abs(delta) + b, l[m]),
                                                          -se_kernel_derive(abs(delta) - b, l[m])) * fixeddimkernel[1] * \
                                                     lambdavector[k] * (sigmaf ** 2)
                                    upperbound[m] += se_kernel_derive(l[m], l[m]) * fixeddimkernel[0] * lambdavector[k] * (
                                                sigmaf ** 2)
                # Calculate local Lipschitz constant from separate bounds in each dimension
                lipschitz = max(math.sqrt(upperbound[0] ** 2 + upperbound[1] ** 2),
                                           math.sqrt(lowerbound[0] ** 2 + lowerbound[1] ** 2))
                # Evaluation of left side of the decrease condition
                devresultgrid[i, j] = meanvf - meanv + lipschitz * (dynlipconst + 1) * math.sqrt(2.0) * b
        cuda.syncthreads()

    # Initialize CUDA stream
    stream = cuda.stream()
    nrboxesx = numpy.size(gridx)    # Nr. of intervals on x-axis
    nrboxesy = numpy.size(gridy)    # Nr. of intervals on y-axis
    xlast = 0
    # Division if data needs to be subdivided due to VRAM size limitation:
    for indx in range(1, divisionfactor+1):         # on the first axis
        xstart = xlast                              # Lowest index of subblock in x
        xlast = int(indx/divisionfactor*nrboxesx)   # Highest index of subblock in x
        ylast = 0
        for indy in range(1, divisionfactor+1):             # on the second axis
            ystart = ylast                                  # Lowest index of subblock in y
            ylast = int(indy / divisionfactor * nrboxesy)   # Highest index of subblock in y
            print("\nInitializing GPU computation of x-indexes %d to %d, y-indexes %d to %d"
                  % (xstart, xlast, ystart, ylast))

            # Initialize GPU thread, block and computational grid size
            tpb = (16, 16)
            blockspergrid_x = (xlast-xstart + tpb[0]) // tpb[0]
            blockspergrid_y = (ylast-ystart + tpb[1]) // tpb[1]
            bpg = (blockspergrid_x, blockspergrid_y)
            print("CUDA computational grid initialized\n")

            # Copy data from RAM to GPU memory (host to device)
            start = time.perf_counter()
            dgx = cuda.to_device(numpy.ascontiguousarray(gridx[xstart:xlast]), stream=stream)
            dgy = cuda.to_device(numpy.ascontiguousarray(gridy[ystart:ylast]), stream=stream)
            dynfp1 = cuda.to_device(numpy.ascontiguousarray(xnext[xstart:xlast, ystart:ylast]), stream=stream)
            dresult = cuda.device_array_like(numpy.ascontiguousarray(decreaseval.astype(numpy.float64)[xstart:xlast, ystart:ylast]), stream=stream)
            ddecreasecond = cuda.to_device(numpy.ascontiguousarray(drcondfulfilled[xstart:xlast, ystart:ylast]), stream=stream)
            h2dtime = time.perf_counter() - start

            # execute GPU calculation
            start = time.perf_counter()
            calc_decrease_region[bpg, tpb, stream](dgx, dgy, dresult, ddecreasecond, dynfp1)
            exectime = time.perf_counter() - start

            # copy results from GPU memory to RAM (device to host)
            start = time.perf_counter()
            decreaseval[xstart:xlast, ystart:ylast] = dresult.copy_to_host(stream=stream)
            stream.synchronize()
            d2htime = time.perf_counter() - start

            print("GPU computation done:")
            print("H2D copy time:", h2dtime)
            print("Kernel invoke time:", exectime)
            print("Calcualtion % D2H copy time:", d2htime)
            print("Total execution time:", d2htime + exectime + h2dtime)

            # Deallocation of device arrays
            dgx = dgy = dynfp1 = dresult = ddecreasecond = None

    cuda.close()
    return decreaseval


def estimate_region_of_attraction(gridx, gridy, traindata, lengthscale, sigmaf, lambdavec, decreaseregion):
    """Predict mean values from the GP trained on a cost function over the given analytic grid on the GPU and find
    its minimum value outside of the decrease function to estimate the region of attraction.
    Args:
        gridx (ndarray): Coordinates of interval centers on the first axis (1D array)
        gridy (ndarray): Coordinates of interval centers on the second axis (1D array)
        traindata (ndarray): N training data points of cost function used for GP training. Shape = (N, 2).
        lengthscale (tuple): Kernel parameter: separate length scales of ARD kernel used for GP training.
            First element l_1, second element l_2. If non-ARD kernel is used both elements should be set equal.
        sigmaf (float64): Kernel parameter: signal standard deviation
        lambdavec (ndarray): Weights vector used to make predictions from the trained GP model.
            lambda = (K + sigma_n*I)^{-1}. Shape = (N).
        decreaseregion (ndarray of bool): Indicates indexes of the grid where decrease condition is fulfilled (TRUE).
            Shape = (gridx.size, gridy.size).

    Returns:
        meangp (ndarray): Predictions of the GP mean at all grid points. Shape = (gridx.size, gridy.size)
        vmin (float64): Minimum value of meangp outside of the decrease region.
        Note:
            By filtering meangp by all values <= vmin, the estimated region of attraction can be plotted"""

    # Initialize result array
    meangp = numpy.empty((gridx.shape[0], gridy.shape[0]))

    # Required / available VRAM size calculations
    constantvram = traindata.nbytes + lengthscale.nbytes + lambdavec.nbytes + sigmaf.nbytes
    availablevram = cuda.current_context().get_memory_info()[0] - constantvram
    requiredvram = gridx.nbytes + gridy.nbytes + meangp.nbytes

    # If the size of all arrays together exceeds the available VRAM, division each axis by divisionfactor later.
    # This results in a division in divisionfactor^2 subblocks.
    divisionfactor = 1
    while requiredvram / divisionfactor ** 2 > MAX_VRAM_OCCUPATION * availablevram:
        divisionfactor += 1  # determine how many times data needs to be divided to fit into VRAM

    @cuda.jit(device=True)
    def se_kernel(x1, x2, l):
        """CUDA device function: Returns covariance between points calculated by the squared exponential kernel
        Args:
            x1 (tuple): Coordinates of a single point on the analyzed region
            x2 (tuple): Coordinates of a single point on the analyzed region
            l (tuple): Kernel parameter: separate length scales of ARD kernel used for GP training.

        Returns:
            cov (float64): Covariance between x1 and x2 calculated by the Matern 3/2 kernel"""
        dist = (x1[0]-x2[0]) ** 2 / l[0] ** 2 + (x1[1] - x2[1]) ** 2 / l[1] ** 2
        cov = sigmaf**2 * math.exp(-dist / 2)
        return cov

    @cuda.jit('void(float64[:], float64[:], float64[:,:])', nopython=True, parallel=True)
    def calc_region_of_attraction(devgridx, devgridy, devgpmean):
        """Predict GP mean values at all grid points on the GPU.
        Args:
            devgridx (DeviceNDArray): Device array of gridx of outer scope inside VRAM.
            devgridy (DeviceNDArray): Device array of gridy of outer scope inside VRAM.
            devgpmean (DeviceNDArray): Array reference to write results into. Shape = (devgridx.size, devgridy.size)

        Returns:
            devgpmean (DeviceNDArray): GP mean predictions at all grid points defined by devgridx, devgridy.
            Note:
                This CUDA device function does not have a direct return. Instead, it writes the results of the
                calculation to devgpmean inside the VRAM. To return the array, it needs to be transferred back to
                the hosts RAM by calling copy_to_host() with its reference in the outer scope."""
        # Initialize computational CUDA grid
        i, j = cuda.grid(2)
        # Allocate constant arrays on the GPU
        lambdavector = cuda.const.array_like(lambdavec)
        xtrain = cuda.const.array_like(traindata)
        l = cuda.const.array_like(lengthscale)

        if (i < devgpmean.shape[0]) and (j < devgpmean.shape[1]):
            # Allocate local device array for the coordinate tuple of a single point i,j on the grid
            x = cuda.local.array(shape=2, dtype=numba.float64)
            x[0] = devgridx[i]
            x[1] = devgridy[j]
            # Initialize variable to store intermediate results of the vector product
            meanvalue = 0
            for k in range(0, xtrain.shape[0]):                             # Loop over all training data points
                meanvalue += se_kernel(x, xtrain[k], l) * lambdavector[k]   # vector product

            devgpmean[i, j] = meanvalue     # write to result array
        cuda.syncthreads()

    # Initialize CUDA stream
    stream = cuda.stream()
    nrboxesx = numpy.size(gridx)    # Nr. of intervals on x-axis
    nrboxesy = numpy.size(gridy)    # Nr. of intervals on y-axis
    xlast = 0

    # Division if data needs to be subdivided due to VRAM size limitation:
    for indx in range(1, divisionfactor+1):         # on the first axis
        xstart = xlast                              # Lowest index of subblock in x
        xlast = int(indx/divisionfactor*nrboxesx)   # Highest index of subblock in x
        ylast = 0
        for indy in range(1, divisionfactor+1):             # on the second axis
            ystart = ylast                                  # Lowest index of subblock in y
            ylast = int(indy / divisionfactor * nrboxesy)   # Highest index of subblock in y
            print("\nInitializing GPU computation of x-indexes %d to %d, y-indexes %d to %d"
                  % (xstart, xlast, ystart, ylast))

            # Initialize GPU thread, block and computational grid size
            tpb = (16, 16)
            blockspergrid_x = (xlast-xstart + tpb[0]) // tpb[0]
            blockspergrid_y = (ylast-ystart + tpb[1]) // tpb[1]
            bpg = (blockspergrid_x, blockspergrid_y)
            print("CUDA computational grid initialized\n")

            # Copy data from RAM to GPU memory (H2D)
            start = time.perf_counter()
            dgx = cuda.to_device(numpy.ascontiguousarray(gridx[xstart:xlast]), stream=stream)
            dgy = cuda.to_device(numpy.ascontiguousarray(gridy[ystart:ylast]), stream=stream)
            dgpmean = cuda.device_array_like(numpy.ascontiguousarray(meangp.astype(numpy.float64)[xstart:xlast, ystart:ylast]), stream=stream)
            h2dtime = time.perf_counter() - start

            # execute GPU calculation
            start = time.perf_counter()
            calc_region_of_attraction[bpg, tpb, stream](dgx, dgy, dgpmean)
            exectime = time.perf_counter() - start

            # copy results from GPU memory to RAM (D2H)
            start = time.perf_counter()
            meangp[xstart:xlast, ystart:ylast] = dgpmean.copy_to_host(stream=stream)
            stream.synchronize()
            d2htime = time.perf_counter() - start

            print("GPU computation done:")
            print("H2D copy time:", h2dtime)
            print("Kernel invoke time:", exectime)
            print("Calcualtion % D2H copy time:", d2htime)
            print("Total execution time:", d2htime + exectime + h2dtime)

            # Deallocation of device arrays
            dgx = dgy = dresult = None
    cuda.close()

    # Find minimum of the GP mean masked by the decrease region (outside of the decrease region = FALSE)
    # All areas where the GP mean function <= this value are part of the ROA
    vmin = numpy.min(meangp[decreaseregion != 1])
    return meangp, vmin



