#================================================================================
# GadgetSpectrum
#================================================================================
#================================================================================
#
# Power spectrum computation tool, which uses GADGET files to generate the 1D 
# matter power spectrum. 
#
# - Uses numpy and FFTW. Therefore vectorised. And fast. 
# - Currently reads snapshots in ONLY single files. 
# - NOT COMPLETE.
#
# - The internal units of a GADGET snapshot are by default Kpc/h. 
# - This converts all relevant length variables to Mpc/h for the computation.
# - Thus there are dividing factors of 1000.0 for the boxsize and position arrays.  
#================================================================================
#
# Himanish Ganjoo, 14 Mar 2018
#
#================================================================================


from numpy import *
import pyfftw
import itertools
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_dd as hst
import numpy as np
import cpy

# Histogram Function in N-Dimensions
#================================================================================
# Taken from https://gist.github.com/letmaik/e625cb7777376899adca
#================================================================================

from numpy import atleast_2d, asarray, zeros, ones, array, atleast_1d, arange,\
    isscalar, linspace, diff, empty, around, where, bincount, sort, log10,\
    searchsorted

def fasthistogram(sample, bins=10, range=None, normed=False, weights=None):
    """
    Compute the multidimensional histogram of some data.

    Parameters
    ----------
    sample : array_like
        The data to be histogrammed. It must be an (N,D) array or data
        that can be converted to such. The rows of the resulting array
        are the coordinates of points in a D dimensional polytope.
    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitly in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    normed : bool, optional
        If False, returns the number of samples in each bin. If True,
        returns the bin density ``bin_count / sample_count / bin_volume``.
    weights : array_like (N,), optional
        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
        Weights are normalized to 1 if normed is True. If normed is False,
        the values of the returned histogram are equal to the sum of the
        weights belonging to the samples falling into each bin.
        Weights can also be a list of (weight arrays or None), in which case
        a list of histograms is returned as H.

    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. See normed and weights
        for the different possible semantics.
    edges : list
        A list of D arrays describing the bin edges for each dimension.

    See Also
    --------
    histogram: 1-D histogram
    histogram2d: 2-D histogram

    Examples
    --------
    >>> r = np.random.randn(100,3)
    >>> H, edges = np.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)

    """

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = atleast_2d(sample).T
        N, D = sample.shape
    
    if weights is None:
        W = None
    else:    
        try:
            # Weights is a 1D-array
            weights.shape
            W = -1
        except (AttributeError, ValueError):
            # Weights is a list of 1D-arrays or None's
            W = len(weights)

    if W == -1 and weights.ndim != 1:
        raise AttributeError('Weights must be a 1D-array, None, or a list of both')

    nbin = empty(D, int)
    edges = D*[None]
    dedges = D*[None]
    if weights is not None:
        if W == -1:
            weights = asarray(weights)
            assert weights.shape == (N,)
        else:
            for i in arange(W):
                if weights[i] is not None:
                    weights[i] = asarray(weights[i])
                    assert weights[i].shape == (N,)

    try:
        M = len(bins)
        if M != D:
            raise AttributeError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D*[bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        # Handle empty input. Range can't be determined in that case, use 0-1.
        if N == 0:
            smin = zeros(D)
            smax = ones(D)
        else:
            smin = atleast_1d(array(sample.min(0), float))
            smax = atleast_1d(array(sample.max(0), float))
    else:
        smin = zeros(D)
        smax = zeros(D)
        for i in arange(D):
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in arange(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Create edge arrays
    for i in arange(D):
        if isscalar(bins[i]):
            if bins[i] < 1:
                raise ValueError(
                    "Element at index %s in `bins` should be a positive "
                    "integer." % i)
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = linspace(smin[i], smax[i], nbin[i]-1)
        else:
            edges[i] = asarray(bins[i], float)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = diff(edges[i])
        if np.any(np.asarray(dedges[i]) <= 0):
            raise ValueError(
                "Found bin edge of size <= 0. Did you specify `bins` with"
                "non-monotonic sequence?")

    nbin = asarray(nbin)

    # Handle empty input.
    if N == 0:
        if W > 0:
            return [np.zeros(nbin-2) for _ in arange(W)], edges
        else:
            return np.zeros(nbin-2), edges

    # Compute the bin number each sample falls into.
    Ncount = {}
    for i in arange(D):
        # searchsorted is faster for many bins
        Ncount[i] = searchsorted(edges[i], sample[:, i], "right")
        #Ncount[i] = digitize(sample[:, i], edges[i])

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    for i in arange(D):
        # Rounding precision
        mindiff = dedges[i].min()
        if not np.isinf(mindiff):
            decimal = int(-log10(mindiff)) + 6
            # Find which points are on the rightmost edge.
            not_smaller_than_edge = (sample[:, i] >= edges[i][-1])
            on_edge = (around(sample[:, i], decimal) == around(edges[i][-1], decimal))
            # Shift these points one bin to the left.
            Ncount[i][where(on_edge & not_smaller_than_edge)[0]] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    ni = nbin.argsort()
    xy = zeros(N, int)
    for i in arange(0, D-1):
        xy += Ncount[ni[i]] * nbin[ni[i+1:]].prod()
    xy += Ncount[ni[-1]]

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    if len(xy) == 0:
        if W > 0:
            return [np.zeros(nbin-2) for _ in arange(W)], edges
        else:
            return zeros(nbin-2, int), edges

    # Flattened histogram matrix (1D)
    # Reshape is used so that overlarge arrays
    # will raise an error.
    Wd = W if W > 0 else 1
    hists = [zeros(nbin, float).reshape(-1) for _ in arange(Wd)]
    for histidx, hist in enumerate(hists):
        weights_ = weights[histidx] if W > 0 else weights
        flatcount = bincount(xy, weights_)
        a = arange(len(flatcount))
        hist[a] = flatcount
    
        # Shape into a proper matrix
        hist = hist.reshape(sort(nbin))
        ni = nbin.argsort()
        for i in arange(nbin.size):
            j = ni.argsort()[i]
            hist = hist.swapaxes(i, j)
            ni[i], ni[j] = ni[j], ni[i]
    
        # Remove outliers (indices 0 and -1 for each dimension).
        core = D*[slice(1, -1)]
        hist = hist[core]
    
        # Normalize if normed is True
        if normed:
            s = hist.sum()
            for i in arange(D):
                shape = ones(D, int)
                shape[i] = nbin[i] - 2
                hist = hist / dedges[i].reshape(shape)
            hist /= s
    
        if (hist.shape != nbin - 2).any():
            raise RuntimeError(
                "Internal Shape Error: hist.shape != nbin-2 -> " + str(hist.shape) + " != " + str(nbin-2))
        
        hists[histidx] = hist
    
    if W in [None, -1]:
        return hists[0], edges
    else:
        return hists, edges



#================================================================================
#================================================================================



# Snapshot File
#================================================================================

#filename = '/home/storage1/sumint/n-00-256/csnap_n-00-256_014'
#filename = './snap'
filename = '/home/sumint/N-GenIC/ICs/ics'

#================================================================================



# Reading Snapshot (only positions are needed for power spectrum)
#================================================================================

f = open(filename,'r')

# Reading header
#================================================================================

header_length = fromstring(f.read(4),int32)
num_parts = fromstring(f.read(6*4),int32)[1]
mass_parts = fromstring(f.read(6*8),float64)

a = fromstring(f.read(8),float64)
z = fromstring(f.read(8),float64)

flag_sfr = fromstring(f.read(4),int32)
flag_feedback = fromstring(f.read(4),int32)
dummy = fromstring(f.read(6*4),int32)
flag_cooling = fromstring(f.read(4),int32)
num_files = fromstring(f.read(4),int32)

boxsize = fromstring(f.read(8),float64) / 1000.0
omega_m = fromstring(f.read(8),float64)
omega_l = fromstring(f.read(8),float64)
h = fromstring(f.read(8),float64)

f.close()

# Header read. 
#================================================================================


# Open again, jump to after header (header is 256 bytes)
#================================================================================

f = open(filename,'r')

dummy = fromstring(f.read(4),int32)

jump = fromstring(f.read(256))
del jump

dummy = fromstring(f.read(4),int32)
dummy = fromstring(f.read(4),int32)

# Read positions array.
#================================================================================

pos = fromstring(f.read(4*3*num_parts),float32)
pos = reshape(pos,(num_parts,3)) / 1000.0



f.close()

print "File read. Positions loaded."

# File done with. 
#================================================================================



# Assign grid for smoothed potential
# Extract CIC values for all positions
# Do CIC smoothing
#================================================================================


ngrid = int(boxsize)

npart = len(pos)

# CIC Breaking
# ints contains the integer values of the positions. 
# fracs contains the values [(1-f)_x,(1-f)_y,(1-f)_z,f_x,f_y,f_z ]
# where f is the fractional part of the position. 
#================================================================================

ints = floor(pos).astype(int)
frac = modf(pos)[0]
remfrac = 1 - frac
fracs = hstack((remfrac,frac))


print 'CIC values computed for all positions.'

#================================================================================

# Grid
#================================================================================

r = zeros((ngrid,ngrid,ngrid))
r = r.astype('float64')

print 'Grid initialised.'


# Potential field assignment
#================================================================================
# generates a list of all 3-length combinations of (0,1).
# This is to be added to the whole-number indices of the grid point for CIC. 
# If we are processing the grid point (a,b,c), then all points (a+n,b+n,c+n) are to be covered,
# where n belongs to (0,1)
# If n = 1, the fractional part is added to the point. If n = 0, the remainder of the fractional part is added. 
# (based on the fracs array).
#================================================================================

# Generate all possible shifts in 3D.
shiftlist = list(itertools.product([0,1],repeat=3))

# Generate array indices for picking the fractional values based on the shift.
# For instance, for shift [0,0,0], the indices [0,1,2] are chosen from the fracs array, 
# which contain [ (1-f)_x,(1-f)_y,(1-f)_z ].
# And so, based on the 3 components of the shift, we pick the (1-f) or (f) from the fracs array. 
# The general formula is: [3*s_x , 3*s_y + 1 , 3*s_z + 2] where s_i is the shift in the i-component. 
ciclist = 3*array(shiftlist) + array([0,1,2])

bins_field = arange(-0.5,ngrid+0.5,1)

for i,shift in enumerate(shiftlist):

	 indices = (ints + array(shift)) % ngrid # Indices with shift.

	 weights = prod( fracs[:,ciclist[i]] , axis = 1) # Product of (1-f) or (f) based on the shift. 
	 
         indices = indices.astype('int64')
         weights = weights.astype('float64')
	 
	 # Add the product of the (1-f) or (f) to the requisite box in the grid.
	 # The bins pick up integers. 
	 #r = r + hst(indices,weights,statistic = 'sum',bins = (bins_field,bins_field,bins_field))[0] 
	 
         r = cpy.load_grid(indices,weights,r,num_parts)


print 'Assigning done'
'''	 

ciclist = list(itertools.product([0,1],repeat=3)) 

for i in range(0,npart):


	for shift in ciclist:
	
		# add shift to the whole number positions.
		indices = ints[i] + asarray(shift)		
		
		# periodic wrapping: if the index reaches the edge in any dimension, go to zero. 
		for n,ind in enumerate(indices):
		
			if ind == ngrid:
				indices[n] = 0
				
		indices = tuple(indices)
		
		weight = 1.0
		
		# add the required weight to the grid point. 
		# According to the value of the shift, f or (1-f) get selected for the weight. 
		
		for d in range(0,3):
			weight = weight*fracs[shift[d]][i][d]

		r[indices] = r[indices] + weight
	
'''

   
print 'Rho values assigned after CIC smoothing.'   

# Grid assignment done.
#================================================================================

# Fourier Transform
#================================================================================

plan = pyfftw.empty_aligned((ngrid,ngrid,ngrid), dtype='complex128')

plan = r

delk = pyfftw.interfaces.numpy_fft.fftn(plan)

print 'FFT done.'

#================================================================================

#================================================================================
del pos
del r
del plan
#================================================================================

# Computing del^2(k) for each mode. 
#================================================================================

power = absolute(delk)*absolute(delk)
power = power[1:,1:,1:]

#================================================================================

# Generate frequency basis arrays in all 3 dimensions.
#================================================================================

# Remove the first element as that is (0,0,0) and simply yields the mean of the field.
x = 2*pi*fft.fftfreq(ngrid,d=1.0)[1:]
y = 2*pi*fft.fftfreq(ngrid,d=1.0)[1:]
z = 2*pi*fft.fftfreq(ngrid,d=1.0)[1:]

#================================================================================

# Construct a 3D basis from the three basis arrays.
#================================================================================

kx,ky,kz = meshgrid(x,y,z)

print 'Basis constructed in Fourier space.'

#================================================================================


# Power Spectrum computation
#================================================================================
# The idea is:
#
# 1. We compute the |k| of each point in the Fourier space.
# 2. We generate bins for |k| equally spaced in log space.
# 3. We create a histogram using the values of |k| using the bins we made. 
# 4. BUT, instead of adding a unit count to the histogram, we add the corresponding 
#    value of the power at that point. 
# 5. This gives us the power in logarithmic bins of |k|.
# 6. Now, the power spectrum is the mean power per logarithmic bin. 
#    So, we divide the total power per bin by the number of contributions in that bin. 
#
# We have generated the 1D power spectrum from the 3D power. 
#================================================================================

k_mod = sqrt(kx*kx + ky*ky + kz*kz)

k = logspace(-4,1.2,50)



numpk = histogram(log10(k_mod),bins=log10(k),weights=power)[0] # Power contribution: Numerator. 
denpk = histogram(log10(k_mod),bins=log10(k))[0] # Count contribution: Denominator. 

print 'Fourier space binning done.'

# Either use centre points or remove the first point, for plotting histogram.
k = k[1:]



#================================================================================
#================================================================================

plt.figure()

plt.loglog(k,numpk/denpk,'.')
#plt.xlabel('k in h^{-1} Mpc')
#plt.ylabel('P(k) in (h/Mpc)^{3}')
#plt.title('Power Spectrum')
#plt.savefig('psnew.png')
#plt.close()

#print 'plot saved'





















