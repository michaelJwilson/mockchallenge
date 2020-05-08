import sys
import numpy as np
import pylab as pl

from   scipy.special     import eval_legendre
from   scipy.interpolate import lagrange
from   scipy.linalg      import pinvh, svd


np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize)

def model(mu, N, stride=2):
    # Generator for \sum_i L_i, i=0 to N with stride.
    result = 0.0

    i = 0
    
    while i <= N:
      result += eval_legendre(i, mu)
      i      += stride
  
    return result

print('\n\nRegression multipole calc.\n')

# Highest expected multipole.
deg    = 3

poles  = np.arange(deg)
npole  = len(poles)

# 1 include odd, otherwise 2.
stride = 1

# mu = n0 / (n2^2 +n1^2 + n0^2); dmu = (2 * pi) n / L / k at fixed k.
L   = 30000.
k   = 0.05

dmu = 2. * np.pi / L / k

print('Solving for L:  {}, k:  {} and dmu:  {}, deg.: {}.\n\n'.format(L, k, dmu, deg))

ms  = np.arange(-1., 1., dmu)
ps  = model(ms, deg, stride=stride)

norig_sample = len(ms)

# pl.plot(ms, ps, marker='^', lw=0.0) 

# Add noise.
# ps  = np.random.exponential(ps)

# pl.plot(ms, ps, marker='^', lw=0.0)

# Regression matrix.
R     = np.zeros((npole, npole))
S     = np.zeros(npole).T

for i in range(npole):
    S[i] = np.sum(eval_legendre(poles[i], ms) * ps)
    
    for j in range(npole):
        R[i,j] = np.sum(eval_legendre(poles[i], ms) * eval_legendre(poles[j], ms))

iR, erank = pinvh(R, return_rank=True)

# Continuum limit. 
print('Continuum limit of inverse.')

# Factor of 2 in units.
print(np.diag((2. * poles + 1)))

print('\nRealized R.')

print(R / R[0,0])

# Normalize to first entry of 2.  Continuum limit is 2 . (2 ell + 1).  
print('\nRealized inverse.')

print(iR / iR[0,0])

print('\nRealized estimate.')

print(np.dot(iR, S))

print('\nPrint effective rank {}'.format(erank))

# SVD
U, s, Vh = svd(R)

print('\n\nSVD decomposition of regression matrix.')

print(s)

condition = (np.max(s) / np.min(s))

print('\nReciprocal of condition number: {}.\n'.format(1. / condition))

# ------------------------------------------------------------------ #

# Laplace interpolation requires * at least * (deg + 1) samples.
# i) Sets min. number of independent mu points required for given k.
# ii) 

#
nsample_Laplace  = (deg+1)

assert	len(ms) > nsample_Laplace

print('Downsampling to mu mesh required for exact Laplace interpolation for max poly. of deg. {}'.format(deg))

sampling = 1

while len(ms[::sampling]) > nsample_Laplace:
  sampling *= 2

# Catch overshooting
if len(ms[::sampling]) < nsample_Laplace:
    sampling /= 2

sampling = np.int(sampling)

ms = ms[::sampling]

# Here we should be averaging the power across down samples. 
# Potentially weighting by Ll mu -> mu'.
ps = ps[::sampling]

ndmu = np.diff(ms)[0]

print('\nOriginal mu sampling:  {:.6f} ({}), new sampling:  {:.6f} ({}), sufficient sampling for Laplace {:.6f} ({}).'.format(dmu, norig_sample, ndmu, len(ms), 2. / nsample_Laplace, nsample_Laplace))

ipoly = lagrange(ms, ps)

# Unique polynomial of degree N.                                                                                                                                                                             
print('\nDegree of interpolating polynomial: {}\n'.format(ipoly.order))

# pl.plot(ms, ipoly(ms))
# pl.ylim(0., 5.)
# pl.show()

# N-point rule is exact for degree D = 2N - 1.
# Note here we double required deg. arg. as P(k,mu) * L_ell(mu) in integrand.

# Generated samplies in mu for solving orthogonality integral.                                                                                                                                                                                
samples, weights = np.polynomial.legendre.leggauss(2 * deg)
nsample_GaussLeg = len(samples)

print('Number of interpolated samples required in Gauss-Legendre integration: {}.\n'.format(len(samples)))

# Evaluate with new polynomial.
fs               = ipoly(samples)

print('ell:\tModel.\t\tMeasured.')

for ell in np.arange(0, deg, stride):
    Ls           = eval_legendre(ell, samples)

    integrand    = (2. * ell + 1) * fs * Ls / 2.
    multipole    = np.sum(weights * integrand)

    print('{:03d}:\t{:.12f}\t{:.12f}'.format(ell, 1.0, multipole))

print('\n\nDone.\n\n')
