import sys
import numpy
import numpy as np
import pylab as pl

from   scipy.special     import eval_legendre
from   scipy.interpolate import lagrange
from   scipy.linalg      import pinvh, svd
from   scipy.special     import legendre


np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize)

def model(mu, N, stride=2):
    # Generator for \sum_i L_i, i=0 to N with stride.
    result = 0.0

    i      = 0

    print('Generating power spectrum model.\n'.format(i))
    
    while i <= N:
      print('Including L{}.'.format(i))  

      result += eval_legendre(i, mu)
      i      += stride

    print('\nFinialised power spectrum model.\n'.format(i))
      
    return result

print('\n\nRegression multipole calc.')

# Highest expected multipole.
deg        = 4

poles      = np.arange( 1 + deg )
npole      = len(poles)

# 1 include odd, otherwise 2.
stride     = 1

# mu = n0 / (n2^2 +n1^2 + n0^2); dmu = (2 * pi) n / L / k at fixed k.
L          = 3000.
NN         = 64

# grid     = np.linspace(-NN/2, NN/2 - 1, NN, dtype=np.float)

grid                = np.linspace(0, NN - 1, NN, dtype=np.float)
grid[grid >= NN/2] -= NN

kx         = (2. * np.pi / L) * grid
ky         = (2. * np.pi / L) * grid
kz         = (2. * np.pi / L) * grid

kx, ky, kz = np.meshgrid(kx, ky, kz)
k          = np.sqrt(kx ** 2. + ky ** 2. + kz ** 2.) 
ms         = kz / k 
ms[k ==0.] = 0.0

# cut      = (k > 0.05) & (k < 0.06)

# kedges     = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
kedges       = np.arange(0.0, 0.2, 0.05)

k2edges      = kedges **2.

dmu          = 1.0 / 120.0
muedges      = dmu * np.arange(0, 121, 1)

Nk           = len(kedges)  - 1
Nmu          = len(muedges) - 1

lsum         = numpy.zeros((npole, npole, Nk+2, Nmu+2), dtype=np.cdouble)
Nsum         = numpy.zeros((Nk+2, Nmu+2), dtype='i8')

for i in range(NN):
    # print('{:6e} \t {:.6e}'.format(np.min(ms[i,...]), np.max(ms[i,...])))

    mu          = ms[i,...]
    ks          =  k[i,...]
    
    dig_mu      = numpy.digitize(abs(mu).flat, muedges)
    dig_k       = np.digitize(ks.flatten(), k2edges)
    
    multi_index = numpy.ravel_multi_index([dig_k, dig_mu], (Nk+2,Nmu+2)) 

    print('Solving {} of {}'.format(i, NN))
    
    for jj, jell in enumerate(poles):
        for kk, kell in enumerate(poles):            
                    prod                       = legendre(jell)(mu) * legendre(kell)(mu)
                    '''
                    print('{:.6e}\t{:.6e}\t{:d}\t{:.6e}\t{:.6e}\t{:d}\t{:.6e}\t{:.6e}\t'.format(numpy.min(mu),\
                                                                                                numpy.max(mu),\
                                                                                                jell,\
                                                                                                numpy.min(legendre(jell)(mu)),\
                                                                                                numpy.max(legendre(jell)(mu)),\
                                                                                                kell,\
                                                                                                numpy.min(legendre(kell)(mu)),\
                                                                                                numpy.max(legendre(kell)(mu))))
                    '''
                    lsum[jj,kk,...].real.flat += numpy.bincount(multi_index, weights=prod.real.flat, minlength=Nsum.size)
                    lsum[jj,kk,...].imag.flat += numpy.bincount(multi_index, weights=prod.imag.flat, minlength=Nsum.size)

                    # print(lsum[jj,kk,...].real)

# reshape and slice to remove out of bounds points.                                                                                                                                                                                     
sl             = slice(1, -1)
lsum[..., -2] += lsum[..., -1]

for ii in range(Nk):
    print('\n\n') 
    print(lsum[..., ii, sl].sum(axis=-1).real)

# dmu        = 2. * np.pi / L / k
# ms         = np.arange(-1., 1., dmu)

print('Solving for L:  {} and lmax: {}, stride: {}.\n\n'.format(L, deg, stride))

ps           = model(ms, deg, stride=stride)
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
print('\nContinuum limit of inverse.')

# Factor of 2 in units.
print(np.diag((2. * poles + 1)))

print('\nRealized R.')

print(R / R[0,0])

# Normalize to first entry of 2.  Continuum limit is 2 . (2 ell + 1).  
print('\nRealized inverse.')

print(iR / iR[0,0])

print('\nRealized estimate.')

result = np.dot(iR, S)

for x in result:
    print('{:.12f}'.format(x))

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

# print('\nOriginal mu sampling:  {:.6f} ({}), new sampling:  {:.6f} ({}), sufficient sampling for Laplace {:.6f} ({}).'.format(dmu, norig_sample, ndmu, len(ms), 2. / nsample_Laplace, nsample_Laplace))

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
