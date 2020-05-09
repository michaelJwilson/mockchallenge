import  os
import  numpy                 as      np

# On worker.                                                                                                                                                                                                                               
if os.getenv('SLURM_NODEID') is not None:
  from    nbodykit.lab        import  cosmology, FFTPower
  from    nbodykit            import  setup_logging, style
  from    nbodykit.source     import  mesh
  from    mpi4py              import  MPI
  from    nbodykit.base.mesh  import  MeshSource
  

  comm  = MPI.COMM_WORLD
  rank  = comm.Get_rank()

else:
  raise  ValueError('You need to be on a worker.  Fix this.')

if rank == 0:
  print('Solving for cosmology.')

redshift = 0.9873
aa       = 1. / (1. + redshift) 

# Fiducial UNIT cosmology: Om = 0.3089, h = 0.6774, ns = 0.9667, sig8 = 0.8147.
h        = 0.6774
Ob0      = 0.022/h**2
Ocdm0    = 0.3089 - Ob0
cosmo    = cosmology.Cosmology(h=h, Omega_b=Ob0, Omega_cdm=Ocdm0, N_ur = 2.0328, N_ncdm = 1, m_ncdm = 0.06, T_cmb=2.7255)
# growth = cosmo.background.f1(aa)
Plin     = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
b1       = 2.0

BoxSize  = 1000. 
Nmesh    = 1024
stride   = 1
lmax     = 4
poles    = np.arange(0, (1 + lmax), stride).tolist()

if rank == 0:
  print('Solving for poles {}.'.format(poles))

binning  = 'linear'

# Model test power spectrum?  PR(k) * {1 + L2 + ...+ L_lmax}.
test     = True
regress  = True

# cat    = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=500., Nmesh=128, bias=b1, seed=42)

# add RSD
# line_of_sight = [0,0,1]
# cat['RSDPosition'] = cat['Position'] + cat['VelocityOffset'] * line_of_sight

# convert to a MeshSource, using TSC interpolation on 256^3 mesh
# mesh   = cat.to_mesh(window='tsc', Nmesh=256, compensated=True, position='RSDPosition')

if rank == 0:
  print('Solving mesh.')

# Compensated=True, interlaced=True.
mesh                = mesh.linear.LinearMesh(Plin, BoxSize, Nmesh, seed=None, unitary_amplitude=True, inverted_phase=False, remove_variance=None)

# Initialize a complex mesh.
# cmesh             = MeshSource(comm=comm, Nmesh=Nmesh, BoxSize=BoxSize, dtype=np.cdouble)

if rank == 0:
  print('Solving for multipoles.')

# Compute the 2D power and multipoles.
r        = FFTPower(mesh, mode='2d', dk=0.01, kmin=0.0, Nmu=120, los=[0,0,1], poles=poles, regress=regress, test=test)

rpoles   = r.poles

output   = [rpoles['k']]

nks      = len(rpoles['k'])

for ell in poles:
    P = rpoles['power_%d' % ell].real

    if ell == 0:
        P = P - rpoles.attrs['shotnoise']

    output.append(P)

output.append(rpoles['modes'])

output = np.array(output).T

if rank == 0:
  print('Output: ({}, {}), {}'.format(nks, len(poles), output.shape))

opath  = 'box_pk_regress_{}_box_{}_lmax_{}_stride_{}_test_{}.txt'.format(np.int(regress), BoxSize, lmax, stride, np.int(test))

if rank == 0:
  if os.path.exists(opath):
      print('Removing {}.'.format(opath))
      
      os.remove(opath)

  print('Writing {}.'.format(opath))

  np.savetxt(opath, output)
