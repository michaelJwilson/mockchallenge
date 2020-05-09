import   os
import   pandas       as     pd
import   numpy        as     np
import   time
import   sys

# On worker.
if os.getenv('SLURM_NODEID') is not None:
    from   nbodykit.lab import *
    from   mpi4py       import MPI

    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

else:
    # rank = 0
    raise  ValueError('You need to be on a worker.  Fix this.')

# Input arguments: file paths and names, maximum R, number of bins, box size
# Number of bins is actually the number of *bins*, and not edges, i.e.
# if you want 40 bins spaced at 5 h^-1 Mpc between 0 and 200, specify Nbins = 40.
regress          = True
test             = False
master           = True
Nmesh            = 1024
BoxSize          = 2000.
binning          = 'linear'

stride           = 1

# Inclusive.
lmax             = 4
poles            = np.arange(0, (1 + lmax), stride).tolist()

t0               = time.time()

output_path      = '/project/projectdirs/desi/users/mjwilson/mockchallenge/'

box              = '{}Gpc'.format(np.int(np.floor(BoxSize / 1000.)))

input_path       = '/global/project/projectdirs/desi/cosmosim/UNIT-BAO-RSD-challenge/{}/ELG-single-tracer/'.format(box)
input_file       = 'UNIT_DESI_Shadab_HOD_snap97_ELG_v1_4col.txt'
	
# output_name_pk = 'Pk2D_%s_Wilson_' % (binning) + input_file.split('.txt')[0] + '_{}_regress_{:d}_lmax_{}_stride_{}.txt'.format(box, np.int(regress), lmax, stride)
output_name_pk_l = 'Pkl_%s_Wilson_'  % (binning) + input_file.split('.txt')[0] + '_{}_regress_{:d}_lmax_{}_stride_{}{}.txt'.format(box, np.int(regress), lmax, stride, '_TEST' if test else '')

print('Solving for {}.'.format(input_file))
print('Writing:\n{}'.format(output_name_pk_l))

# Create catalog.
names            = ['x','y','z','z_red']

# If comments present in file:
# see https://nbodykit.readthedocs.io/en/latest/catalogs/reading.html#reading-multiple-files 
# cat            = pd.read_csv(input_path + input_file, comment='#', names=names, sep='\s+')
# cat            = ArrayCatalog({'x': cat['x'], 'y': cat['y'], 'z': cat['z'], 'z_red': cat['z_red']})

cat              = CSVCatalog(input_path + input_file, names=names)

if test:
    # Real-space power spectrum input, multiplied by polys in project_to_basis.
    cat['RSDPosition']   = cat['x'][:,None] * [1, 0, 0] + cat['y'][:,None] * [0, 1, 0] + cat['z'][:,None] * [0, 0, 1]
    
else:
    cat['RSDPosition']   = cat['x'][:,None] * [1, 0, 0] + cat['y'][:,None] * [0, 1, 0] + cat['z_red'][:,None] * [0, 0, 1]

cat.attrs['BoxSize'] = BoxSize

# Create mesh (real space).
mesh                 = cat.to_mesh(resampler='tsc', Nmesh=Nmesh, compensated=True, interlaced=True, position='RSDPosition')
# Need mesh to be big enough so that nyquist frequency is > kmax
# pi * Nmesh/Lbox = 3.22 h Mpc^{-1} for Nmesh = 1024 and Lbox = 1000 h^{-1} Mpc, 
# vs kmax = 2.262 h Mpc^{-1}
# Interlacing + TSC gives very unbiased results up to the Nyquist frequency:
# https://nbodykit.readthedocs.io/en/latest/cookbook/interlacing.html
k_Nyquist            = np.pi * Nmesh/BoxSize

r                    = FFTPower(mesh, mode='2d', dk=0.01, kmin=0.0, kmax=k_Nyquist, Nmu=120, los=[0,0,1], regress=regress, test=test, poles=poles)
Pkmu                 = r.power
rpoles               = r.poles

'''
f = open(output_path + output_name_pk,'w')
f.write('# Wilson mock challenge 1, input from ' + input_path + input_file + '\n')
f.write('# Estimated shot noise subtracted from power spectra\n')
f.write('# Estimated shot noise: %.5f\n' % (Pkmu[:,0].attrs['shotnoise']))
f.write('# Code to generate this measurement in ' + __file__ + '\n')
f.write('# Boxsize = %.1f\n'  % BoxSize)
f.write('# Nmesh =  %i\n' % Nmesh)
f.write('# Binning = ' + binning + '\n')
f.write('# k mu pk Nmodes\n')
for i in range(Pkmu.shape[1]):
	Pk = Pkmu[:,i]
	for j in range(len(Pk['k'])):
		f.write('%20.8e %20.8e %20.8e %i\n' % (Pk['k'][j], Pkmu.coords['mu'][i], Pk['power'][j].real-Pk.attrs['shotnoise'], 
			Pkmu.data['modes'][:,i][j]))
	
f.close()
'''

assert  poles[0] == 0

f = open(output_path + output_name_pk_l,'w')
f.write('# k\t{}Nmodes\n'.format('\t'.join(['P{} '.format(x) for x in poles])))
f.write('# Wilson mock challenge 1, input from ' + input_path + input_file + '\n')
f.write('# Estimated shot noise subtracted from power spectra\n')
f.write('# Estimated shot noise: %.5f\n' % (Pkmu[:,0].attrs['shotnoise']))
f.write('# Code to generate this measurement in {}/{}\n'.format(os.getcwd(), __file__))
f.write('# Boxsize = %.1f\n'  % BoxSize)
f.write('# Nmesh =  %i\n' % Nmesh)
f.write('# Binning = ' + binning + '\n')
f.write('# Regress = {}\n'.format(np.int(regress)))
f.write('# Computation time = {}\n'.format(time.time()-t0))

output = {}

for pole in poles:
    # Labelled by multipole, not index. 
    output[pole] = rpoles['power_{}'.format(pole)]
    
for i in range(len(output[0])):
    sstr      =  '{:20.8e}'.format(rpoles['k'][i])
    sstr     +=  '\t{:20.8e}'.format(output[0][i] - rpoles.attrs['shotnoise'])
    
    for pole in poles[1:]:
        sstr +=  '\t{:20.8e}'.format(output[pole][i])

    sstr     +=  '\t{:20.8e}'.format(rpoles.data['modes'][i])
    sstr     +=  '\n'
 
    f.write(sstr)
    
f.close()

print('Total time (s): ', time.time()-t0)
