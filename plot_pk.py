import  matplotlib; matplotlib.use('PDF')

import  pandas as pd
import  pylab  as pl
import  numpy  as np


fnames  =  []

test    =  True
stride  =  2
lmax    =  4
regress =  1

for boxsize in [30000.]:
  fname = 'box_pk_regress_{}_box_{:.1f}_lmax_{}_stride_{}_test_{}.txt'.format(regress, boxsize, lmax, stride, test)

  # print('Appending {}.'.format(fname))
  # fnames.append(fname)

# fnames.append('submit/Pkl_linear_Wilson_UNIT_DESI_Shadab_HOD_snap97_ELG_v1_4col_1_TEST_0.txt')
# fnames.append('Pkl_linear_Krolewski_UNIT_DESI_Shadab_HOD_snap97_ELG_v1_4col_1_TEST_1.txt')
# fnames.append('Pkl_linear_Wilson_UNIT_DESI_Shadab_HOD_snap97_ELG_v1_4col_1_TEST_0.txt')

boxes              = [3, 2, 1]
alphas             = [1.0, 0.6, 0.3]

boxes              = [3]

for boxsize, alpha in zip(boxes, alphas):
  fname            = 'Pkl_linear_Wilson_UNIT_DESI_Shadab_HOD_snap97_ELG_v1_4col_{}Gpc_regress_{}_lmax_{}_stride_{}{}.txt'.format(boxsize, regress, lmax, stride, '_TEST' if test else '')

  # Get header.
  with open(fname) as f:
    header         = f.readline()
    header         = header.replace('\n', '').replace('#','').split()
    
  dat              = pd.read_csv(fname, sep='\s+', comment='#', names=header).applymap(complex)
  
  poles            = [x for x in header[1:-1]] 
  
  '''
  k                = dat[:,0]
  N                = dat[:,-1]
  poles            = np.arange(0, (lmax + 1), stride)
  Ps               = dat[:, 1:-1]
  
  assert  len(Ps.T) == len(poles)
  
  for pole, P in zip(poles, Ps.T):
    pl.plot(k, k*np.abs(P), label=r'$k \cdot P_{}$'.format(pole))
  '''
  
  for pole, color in zip(poles, ['b', 'g', 'r']):
    even           = np.int(pole[1]) % 2 
    
    if even == 0:
      k              = dat['k'].to_numpy().real
      P              = dat[pole].to_numpy().real

      if pole == 'P0':
        label = r'$k \cdot P0, {}Gpc$'.format(boxsize)

      else:
        label = ''
         
      pl.plot(k, k * P, label=label, color=color, alpha=alpha)   

  pl.xlim(   0.0,  0.2)
  pl.ylim(-100.0, 800.)

  pl.legend(ncol=2, loc=1, frameon=False)

  pl.savefig('pk.pdf')
