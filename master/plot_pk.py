import  matplotlib; matplotlib.use('PDF')

import  pandas             as pd
import  pylab              as pl
import  numpy              as np
import  matplotlib.pyplot  as plt


fnames  =  []

test    =  False
stride  =  2
lmax    =  4
regress =  0

alphas  = np.arange(1.0, 0.0, -0.25)

colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

if test:
  for jj, (boxsize, alpha) in enumerate(zip([3000., 1000.], alphas)):
    fname = 'box_pk_regress_{}_box_{:.1f}_lmax_{}_stride_{}_test_{}.txt'.format(regress, boxsize, lmax, stride, np.int(test))

    print('Appending {}.'.format(fname))

    dat              = np.loadtxt(fname)                                                                                                                                                                                                 
    k                = dat[:,0]                                                                                                                                                                                                            
    N                = dat[:,-1]                                                                                                                                                                                                           
    poles            = np.arange(0, (lmax + 1), stride)                                                                                                                                                                                       
    Ps               = dat[:, 1:-1]                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    assert  len(Ps.T) == len(poles)                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    for pole, P, amp, color in zip(poles, Ps.T, 1. + jj + 0.0 * np.arange(len(poles)), colors):                                                                                                                                              
      pl.plot(k, amp * k*np.abs(P), label=r'$k \cdot P_{}$'.format(pole), lw=0.1)
    
    pl.xlim(   0.0,  0.2)
    pl.ylim(-100.0, 800.)

    pl.legend(ncol=2, loc=1, frameon=False)

    pl.title(r'$P_s(k, \mu) = ({}) \cdot P_R(k)$'.format(r' + {}'.join(['L_{}'.format(x) for x in poles])))
    
    pl.savefig('box_pk.pdf')

  exit(0)
    
# fnames.append('submit/Pkl_linear_Wilson_UNIT_DESI_Shadab_HOD_snap97_ELG_v1_4col_1_TEST_0.txt')
# fnames.append('Pkl_linear_Krolewski_UNIT_DESI_Shadab_HOD_snap97_ELG_v1_4col_1_TEST_1.txt')
# fnames.append('Pkl_linear_Wilson_UNIT_DESI_Shadab_HOD_snap97_ELG_v1_4col_1_TEST_0.txt')

alphas             = np.arange(1.0, 0.0, -0.4)

boxes              = [3, 1]

colors             = plt.rcParams['axes.prop_cycle'].by_key()['color']

for boxsize, alpha in zip(boxes, alphas):
  fname            = 'Pkl_linear_Wilson_UNIT_DESI_Shadab_HOD_snap97_ELG_v1_4col_{}Gpc_regress_{}_lmax_{}_stride_{}{}.txt'.format(boxsize, regress, lmax, stride, '_TEST' if test else '')

  # Get header.
  with open(fname) as f:
    header         = f.readline()
    header         = header.replace('\n', '').replace('#','').split()
    
  dat              = pd.read_csv(fname, sep='\s+', comment='#', names=header).applymap(complex)

  print(dat)
  
  poles            = [x for x in header[1:-1]] 
  
  for pole, color in zip(poles, colors):
    even           = np.int(pole[1]) % 2 
    
    k              = dat['k'].to_numpy().real
    P              = dat[pole].to_numpy().real

    if boxsize == 3:
      label = r'$k \cdot {}, {}Gpc$'.format(pole, boxsize)

    else:
      label = ''
      
    pl.plot(k, k * P, label=label, color=color, alpha=alpha)   
    # pl.loglog(k, np.abs(P), label=label, color=color, alpha=alpha)
    
  pl.xlim(0.0,  0.2)

  # pl.ylim(100.0, 5.e4)   
  pl.ylim(-100.0, 800.)

  pl.legend(ncol=1, loc=1, frameon=False)

  pl.savefig('master_pk.pdf')
