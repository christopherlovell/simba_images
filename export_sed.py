import sys 
sys.path.append('..') 

import glob
import numpy as np 
import h5py
 
from simba import simba 
sb = simba() 

snap = '078'
z = 2.025
rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run_sed/'
# rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run/'

for gidx in [3,8,51,54,94,100,134,139]:
# for gidx in [3,8,51,94,100]:
    snap_fname = f'{rt_directory}/snap_{snap}_hires_orthogonal/gal_{gidx}/snap{snap}.galaxy*.rtout.sed'
    # snap_fname = f'{rt_directory}/snap_{snap}_hires/gal_{gidx}/snap{snap}.galaxy*.rtout.sed'
    # snap_fname = f'{rt_directory}/snap_{snap}/gal_{gidx}/snap{snap}.galaxy*.rtout.sed'
    print(snap_fname)
    fname = glob.glob(snap_fname)[0]

    wav,spec = sb.get_spectrum(fname,gal_id=None)
    print(spec.shape)
    # m = ModelOutput(filename=fname)#,group='00000')
    # wav,spec = m.get_sed(inclination='all',aperture=-1)
    
    with h5py.File('sed_out_orthogonal.h5','a') as f:
        f.require_group(str(gidx))
        dset = f.create_dataset('%s/Wavelength'%gidx, data=wav)
        dset.attrs['Units'] = 'microns'
        dset = f.create_dataset('%s/SED'%gidx, data=spec)
        dset.attrs['Units'] = 'erg/s'
        # dset = f.create_dataset('%s/850 flux'%gidx,data=flux_850)
        # dset.attrs['Units'] = 'mJy'
        # f.create_dataset('%s/cosine similarity'%gidx,data=cos_dist)

