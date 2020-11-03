import numpy as np
import glob
import json
import h5py

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
        
# from schwimmbad import MultiPool
from hyperion.model import ModelOutput

from simba import simba 
sb = simba()

rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run_sed'

# for z,snap in zip(zeds,snaps):
snap = '078'
z = 2.025
 
with open('m100_sed/galaxy_selection.json', 'r') as fp:
    _dat = json.load(fp)[snap]
    gidx = list(_dat.keys())
    hidx = np.array([int(h['hidx']) for k,h in _dat.items()])
    #_coods = [h['lcone_pos'] for k,h in _dat.items()]

for _gidx in ['3']:#gidx:

    # snap_fname = f'{rt_directory}/snap_{snap}/gal_{_gidx}/snap{snap}.galaxy*.rtout.sed'
    snap_fname = f'{rt_directory}/snap_{snap}/gal_{_gidx}/snap{snap}.galaxy*.rtout.sed'
    fname = glob.glob(snap_fname)[0]
    
    m = ModelOutput(filename=fname)#,group='00000')
    wav,lum = m.get_sed(inclination='all',aperture=-1)
    
    ## High res
    snap_fname = f'{rt_directory}/snap_{snap}_hires/gal_{_gidx}/snap{snap}.galaxy*.rtout.sed'
    fname = glob.glob(snap_fname)[0]
    
    m = ModelOutput(filename=fname)#,group='00000')
    wav_hr,lum_hr = m.get_sed(inclination='all',aperture=-1)
    
    
    # with h5py.File('sed_out.h5','a') as f:
    #     f.create_group(_gidx)
    #     dset = f.create_dataset('%s/Wavelength'%_gidx, data=wav)
    #     dset.attrs['Units'] = 'microns'
    #     dset = f.create_dataset('%s/SED'%_gidx, data=lum)
    #     dset.attrs['Units'] = 'erg/s'


fig,ax = plt.subplots(1,1)
# ax.plot(np.log10(wav),lum.T,alpha=0.1,color='black')
# ax.plot(np.log10(wav_hr),lum_hr.T,alpha=0.1,color='black')
ax.plot(np.log10(wav),lum.T/lum_hr.T,alpha=0.1,color='black')
plt.show()

# with open('data/spectra.json', 'w') as fp:
#     json.dump(spec,fp,sort_keys=True,indent=4,separators=(',', ': '))
 
