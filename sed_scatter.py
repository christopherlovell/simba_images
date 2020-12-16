import sys 
sys.path.append('..') 

import glob
import numpy as np 
import json 
import h5py
 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

from scipy.spatial.distance import cosine

from hyperion.model import ModelOutput

import caesar 
from caesar.data_manager import DataManager

from simba import simba 
sb = simba() 


snap = '078'
z = 2.025
rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run'
#rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run_sed'

_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)
redshift = cs.simulation.redshift 
_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]


fig,ax = plt.subplots(1,1)

gindexes = [3,8,51,54,94,100,134,139]
for i,gidx in enumerate(gindexes):
    _g = cs.galaxies[gidx]     

    gidx = galaxies[i].GroupID
    with h5py.File('sed_out_hires.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        spec = sb.luminosity_to_flux_density(wav,spec,redshift)

    ax.plot(np.log10(wav*(1+redshift)),np.std(spec,axis=0), label=gidx)


ax.set_ylabel('$\sigma \,/\, \mathrm{mJy}$', size=12)
ax.set_xlabel('$\lambda \,/\, \AA$', size=12)
ax.set_xlim(2,3)
ax.set_ylim(0,)
ax.legend(title='Galaxy:', frameon=False)

# plt.show()
plt.savefig('plots/sigma_sed.png', dpi=300, bbox_inches='tight')

