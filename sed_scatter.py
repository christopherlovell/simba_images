import sys 
sys.path.append('..') 

import glob
import numpy as np 
import json 
import h5py
 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter, MultipleLocator

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
galaxy_names = {3: 'Smiley', 8: 'Haydon', 51: 'Guillam', 54: 'Alleline',
                94: 'Esterhase', 100: 'Prideaux', 134: 'Bland', 139: 'Lacon'}

for i,(x1,x2,band) in enumerate(zip([60, 90,130,200,290,420],
                                    [80,120,190,280,410,600],
                                    ["PACS (blue)","PACS (green)","PACS (red)",
                                     'SPIRE-1','SPIRE-2','SPIRE-3'])):

    ax.fill_betweenx([0,20],[x1,x1],[x2,x2], color='grey', alpha=0.12*((i+1)/2))
    ax.text(x1*1.05, 15.5, '%s'%band, rotation=90, va='top')


gindexes = [3,8,51,54,94,100,134,139]
for i,gidx in enumerate(gindexes):
    _g = cs.galaxies[gidx]     

    gidx = galaxies[i].GroupID
    with h5py.File('sed_out_hires.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        spec = sb.luminosity_to_flux_density(wav,spec,redshift)

    ax.plot((wav*(1+redshift)),np.std(spec,axis=0), 
            color='C%i'%i, label=galaxy_names[gidx])

ax.set_xlabel('$\lambda \,/\, \mathrm{\mu m}$',size=12)
ax.set_xscale('log')
formatter = ScalarFormatter()
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_minor_formatter(formatter)
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.set_xticks([60,70,80,90,100,200,300,400,500,600,700,800,900,1000])
ax.set_xticklabels(['60','','','','100','200','','400','','','700','','','1000'])

ax.set_ylabel('$\sigma \,/\, \mathrm{mJy}$', size=12)
ax.set_xlim(60,1600)
ax.set_ylim(0,16)
ax.legend(frameon=False, loc='center right') # title='Galaxy:', )

# plt.show()
plt.savefig('plots/sigma_sed.png', dpi=300, bbox_inches='tight')

