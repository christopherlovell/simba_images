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


def obs2rest(x):
    return x / (1+redshift)

def rest2obs(x):
    return x * (1+redshift)

fig,ax = plt.subplots(1,1)
galaxy_names = {3: 'Smiley', 8: 'Haydon', 51: 'Guillam', 54: 'Alleline',
                94: 'Esterhase', 100: 'Prideaux', 134: 'Bland', 139: 'Lacon'}

for i,(x1,x2,band,_max,_alpha) in enumerate(zip([60, 90,130,200,290,420,300], 
                                    [80,120,190,280,410,600,1000],
                                    ["PACS (blue)","PACS (green)","PACS (red)",
                                     'SPIRE-1','SPIRE-2','SPIRE-3','AtLAST'],
                                    [16,16,16,16,16,16,8],
                                    [0.06, 0.12, 0.18, 0.24, 0.3, 0.36, 0.1])):

    ax.fill_betweenx([0,_max],[x1,x1],[x2,x2], color='grey', alpha=_alpha)
    ax.text(x1*1.05, _max * 0.97, '%s'%band, rotation=90, va='top')


dat = np.loadtxt('data/atmospheric_transmission.data', skiprows=5) 
wl = ((2.9e8) / (dat[:,0] * 1e9)) / 1e-6 # micron 
mask = (wl > 300) & (wl < 1000) 
# mask = (wl > 0) & (wl < 10000)
_y = (dat[mask,1] / dat[mask,1].max()) * 8
ax.fill_between(wl[mask], np.zeros(np.sum(mask)), _y, color='blue', alpha=0.15)


gindexes = [3,8,51,54,94,100,134,139]
for i,gidx in enumerate(gindexes):
    _g = cs.galaxies[gidx]     

    gidx = galaxies[i].GroupID
    with h5py.File('sed_out_hires_test.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        spec = sb.luminosity_to_flux_density(wav,spec,redshift)

    ax.plot((wav*(1+redshift)),np.std(spec,axis=0), 
            color='C%i'%i, label=galaxy_names[gidx])

ax.set_xlabel('$\lambda_{\mathrm{obs}} \,/\, \mathrm{\mu m}$',size=12)
ax.set_xscale('log')
secax = ax.secondary_xaxis(-0.2, functions=(obs2rest, rest2obs))
secax.set_xlabel('$\lambda_{\mathrm{rest}} \,/\, \mathrm{\mu m}$',size=12)

formatter = ScalarFormatter()
formatter.set_scientific(False)
for _ax in [ax,secax]:
    _ax.xaxis.set_major_formatter(formatter)
    _ax.xaxis.set_minor_formatter(formatter)
    _ax.xaxis.set_major_locator(MultipleLocator(100))
    _ax.set_xticks([20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000])
    _ax.set_xticklabels(['20','','','','60','','','','100','200','','400','','','700','','','1000'])

ax.set_ylabel('$\sigma \,/\, \mathrm{mJy}$', size=12)
ax.set_xlim(60,1600)
ax.set_ylim(0,16)
ax.text(0.84,0.05,'$z = 2.025$', transform=ax.transAxes, size=12)
ax.legend(frameon=False, loc='upper right') # title='Galaxy:', )

# plt.show()
plt.savefig('plots/sigma_sed.pdf', dpi=300, bbox_inches='tight'); plt.close()

