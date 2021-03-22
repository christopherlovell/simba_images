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

def recalculate_ang_mom_vector(_g,ptype=4):
    """
    Recalculate angular momentum vector
    """
    fname = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/snap_m100n1024_078.hdf5'
        
    if ptype==0: plist = np.sort(_g.glist)
    elif ptype==4: plist = np.sort(_g.slist)
    
    with h5py.File(fname, 'r') as f:
        _h = f['Header'].attrs['HubbleParam']
        _a = f['Header'].attrs['Time']

        _masses = f['PartType%s/Masses'%ptype][plist] 
        _vels = f['PartType%s/Velocities'%ptype][plist] * _a * (1/_h) 
        _coods = f['PartType%s/Coordinates'%ptype][plist] * (1/_h)
         
    _vels  -= _g.vel.value 
    _coods -= _g.pos.value 

    R = np.sqrt(np.sum(_coods**2,axis=1))
    mask = R < 100;
    L = np.sum(np.cross(_coods[mask], (_vels[mask] * _masses[mask,None])),axis=0)
    return L 


snap = '078'
z = 2.025
rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run'
#rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run_sed'

_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)

_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]


## spherical coordinates of viewing angle in box coordinates (not pd)
# phi   = np.array([0,90,180,270,0,0]) * (np.pi/180.)
# theta = np.array([90,90,90,90,0,180]) * (np.pi/180.)
np.random.seed(0); _N = 50
theta = np.arccos(1 - 2 * np.random.rand(_N)) #* (180 / np.pi)
phi   = 2 * np.pi * np.random.rand(_N) #* (180 / np.pi)

# filt_wl, filt_trans = sb.scuba850_filter()

norm = mpl.colors.Normalize(vmin=0, vmax=1)
m = cm.ScalarMappable(norm=norm, cmap=cm.copper)

gindexes = [3,8,51,54,94,100,134,139]
cos_dist = {gidx: None for gidx in gindexes}
S350 = {gidx: None for gidx in gindexes}
for i,gidx in enumerate(gindexes):
    print(gidx)
    _g = cs.galaxies[gidx]     

    coods = np.zeros((len(theta),3))
    coods[:,0] = np.sin(theta) * np.cos(phi)
    coods[:,1] = np.sin(theta) * np.sin(phi)
    coods[:,2] = np.cos(theta)
    coods = np.round(coods,2)

    _L = recalculate_ang_mom_vector(_g,ptype=0)
    cos_dist[gidx] = [round(1 - cosine(_c,_L),3) for _c in coods] 

    gidx = galaxies[i].GroupID
    with h5py.File('sed_out_hires.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
    
    S350[gidx] = spec[:,np.argmin(np.abs((wav*(1+z))-250))]


fig,ax = plt.subplots(1,1)

galaxy_names = {3: 'Smiley', 8: 'Haydon', 51: 'Guillam', 54: 'Alleline',
                94: 'Esterhase', 100: 'Prideaux', 134: 'Bland', 139: 'Lacon'}


for i,gidx,_alpha in zip([0,2,4,7,1,3,5,6],
                         [3,51,94,139,8,54,100,134],
                         [1,1,1,1,0.3,0.3,0.3,0.3]):

    print(i,gidx)
    ax.scatter(S350[gidx] / S350[gidx].max(), np.abs(cos_dist[gidx]),
               color='C%i'%i, label=galaxy_names[gidx],alpha=_alpha)

    if i in [0,2,4,7]:
        binlimits = np.linspace(0,1,20)
        _Cbin = np.array([binned_statistic(S350[gidx] / S350[gidx].max(), np.abs(cos_dist[gidx]),
                                          statistic=lambda y: np.percentile(y, p), bins=binlimits)[0] \
                                          for p in [16,50,84]]).T
    
        plt.plot(binlimits[1:] - np.diff(binlimits)[0]/2, _Cbin[:,1], color='C%i'%i)
    #plt.fill_between(bins[1:], _Cbin[:,0], _Cbin[:,2], alpha=0.4)


# for i,gidx in zip([1,3,5,6],):
#     print(i,gidx)
#     ax.scatter(S350[gidx] / S350[gidx].max(), np.abs(cos_dist[gidx]),
#                color='C%i'%i, alpha=0.3, label=galaxy_names[gidx])

ax.set_xlim(None,1)
ax.set_ylim(0,1)
ax.set_ylabel('$C_i$', size=14)
ax.set_xlabel('$S_{250,i} \,/\, S_{250,\mathrm{max}}$', size=14)
ax.legend()
# plt.show()
plt.savefig('plots/cosine_similarity.png',dpi=300,bbox_inches='tight')

