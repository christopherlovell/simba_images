import sys 
sys.path.append('..') 
 
import numpy as np 
import json 
import h5py
 
import matplotlib 
import matplotlib.pyplot as plt 

from scipy.spatial.distance import cosine
 
import caesar 
 
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

    mask = R < 10;
     
    L = np.sum(np.cross(_coods[mask], (_vels[mask] * _masses[mask,None])),axis=0)
    return L 



snap = '078' 
_dat = json.load(open('m100/galaxy_selection.json','r'))

cs = caesar.load('%sm100n1024_%s.hdf5'%(sb.cs_directory,snap))
    
## spherical coordinates of viewing angle in box coordinates (not pd)
# theta = np.array([180,90,180,90,90,270]) * (np.pi/180.)
# phi = np.array([0,270,180,90,0,0]) * (np.pi/180.)
theta = np.array([90,180,270,0,90,270]) * (np.pi/180.)
phi = np.array([0,0,0,0,90,90]) * (np.pi/180.)


for gidx in [3,8,51,54,94,100,134,139]:
    _g = cs.galaxies[gidx]     

    coods = np.zeros((len(theta),3))
    coods[:,0] = np.sin(theta) * np.cos(phi)
    coods[:,1] = np.sin(theta) * np.sin(phi)
    coods[:,2] = np.cos(theta)
    coods = np.round(coods,2)

    _L = recalculate_ang_mom_vector(_g,ptype=0)
    
    # cos_dist = [round(1 - cosine(_c,_g.rotation['baryon_L']),3) for _c in coods] 
    cos_dist = [round(1 - cosine(_c,_L),3) for _c in coods] 
    
    print(gidx,np.log10(np.linalg.norm(_L)))
    print(cos_dist)


# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(11,11))
# for species,ax in zip(['stellar_L','baryon_L','dm_L','gas_L'],
#                       [ax1,ax2,ax3,ax4]):
# 
#     _L = np.array([gal.rotation[species].value for gal in cs.galaxies[:100]])
#     _L[_L == 0.] = 1e4
#     ax.hist(np.log10(_L[_L[:,0] > 0,0]), histtype='step', label='x = i') 
#     ax.hist(np.log10(_L[_L[:,1] > 0,1]), histtype='step', label='x = j') 
#     ax.hist(np.log10(_L[_L[:,2] > 0,2]), histtype='step', label='x = k') 
#     ax.legend()
#     ax.text(0.1,0.5,species,transform=ax.transAxes)
#     ax.set_xlabel('$\mathrm{log_{10}}(L_{x})$')
#     ax.set_ylabel('$N$')
#    
# 
# plt.savefig('angular_momentum_vector_components.png', dpi=300, bbox_inches='tight')

