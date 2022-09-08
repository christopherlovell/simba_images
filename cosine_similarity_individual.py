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
# cs.data_manager = DataManager(cs)

_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]

# cs = caesar.load('%sm100n1024_%s.hdf5'%(sb.cs_directory,snap))

## spherical coordinates of viewing angle in box coordinates (not pd)
# phi   = np.array([0,90,180,270,0,0]) * (np.pi/180.)
# theta = np.array([90,90,90,90,0,180]) * (np.pi/180.)
np.random.seed(0); _N = 50
theta = np.arccos(1 - 2 * np.random.rand(_N)) #* (180 / np.pi)
phi   = 2 * np.pi * np.random.rand(_N) #* (180 / np.pi)

norm = mpl.colors.Normalize(vmin=0, vmax=1)
m = cm.ScalarMappable(norm=norm, cmap=cm.copper)

gindexes = [3,8,51,54,94,100,134,139]

gidx = 3
if True:
    print(gidx)
    _g = cs.galaxies[gidx]     

    coods = np.zeros((len(theta),3))
    coods[:,0] = np.sin(theta) * np.cos(phi)
    coods[:,1] = np.sin(theta) * np.sin(phi)
    coods[:,2] = np.cos(theta)
    coods = np.round(coods,2)

    _L = recalculate_ang_mom_vector(_g,ptype=0)
    cos_dist = [round(1 - cosine(_c,_L),3) for _c in coods] 

    with h5py.File('sed_out_hires_test.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        spec = sb.luminosity_to_flux_density(wav,spec,z)
    
    
    fig,(ax2,ax3) = plt.subplots(2,1,figsize=(5,6.7))
    plt.subplots_adjust(hspace=0.)
    
    [ax2.plot(wav * (1+z), s, alpha=1,
            c=m.to_rgba(np.abs(cos_dist[i]))) for i,s in enumerate(spec)]

    mean_spec = np.mean(spec,axis=0)
    [ax3.plot(wav * (1+z), s/mean_spec, alpha=1, 
            c=m.to_rgba(np.abs(cos_dist[i]))) for i,s in enumerate(spec)]


    ax2.set_ylim(0,)
    #ax2.set_xlabel('$\mathrm{log_{10}}(\lambda \,/\, \AA)$', size=13)
    ax2.set_ylabel('$S \,/\, \mathrm{mJy}$', size=13)
    
    ax3.set_ylabel('$S_i \,/\, \mathrm{ \left< S \\right>}$',size=13)
    ax3.set_ylim(0.0,3)
    # ax3.set_ylim(0.3,1.7)
    
    for ax in [ax2,ax3]:
        ax.set_xlim(0.1,1000)
        ax.set_xlabel('$\lambda_{\mathrm{obs}} \,/\, \mathrm{\mu m}$',size=12)
        ax.set_xscale('log')
        # formatter = ScalarFormatter()
        # formatter.set_scientific(False)
        # ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_formatter(formatter)
        # ax.xaxis.set_major_locator(MultipleLocator(100))
        # ax.set_xticks([60,70,80,90,100,200,300,400,500,600,700,800,900,1000])
        # ax.set_xticklabels(['60','','','','100','200','','400','','','700','','','1000'])
        ax.grid(alpha=0.3)
    
    ax3.hlines(1, 60,1000, linestyle='dashed', color='black')
    # mean_flux = np.round(np.mean(flux_850),2)
    ax2.text(0.03,0.9,f'$z = {z}$',size=12,transform=ax2.transAxes)
    # ax1.text(0.2,0.8,'$S_{850} = %s$'%mean_flux,size=13,transform=ax1.transAxes)
    # ax3.set_xlabel('$\lambda \,/\, \AA$',size=13)
    #ax3.set_xlim(250,1000)

    cax = fig.add_axes([0.68, 0.7, 0.04, 0.15])
    cbar = fig.colorbar(m, aspect=10, orientation='vertical',
                        cax=cax, label='$\left| \,\mathrm{Cosine \; similarity}\, \\right|$')
    
    ax2.set_xticklabels([])

    # for i,_l in enumerate(lum_hr):
    #     ax.plot(np.log10(wav_hr),_l,alpha=0.1,color=m.to_rgba(np.abs(cos_dist[i])))
    # ax.plot(np.log10(wav),lum.T/lum_hr.T,alpha=0.1,color='black')
    plt.show()
    # plt.savefig(f'plots/cosine_similarity_g{gidx}.pdf',dpi=300,bbox_inches='tight'); plt.close()


