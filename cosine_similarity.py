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

_dat = json.load(open('m100/galaxy_selection.json','r'))

cs = caesar.load('%sm100n1024_%s.hdf5'%(sb.cs_directory,snap))
    
## spherical coordinates of viewing angle in box coordinates (not pd)
phi   = np.array([0,90,180,270,0,0]) * (np.pi/180.)
theta = np.array([90,90,90,90,0,180]) * (np.pi/180.)
# np.random.seed(0)
# theta = np.arccos(1 - 2 * np.random.rand(_N)) #* (180 / np.pi)
# phi   = 2 * np.pi * np.random.rand(_N) #* (180 / np.pi)


filt_wl, filt_trans = sb.scuba850_filter()


norm = mpl.colors.Normalize(vmin=0, vmax=1)
m = cm.ScalarMappable(norm=norm, cmap=cm.copper)

for gidx,_N in zip([3],#,8,51,54,94,100,134,139],
                   [10,50,50,50,50,10,50,50]):
    


    print(gidx)
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

    # snap_fname = f'{rt_directory}/snap_{snap}_hires2/gal_{gidx}/snap{snap}.galaxy*.rtout.sed'
    snap_fname = f'{rt_directory}/snap_{snap}/gal_{gidx}/snap{snap}.galaxy*.rtout.sed'
    fname = glob.glob(snap_fname)[0]

    wav,spec = sb.get_spectrum(fname,gal_id=None)
    print(spec.shape)
    # m = ModelOutput(filename=fname)#,group='00000')
    # wav,spec = m.get_sed(inclination='all',aperture=-1)

    flux_850 = sb.calc_mags(wav.copy(), spec.copy(), z,
                            filt_wl=filt_wl, filt_trans=filt_trans).value
    
    with h5py.File('sed_out.h5','w') as f:
        f.require_group(str(gidx))
        dset = f.create_dataset('%s/Wavelength'%gidx, data=wav)
        dset.attrs['Units'] = 'microns'
        dset = f.create_dataset('%s/SED'%gidx, data=spec)
        dset.attrs['Units'] = 'erg/s'
        dset = f.create_dataset('%s/850 flux'%gidx,data=flux_850)
        dset.attrs['Units'] = 'mJy'
        f.create_dataset('%s/cosine similarity'%gidx,data=cos_dist)

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(5,15))

    for ax in [ax1,ax2]:
        [ax.plot(np.log10(wav.value * (1+z)), s, alpha=1,
            c=m.to_rgba(np.abs(cos_dist[i]))) for i,s in enumerate(spec)]

    mean_spec = np.mean(spec.value,axis=0)
    [ax3.plot(wav.value * (1+z), s/mean_spec, alpha=1, 
            c=m.to_rgba(np.abs(cos_dist[i]))) for i,s in enumerate(spec)]


    for ax in [ax1,ax2]:
        ax.set_ylim(0,)
        ax.set_xlabel('$\mathrm{log_{10}}(\lambda \,/\, \AA)$', size=15)
        ax.set_ylabel('$\mathrm{erg \,/\, s}$', size=15)
    
    for ax in [ax2,ax3]:
        ax.set_xlim(2,3)

    mean_flux = np.round(np.mean(flux_850),2)
    ax1.text(0.2,0.9,f'$z = {z}$',size=13,transform=ax1.transAxes)
    ax1.text(0.2,0.8,'$S_{850} = %s$'%mean_flux,size=13,transform=ax1.transAxes)
    ax3.set_xlabel('$\lambda \,/\, \AA$',size=15)
    ax3.set_ylabel('$\mathrm{Flux}_i / \mathrm{Flux_{mean}}$',size=15)
    ax3.set_ylim(0.7,1.3)
    ax3.set_xlim(250,1000)

    cax = fig.add_axes([0.14, 0.7, 0.04, 0.15])
    cbar = fig.colorbar(m, aspect=10, orientation='vertical',
                        cax=cax, label='cosine similarity')

    # for i,_l in enumerate(lum_hr):
    #     ax.plot(np.log10(wav_hr),_l,alpha=0.1,color=m.to_rgba(np.abs(cos_dist[i])))
    # ax.plot(np.log10(wav),lum.T/lum_hr.T,alpha=0.1,color='black')
    plt.show()
    # plt.savefig(f'plots/cosine_similarity_g{gidx}.png',dpi=300,bbox_inches='tight')
    plt.close()

