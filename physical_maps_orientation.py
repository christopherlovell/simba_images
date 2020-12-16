import sys
sys.path.append('..')

import numpy as np
import h5py
import json

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import sphviewer as sph
from sphviewer.tools import cmaps
import caesar

import astropy.units as u
from astropy.cosmology import Planck13
from astropy import constants

from hyperion.model import ModelOutput

from simba import simba 
sb = simba()


# g_idx = int(sys.argv[1])


def plot_dist(fig, ax,P,coods,cmap,
              vmin=1e-4,vmax=None,extent=300,
              p=0,t=0,roll=0):
 
    C = sph.Camera(x=coods[0], y=coods[1], z=coods[2], 
                   r='infinity', zoom=1,
                   t=t, p=p, roll=roll,
                   extent=[-extent,extent,-extent,extent],
                   xsize=512, ysize=512)
 
    S = sph.Scene(P, Camera=C)
    R = sph.Render(S)
    #R.set_logscale()
    img = R.get_image()
    # img = img / img.mean()

    if vmax is None:    
        vmax = img.max()

    if vmin > vmax:
        print("vmin larger than vmax! Setting lower")
        vmin = img.max() / 10
    print("vmin,vmax:",vmin,vmax)
    
    cNorm  = colors.LogNorm(vmin=vmin,vmax=vmax)
    sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    ax.imshow(img, extent=[-extent,extent,-extent,extent],
              cmap=cmap, norm=cNorm)

    return sm,img


snap = '078'

cs = caesar.load('%sm100n1024_%s.hdf5'%(sb.cs_directory,snap))
redshift = cs.simulation.redshift
_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]

fig,axes = plt.subplots(4,4,figsize=(11,11)) 
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for k,gidx in enumerate(['3','8','51','54']):#,'94','100','134','139']):
# for k,gidx in enumerate(['94','100','134','139']):
    print("gidx:",gidx)
    hcood =  np.array(_dat[snap][gidx]['pos'])
    hidx =  _dat[snap][gidx]['hidx']
    
    fdir = '/blue/narayanan/c.lovell/simba/m100n1024/out/snap_078/subset_%05d.h5'%hidx
    
    with h5py.File(fdir,'r') as f:
        _a = np.float32(1. / (1+f['Header'].attrs['Redshift']))
        _h = np.float32(sb.cosmo.h)
        _temp = (1. / _h) * _a
        print(_a,_h,_temp)
        hcood *= _temp
             
        print("Getting gas particles...")
        halog_pos = f['PartType0/Coordinates'][()] * _temp
        halog_dust = f['PartType0/Dust_Masses'][()]
        halog_sfr = f['PartType0/StarFormationRate'][()]


    # transform rotation (alpha -> theta, beta -> phi)
    theta = np.array([90,90, 90, 90,0,180]) * np.pi/180.
    phi = np.array([0, 90,180,270,0,  0]) * np.pi/180.

    theta += np.pi
    phi += np.pi

    x = np.round(np.sin(theta),2) * np.round(np.cos(phi),2) 
    y = np.round(np.sin(theta),2) * np.round(np.sin(phi),2) 
    z = np.round(np.cos(theta),2) 
    
    # phi = np.pi - phi
    alpha = np.arctan( (np.round(np.sin(phi),2) * np.round(np.sin(theta),2)) / np.round(np.cos(theta),2) ) 
    alpha[(z<0) & (y>=0)] += np.pi 
    alpha[(z<0) & (y<0)] -= np.pi 
    alpha[(z==0) & (y>0)] = np.pi/2 
    alpha[(z==0) & (y<0)] = -np.pi/2 
         
    beta = np.arctan( (np.round(np.sin(theta),2) * np.round(np.cos(phi),2)) / np.round(np.cos(theta),2) ) 
    # beta[(z<0) & (x>=0)] += np.pi  
    # beta[(z<0) & (x<0)] -= np.pi  
    # beta[(z==0) & (x>0)] = np.pi/2  
    # beta[(z==0) & (x<0)] = -np.pi/2  

    alpha *= 180./np.pi
    beta  *= 180./np.pi
    print(alpha,beta)
    
    for ax,_orientation,_p,_t,_roll in zip(axes[:3,k], [0, 1, 4], 
                                           beta[[0,1,4]], alpha[[0,1,4]], [270,180,270]):

        print("#########\nRendering orientation %s\np=%s, t=%s, roll=%s\n##########"%\
                (_orientation,_p,_t,_roll))
        
        if _orientation == 0:
            mask = np.random.rand(len(halog_pos)) < 1
            #Pg = sph.Particles(halog_pos[mask], halog_mass[mask] * 1e10) 
            Pd = sph.Particles(halog_pos[mask], halog_dust[mask] * 1e10) 
            # Psfr = sph.Particles(halog_pos[mask], halog_sfr[mask]) 


        extent = 24.5
        sm,img = plot_dist(fig, ax, Pd, hcood, cmaps.twilight(), 
                             vmin=1e2, vmax=1e10, extent=extent, p=_p, t=_t, roll=_roll) 
        # sm4,img4 = plot_dist(fig, ax4, Psfr, hcood, cmaps.sunlight(), 
        #                      vmin=0.1, extent=extent, p=_p, t=_t, roll=_roll) 


    ## ---- plot SEDs
    # gidx = galaxies[k].GroupID
    with h5py.File('sed_out_hires.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        f850 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,redshift)
        spec = sb.luminosity_to_flux_density(wav,spec,redshift)


    [axes[3,k].plot(np.log10(wav*(1+redshift)), s, color='black',alpha=0.2) for s in spec]
    
    ## ---- plot orthogonal SEDs
    with h5py.File('sed_out_orthogonal.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        f850 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,redshift)
        spec = sb.luminosity_to_flux_density(wav,spec,redshift)

    for axidx,(s,c) in enumerate(zip(spec[[0,3,4]],['C0','C1','C2'])):
        axes[3,k].plot(np.log10(wav*(1+redshift)), s, color=c,alpha=1)
        circle = plt.Circle((0, 0), 20, color=c, fill=False)
        axes[axidx,k].add_artist(circle)

    axes[3,k].set_xlim(2.01,3)
    axes[3,k].set_ylim(0,93)
    axes[3,k].grid(alpha=0.1)
    axes[3,k].text(0.95, 0.9, '$S_{850}=%.2f \; \mathrm{mJy}$'%np.median(f850).value, 
                   ha='right', transform=axes[3,k].transAxes)
         
    axes[0,k].text(0.7, 0.9, 'g:%s'%gidx, transform=axes[0,k].transAxes, color='white')


cax = fig.add_axes([0.91, 0.698, 0.01, 0.18])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
cbar.ax.set_ylabel('$\mathrm{M_{\odot} \; kpc^{-2}}$', size=12)
    
axes[3,0].set_ylabel('$\mathrm{mJy}$',size=12)

for ax in axes[-1,:]:
    ax.set_xlabel('$\mathrm{log_{10}(\lambda \,/\, \mu m)}$',size=12)
for ax in axes[:3,0]: 
    ax.set_ylabel('$\mathrm{kpc}$', size=12)

for ax in axes[1:-1,:].flatten():
    ax.set_xticklabels([])

for ax in axes[0,:]:
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('$\mathrm{kpc}$')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(labeltop=True)

for ax in axes[:,1:].flatten():
    ax.set_yticklabels([])


# plt.show() 
plt.savefig('plots/dust_maps.png', dpi=300, bbox_inches='tight')
plt.close()


