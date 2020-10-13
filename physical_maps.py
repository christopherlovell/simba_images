import sys
sys.path.append('..')

import numpy as np
import h5py
import json

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import sphviewer as sph
from sphviewer.tools import cmaps

import astropy.units as u
from astropy.cosmology import Planck13
from astropy import constants

from hyperion.model import ModelOutput

from simba import simba 
sb = simba()

gidx = '3'
snap = '078'
_dat = json.load(open('m100/galaxy_selection.json','r'))
hcood =  np.array(_dat[snap][gidx]['pos'])
hidx =  _dat[snap][gidx]['hidx']

fdir = '/blue/narayanan/c.lovell/simba/m100n1024/out/snap_078/subset_%05d.h5'%hidx
# fdir ='/cosma7/data/dp104/dc-dave2/sim/m100n1024/s50j7k/snap_m100n1024_%s.hdf5'%snap



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


with h5py.File(fdir,'r') as f:
    _a = np.float32(1. / (1+f['Header'].attrs['Redshift']))
    _h = np.float32(sb.cosmo.h)
    _temp = _a / _h
    print(_temp)

    print("Getting gas particles...")
    #halog_pos = (np.array(f['PartType0/Coordinates'][()]) * _temp).astype(np.float32)
    halog_pos = f['PartType0/Coordinates'][()] * _temp
    halog_mass = f['PartType0/Masses'][()]
    halog_dust = f['PartType0/Dust_Masses'][()]
    halog_sfr = f['PartType0/StarFormationRate'][()]

    print("Getting star particles...")
    halos_pos = (np.array(f['PartType4/Coordinates'][()]) * _temp).astype(np.float32)
    halos_pmass = f['PartType4/Masses'][()]

    print("Getting dark matter particles...")
    halod_pos = (f['PartType1/Coordinates'][()] * _temp).astype(np.float32)

hcood *= _temp


# _orientation = 0
# p,roll = 0,0

for _orientation,_p,_t,_roll in zip([0,   1,   2,   3,   4,   5],
                                    [0,   270, 180, 90,  0,   0],
                                    [180, 90,  180, 90,  90,  270],
                                    [270, 180, 270, 0, 180, 180]):

    print("#########\nRendering orientation %s\np=%s, t=%s, roll=%s\n##########"%\
            (_orientation,_p,_t,_roll))

    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10)) 
    
    if _orientation == 0:
        mask = np.random.rand(len(halog_pos)) < 1
        Pg = sph.Particles(halog_pos[mask], halog_mass[mask] * 1e10) 
        Pd = sph.Particles(halog_pos[mask], halog_dust[mask] * 1e10) 
        Psfr = sph.Particles(halog_pos[mask], halog_sfr[mask]) 
        
        # mask = np.random.rand(len(halod_pos)) < 1
        # Pd = sph.Particles(halod_pos[mask], np.ones(np.sum(mask))) 

        mask = np.random.rand(len(halos_pos)) < 1
        Ps = sph.Particles(halos_pos[mask], halos_pmass[mask] * 1e10) 
    
    sm1,img1 = plot_dist(fig, ax1, Pg, hcood, cmaps.night(), vmin=1e-1, extent=30, p=_p, t=_t, roll=_roll) 
    sm3,img3 = plot_dist(fig, ax3, Pd, hcood, cmaps.twilight(), vmin=1e-1, extent=30, p=_p, t=_t, roll=_roll) 
    sm2,img2 = plot_dist(fig, ax2, Ps, hcood, cmaps.desert(), vmin=1e6, extent=30, p=_p, t=_t, roll=_roll) 
    sm4,img4 = plot_dist(fig, ax4, Psfr, hcood, cmaps.sunlight(), vmin=0.1, extent=30, p=_p, t=_t, roll=_roll) 
    
    # sm4,img4 = plot_dist(fig, ax4, Pd, hcood, cmaps.sunlight(), vmin=0.1, extent=30, p=_p, roll=_roll) 
    
    for _str,_img in zip(['gas','stellar','dust','sfr'],
                    [img1,img2,img3,img4]):
        np.savetxt('arrays/m100_%s_g%s_o%s'%(_str,gidx,_orientation),_img)
    
    ## colorbars
    for ax,sm in zip([ax1,ax2,ax3,ax4],[sm1,sm2,sm3,sm4]):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
        # cax.xaxis.set_ticks_position('bottom')
        cbar.ax.set_ylabel('$\mathrm{M_{\odot} \; kpc^{-2}}$', size=12)
    
    
    # for ax in [ax1,ax2]:
    #     ax.set_xticklabels([])
    # for ax in [ax3,ax4]:
    #     ax.set_xticklabels([])
    # for ax in [ax2,ax3]:
    #     ax.set_yticklabels([])
    # ax4.yaxis.tick_right()
    
    
    s = 10
    ax1.text(0.75,0.85,'$\mathrm{Gas}$',size=s,transform=ax1.transAxes,c='white')
    ax2.text(0.65,0.85,'$\mathrm{Stellar}$',size=s,transform=ax2.transAxes,c='white')
    ax3.text(0.71,0.85,'$\mathrm{Dust}$',size=s,transform=ax3.transAxes,c='white')
    ax4.text(0.5,0.85,'$\mathrm{SFR}$',size=s,transform=ax4.transAxes,c='black')
    # ax4.text(0.5,0.85,'$\mathrm{Dark \; Matter}$',size=s,transform=ax4.transAxes,c='black')
    
    # for ax in [ax1,ax4,axm]: ax.set_ylabel('$\mathrm{kpc}$')
    ax4.set_ylabel('$\mathrm{kpc}$',rotation=270)
    ax4.yaxis.set_label_position('right')
    
    # plt.show() 
    plt.savefig('images/m100_physical_g%s_o%s_physical.png'%(gidx,_orientation), dpi=300, bbox_inches='tight')
    plt.close()

