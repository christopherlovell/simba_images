import sys
sys.path.append('..')

import numpy as np
import h5py
import json

import matplotlib
matplotlib.use('agg')
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


g_idx = int(sys.argv[1])


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
_dat = json.load(open('m100/galaxy_selection.json','r'))



for gidx in ['3','8','51','54','94','100','134','139']:
# gidx = np.array(['3','8','51','54','94','100','134','139'])[g_idx]
# if (True):
    print("gidx:",gidx)

    cs = caesar.load('%sm100n1024_%s.hdf5'%(sb.cs_directory,snap))
    _mstar = np.log10(cs.galaxies[int(gidx)].masses['stellar'])

    print("Mstar:",_mstar)

    hcood =  np.array(_dat[snap][gidx]['pos'])
    hidx =  _dat[snap][gidx]['hidx']
    
    fdir = '/blue/narayanan/c.lovell/simba/m100n1024/out/snap_078/subset_%05d.h5'%hidx
    # fdir ='/cosma7/data/dp104/dc-dave2/sim/m100n1024/s50j7k/snap_m100n1024_%s.hdf5'%snap
    
    with h5py.File(fdir,'r') as f:
        _a = np.float32(1. / (1+f['Header'].attrs['Redshift']))
        _h = np.float32(sb.cosmo.h)
        _temp = (1. / _h) * _a
        print(_a,_h,_temp)
             
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
    
    for _orientation,_p,_t,_roll in zip([0,   1,   2,   3,   4,   5],
                                        beta, alpha, [270,180,270,0,270,90]):

        print("#########\nRendering orientation %s\np=%s, t=%s, roll=%s\n##########"%\
                (_orientation,_p,_t,_roll))
    
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10)) 
        
        if _orientation == 0:
            mask = np.random.rand(len(halog_pos)) < 1
            Pg = sph.Particles(halog_pos[mask], halog_mass[mask] * 1e10) 
            Pd = sph.Particles(halog_pos[mask], halog_dust[mask] * 1e10) 
            Psfr = sph.Particles(halog_pos[mask], halog_sfr[mask]) 
    
            mask = np.random.rand(len(halos_pos)) < 1
            Ps = sph.Particles(halos_pos[mask], halos_pmass[mask] * 1e10) 


        extent = 30
        sm1,img1 = plot_dist(fig, ax1, Pg, hcood, cmaps.night(), 
                             vmin=1e2, extent=extent, p=_p, t=_t, roll=_roll) 
        sm3,img3 = plot_dist(fig, ax3, Pd, hcood, cmaps.twilight(), 
                             vmin=1e-1, extent=extent, p=_p, t=_t, roll=_roll) 
        sm2,img2 = plot_dist(fig, ax2, Ps, hcood, cmaps.desert(), 
                             vmin=1e6, extent=extent, p=_p, t=_t, roll=_roll) 
        sm4,img4 = plot_dist(fig, ax4, Psfr, hcood, cmaps.sunlight(), 
                             vmin=0.1, extent=extent, p=_p, t=_t, roll=_roll) 
        
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
        plt.savefig('images/m100_physical_g%s_o%s_physical.png'%(gidx,_orientation), 
                    dpi=300, bbox_inches='tight')

        plt.close()
    
