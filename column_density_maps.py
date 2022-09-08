import numpy as np
import json
import h5py

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.units as u
import caesar
from scipy.spatial.distance import cosine

import sphviewer as sph
from sphviewer.tools import cmaps
cmap = cmaps.night()

from simba import simba
sb = simba()


## spherical coordinates of viewing angle in box coordinates (not pd)
# phi   = np.array([0,90,180,270,0,0]) * (np.pi/180.)
# theta = np.array([90,90,90,90,0,180]) * (np.pi/180.)
np.random.seed(0); _N = 50
theta = np.arccos(1 - 2 * np.random.rand(_N)) #* (180 / np.pi)
phi   = 2 * np.pi * np.random.rand(_N) #* (180 / np.pi)

coods = np.zeros((len(theta),3))
coods[:,0] = np.sin(theta) * np.cos(phi)
coods[:,1] = np.sin(theta) * np.sin(phi)
coods[:,2] = np.cos(theta)

# beta = np.arcsin(-1 * z)
# alpha = np.arcsin(y / np.cos(beta))
# beta = np.arccos(z)
# alpha = np.arccos(x / np.sin(beta))
beta = np.arcsin(-1 * coods[:,0])
alpha = np.arcsin(coods[:,1] / np.cos(beta))
beta = 180 * (beta / np.pi)
alpha = 180 * (alpha / np.pi)


def tau_dist(P,coods, cmap, extent=300, p=0, t=0, roll=0):

    C = sph.Camera(x=coods[0], y=coods[1], z=coods[2],
                   r='infinity', zoom=1,
                   t=t, p=p, roll=roll,
                   extent=[-extent,extent,-extent,extent],
                   xsize=512, ysize=512)

    S = sph.Scene(P, Camera=C)
    R = sph.Render(S)
    return R.get_image()


def recalculate_ang_mom_vector(_g,ptype=4):
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

    return np.sum(np.cross(_coods[mask], (_vels[mask] * _masses[mask,None])),axis=0)


snap = '078'
cs = caesar.load('%sm100n1024_%s.hdf5'%(sb.cs_directory,snap))
redshift = cs.simulation.redshift
_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]

gidx_strs = np.array(['3','8','51','54','94','100','134','139'])
_plot = False
_area = {gidx: np.zeros(len(alpha)) for gidx in _dat['078'].keys()}
_cos_dist = {gidx: np.zeros(len(alpha)) for gidx in _dat['078'].keys()}
_f = {gidx: np.zeros(len(alpha)) for gidx in _dat['078'].keys()}
_tau_save = {gidx: np.zeros(len(alpha)) for gidx in _dat['078'].keys()}
for j,gidx in enumerate(gidx_strs):
    # gidx = '3'; j = 0 
    _g = galaxies[j]
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
   

    _L = recalculate_ang_mom_vector(_g,ptype=0)
    _cos_dist[gidx] = [round(1 - cosine(_c,_L),3) for _c in coods]
    
    Pd = sph.Particles(halog_pos, halog_dust * 1e10)

    for k,(_p,_t,_cosine) in enumerate(zip(beta, alpha, _cos_dist[gidx])):
        print("Cosine distance:",_cosine)
        extent = 19.5

        img = tau_dist(Pd, hcood, cmaps.twilight(), extent=extent, p=_p, t=_t) # Units: Msol / Kpc^2

        img *= u.M_sun / u.kpc**2
        img = img.to(u.g / u.cm**2)
        
        _beta = 2; _lambda = 70
        kappa = (0.05 * (_lambda/870)**(-1*_beta) * u.m**2 / u.kg).to(u.cm**2 / u.g)
        tau = (img * kappa).value
        
        _area[gidx][k] = np.sum(tau > 1e-1) * (extent / 512)**2  # kpc^2

        if _plot:
            cmap = plt.get_cmap('viridis')
            vmin = 1e-2; vmax = tau.max()
        
            fig, (ax1, ax2) = plt.subplots(2,1)
        
            cNorm  = colors.LogNorm(vmin=vmin,vmax=vmax)
            sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)
            ax1.imshow(tau, cmap=cmap, norm=cNorm)
            tau[tau < 1] = 0.
            ax2.imshow(tau, cmap=cmap, norm=cNorm)
        
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
            cbar.ax.set_ylabel('$\\tau_{\lambda}$', size=12)
            plt.show()
    
    _tau_save[gidx] = tau

    with h5py.File('sed_out_hires_test.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        _f[gidx] = sb.calc_mags(wav[::-1]*u.micron,spec[:,::-1]*u.erg/u.s,redshift, 
                         filt_wl=np.array([_lambda-5,_lambda-4,_lambda,_lambda+4,_lambda+5]),
                         filt_trans=np.array([0.,1.,1.,1.,0.]))
        # spec = sb.luminosity_to_flux_density(wav,spec,redshift)
   

fig,axes = plt.subplots(8,5,figsize=(10,17))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

cmap = plt.get_cmap('viridis')
vmin = 1e-2; vmax = tau.max()
cNorm  = colors.LogNorm(vmin=vmin,vmax=vmax)
sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

for _axes,gidx in zip(axes,gidx_strs):
    ax1,ax2,ax3,ax4,ax5 = _axes
    ax1.scatter(_area[gidx], _f[gidx], s=2)
    ax2.scatter(_area[gidx], np.abs(_cos_dist[gidx]), s=2)
    ax3.scatter(_f[gidx], np.abs(_cos_dist[gidx]), s=2)

    for _ax in [ax1,ax2]: _ax.set_xlabel('$\mathrm{Area} (\\tau > 1) \; [\mathrm{kpc^2}]$')
    ax1.set_ylabel('S')
    for _ax in [ax2,ax3]: _ax.set_ylabel('C')
    ax3.set_xlabel('S')

    ax5.imshow(_tau_save[gidx], cmap=cmap, norm=cNorm)
    _temp = _tau_save[gidx].copy(); _temp[_temp < 1] = 0.
    ax4.imshow(_temp, cmap=cmap, norm=cNorm)
    ax4.text(0.,0.1,'$\\tau > 1$',transform=ax4.transAxes)
    for _ax in [ax4,ax5]: _ax.set_xticklabels([]); _ax.set_yticklabels([])

    divider = make_axes_locatable(ax5)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel('$\\tau_{\lambda}$', size=12)

plt.show()
# plt.savefig('plots/column_density_maps_%smicron.pdf'%_lambda,dpi=300,bbox_inches='tight'); plt.close()



