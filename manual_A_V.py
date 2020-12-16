import json
import numpy as np
import h5py

import matplotlib as mpl
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
    
from scipy.spatial.distance import cdist

import sphviewer as sph
from sphviewer.tools import cmaps

import yt
import caesar
from caesar.pyloser import pyloser
from caesar.utils import rotator
from caesar.cyloser import compute_AV, init_kerntab, _star_AV
from caesar.data_manager import DataManager
from caesar.utils import rotator


def transform_coods(theta,phi,tol=20,test=False):
    alpha = np.arcsin(-1 * np.round(np.sin(theta),tol) * np.round(np.sin( phi ),tol))
    beta = np.round(np.arcsin((np.round(np.cos(phi),tol) * np.round(np.sin(theta),tol)) / np.round(np.cos(alpha),tol)),tol)

    beta[np.isnan(beta)] = 0.
    mask = (theta > np.pi/2)
    beta[mask] = np.arccos(     np.round(np.cos(theta[mask]),tol) / np.round(np.cos(alpha[mask]),tol))
    mask = (phi > np.pi/2) & (phi < 3*np.pi/2) & (theta > np.pi/2)
    beta[mask] = -beta[mask]

    if test:
        x = np.round(np.sin(theta),tol) * np.round(np.cos(phi),tol)
        y = np.round(np.sin(theta),tol) * np.round(np.sin(phi),tol)
        z = np.round(np.cos(theta),tol)
        out = np.array([x,y,z]).T
        
        c  = np.round(np.cos(alpha),tol); s  = np.round(np.sin(alpha),tol)
        Rx = np.array([[1.0,0.0,0.0],[0.0,  c, -s],[0.0,  s,  c]])
 
        c  = np.round(np.cos(beta),tol); s  = np.round(np.sin(beta),tol)
        Ry = np.array([[  c,0.0, s],[0.0,1.0,0.0],[ -s,0.0,  c]])
        V = np.array([0,0,1])#.T
 
        out2 = np.round(np.vstack(np.dot(np.dot(Ry,Rx),V)).T,tol)
        print(np.round(out,2),"\n\n",np.round(out2,2),"\n")

        match = np.round(out,5) == np.round(out2,5)
        print(np.sum(match),np.product(match.shape))

    return np.round(alpha,tol),np.round(beta,tol)


_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)

_dat = json.load(open('m100/galaxy_selection.json','r'))
halos = [cs.halos[n['hidx']] for k,n in _dat['078'].items()]
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]

subset_files = ["subset_%05d.h5"%h.GroupID for h in halos] 
subset_dir='/blue/narayanan/c.lovell/simba/m100n1024/out/snap_078/'

A_V = {s: None for s in np.arange(len(subset_files))}



## Plot the particle distributions to check orientations ##
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax1b,ax2b,ax3b),(ax4b,ax5b,ax6b)) = \
        plt.subplots(4,3,figsize=(10,10))
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(10,5))
axes = [ax1,ax2,ax3,ax4,ax5,ax6]

cmap = cmaps.night()
roll,extent=0,30
## ---------- ##


for s,s_file in enumerate(subset_files):
# if True:
#     s = 0
#     s_file = subset_files[0]
    ds = yt.load(subset_dir+s_file)

    halo = halos[s]
    galaxy = galaxies[s]
    phot = pyloser.photometry(cs, halo)
    phot.ssp_table_file = '/home/c.lovell/codes/caesar/FSPS_Chab_EL.hdf5'
    init_kerntab(phot)
    hcood = galaxy.pos.value

    with h5py.File(subset_dir+s_file,'r') as f:
        _fact = 1 / cs.simulation.hubble_constant
        gpos = f['PartType0']['Coordinates'][:] * _fact
        spos = f['PartType4']['Coordinates'][:] * _fact
        gm = f['PartType0']['Masses'][:] * 1e10
        gZ = f['PartType0']['Dust_Masses'][:] * 1e10
        gZ /= gm
        ghsm = f['PartType0']['SmoothingLength'][:] 
    
    
    _pos = galaxy.pos.to('kpccm').value
    _mask = (cdist(spos, [_pos]) < 30).flatten()
    spos = spos[_mask,:]
    _mask = (cdist(gpos, [_pos]) < 30).flatten()
    gpos = gpos[_mask]
    spos = spos.astype(np.float64)
    gpos = gpos.astype(np.float64)
    
    idir = 2; igstart = 0; igend = len(gpos)
    Lbox = float(phot.boxsize.value)
    nkerntab = len(phot.kerntab); kerntab = phot.kerntab
    redshift = phot.obj.simulation.redshift
    dtm_MW = 0.4/0.6
    NHcol_fact = 1.99e33*0.76*(1.+redshift)*(1.+redshift)/(3.086e21**2*1.673e-24)
    AV_fact = 1./(2.2e21*0.0189) # Watson 2011 arXiv:1107.6031 (note: Watson calibrates to Zsol=0.0189)
    usedust = phot.use_dust

    # theta = np.array([90,90, 90, 90,0,180]) * np.pi/180.
    # phi = np.array([0, 90,180,270,0,  0]) * np.pi/180.
    np.random.seed(0); _N = 50
    theta = np.arccos(1 - 2 * np.random.rand(_N)) #* (180 / np.pi)
    phi   = 2 * np.pi * np.random.rand(_N) #* (180 / np.pi)
    alpha,beta = transform_coods(theta,phi,tol=15)

    A_V[s] = {i: np.zeros(len(spos)) for i in np.arange(len(alpha))}

    # for i,(_a,_b,ax) in enumerate(zip(alpha,beta,axes)):
    for i,(_a,_b) in enumerate(zip(alpha,beta)):
    
        if (_a != 0.) | (_b != 0.):
            print("Rotating (idx: %d | alpha=%f, beta=%f)"%(i,_a,_b))
            _spos = rotator(spos.copy(), _a, _b)
            _gpos = rotator(gpos.copy(), _a, _b)
            _hcood = rotator(hcood.copy(), _a,_b)
        else:
            _spos = spos.copy(); _gpos = gpos.copy(); _hcood = hcood.copy() 
   
#         if s==0:
#             mask = np.random.rand(len(_gpos)) < 0.5
#             ax.scatter(_gpos[mask,0],_gpos[mask,1],s=1,alpha=0.01,label=i)
#             ax.scatter(_hcood[0],_hcood[1],s=5)
#             ax.legend(); ax.set_aspect('equal')


        for ip in np.arange(len(spos)):
            A_V[s][i][ip] = _star_AV(ip, idir, igstart, igend, _spos, _gpos, 
                                  gm, gZ, ghsm, Lbox, nkerntab, kerntab, redshift, 
                                  dtm_MW,  NHcol_fact, AV_fact, usedust)


# plt.show()

# 
# for i,(_a,_b,t,p,ax,axb) in enumerate(zip(alpha,beta,theta,phi,axes,axesb)):
# 
#     if (_a != 0.) | (_b != 0.):
#         print("Rotating (idx: %d | alpha=%f, beta=%f)"%(i,_a,_b))
#         _spos = rotator(spos.copy(), _a, _b)
#         _gpos = rotator(gpos.copy(), _a, _b)

#     else:
#         _spos = spos.copy(); _gpos = gpos.copy(); _hcood = _hcood.copy()
# 
#     mask = np.random.rand(len(_gpos)) < 0.5
#     ax.scatter(_gpos[mask,0],_gpos[mask,1],s=1,alpha=0.01,label=i)
#     ax.legend(); ax.set_aspect('equal')
# 
#     P = sph.Particles(gpos,gm) 
#     C = sph.Camera(x=hcood[0], y=hcood[1], z=hcood[2], r='infinity', zoom=1,
#                    t=_a*(180./np.pi), p=_b*(180./np.pi), roll=roll,
#                    extent=[-extent,extent,-extent,extent],
#                    xsize=512, ysize=512)
#     S = sph.Scene(P, Camera=C)
#     R = sph.Render(S)
#     img = R.get_image()
#     vmax = img.max(); vmin = 1e2 
#     cNorm  = colors.LogNorm(vmin=vmin,vmax=vmax)
#     sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)
#     axb.imshow(img, extent=[-extent,extent,-extent,extent],
#               cmap=cmap, norm=cNorm)
# 
#     for ip in np.arange(len(spos)):
#         A_V[i][ip] = _star_AV(ip, idir, igstart, igend, _spos, _gpos, 
#                               gm, gZ, ghsm, Lbox, nkerntab, kerntab, redshift, 
#                               dtm_MW,  NHcol_fact, AV_fact, usedust)
# 
# 
#     # print(A_V[i][:5],"\n",spos[:2],"\n",_gpos[:2],"\n")
# 
# plt.show()


# bins = np.linspace(-5,3,20)
# plt.hist(np.log10(A_V[0]), bins=bins, cumulative=True, histtype='step', color='C0', label='C0')
# plt.hist(np.log10(A_V[2]), bins=bins, cumulative=True, histtype='step', linestyle='dashed', color='C0')
# 
# plt.hist(np.log10(A_V[1]), bins=bins, cumulative=True, histtype='step', color='C1', label='C1')#linestyle='dotted', 
# plt.hist(np.log10(A_V[3]), bins=bins, cumulative=True, histtype='step', linestyle='dashed', color='C1')
# 
# plt.hist(np.log10(A_V[4]), bins=bins, cumulative=True, histtype='step', color='C2', label='C2')#linestyle='dotted', 
# plt.hist(np.log10(A_V[5]), bins=bins, cumulative=True, histtype='step', linestyle='dashed', color='C2')
# 
# plt.legend()
# plt.show()



_hidx = 0
lenAV = len(A_V[_hidx][0])
[plt.plot(np.log10(np.sort(a_v)),
          np.cumsum(np.ones(len(a_v)))/lenAV,label=k) for k,a_v in A_V[_hidx].items()]

# plt.plot(np.log10(np.sort(A_V[_hidx][0])),
#          np.cumsum(np.ones(len(A_V[_hidx][0])))/lenAV, color='C0', label='0')
# plt.plot(np.log10(np.sort(A_V[_hidx][2])),
#          np.cumsum(np.ones(len(A_V[_hidx][0])))/lenAV, color='C0', linestyle='dashed', label='2')
# 
# plt.plot(np.log10(np.sort(A_V[_hidx][1])),
#          np.cumsum(np.ones(len(A_V[_hidx][0])))/lenAV, color='C1', label='1')
# plt.plot(np.log10(np.sort(A_V[_hidx][3])),
#          np.cumsum(np.ones(len(A_V[_hidx][0])))/lenAV, color='C1', linestyle='dashed', label='3')
# 
# plt.plot(np.log10(np.sort(A_V[_hidx][4])),
#          np.cumsum(np.ones(len(A_V[_hidx][0])))/lenAV, color='C2', label='4')
# plt.plot(np.log10(np.sort(A_V[_hidx][5])),
#          np.cumsum(np.ones(len(A_V[_hidx][0])))/lenAV, color='C2', linestyle='dashed', label='5')

plt.legend()
plt.xlim(-6,1)
plt.ylim(0,1)
plt.xlabel('$A_V$')
plt.ylabel('$f$')
# plt.savefig('A_V_gal%i.png'%_hidx,dpi=300)
plt.show()


_g = _hidx
gidx = galaxies[_g].GroupID 
with h5py.File('sed_out.h5','r') as f:
    wav = f['%s/Wavelength'%gidx][:]
    spec = f['%s/SED'%gidx][:]
    # flux_850 = f['%s/850 flux'%gidx][:]


# spec = spec[:10]
_p = 90
z = redshift

a_v = np.array([np.percentile(A_V[_g][i],_p) for i in np.arange(len(A_V[_g]))])
norm = mpl.colors.LogNorm(vmin=a_v.min(), vmax=a_v.max())
m = cm.ScalarMappable(norm=norm, cmap=cm.copper)

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(5,15))

for ax in [ax1,ax2]:
    [ax.plot(np.log10(wav * (1+z)), s, alpha=1,
         c=m.to_rgba(np.percentile(A_V[_g][i],_p))) 
        for i,s in enumerate(spec)]

mean_spec = np.mean(spec,axis=0)
[ax3.plot(wav * (1+z), s/mean_spec, alpha=1,
        c=m.to_rgba(np.percentile(A_V[_g][i],_p))) for i,s in enumerate(spec)]

for ax in [ax1,ax2]:
    ax.set_ylim(0,)
    ax.set_xlabel('$\mathrm{log_{10}}(\lambda \,/\, \AA)$', size=15)
    ax.set_ylabel('$\mathrm{erg \,/\, s}$', size=15)

for ax in [ax2,ax3]:
    ax.set_xlim(2,3)

# mean_flux = np.round(np.mean(flux_850),2)
ax1.text(0.2,0.9,'$z = %.2f$'%z,size=13,transform=ax1.transAxes)
# ax1.text(0.2,0.8,'$S_{850} = %s$'%mean_flux,size=13,transform=ax1.transAxes)
ax3.set_xlabel('$\lambda \,/\, \mu m$',size=15)
ax3.set_ylabel('$\mathrm{Flux}_i / \mathrm{Flux_{mean}}$',size=15)
ax3.set_ylim(0.7,1.3)
ax3.set_xlim(250,1000)

cax = fig.add_axes([0.14, 0.7, 0.04, 0.15])
cbar = fig.colorbar(m, aspect=10, orientation='vertical',
                    cax=cax, label='$A_V$')

plt.show()
# plt.savefig(f'plots/A_V_sed_g{gidx}.png',dpi=300,bbox_inches='tight')
plt.close()

