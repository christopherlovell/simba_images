import sys 
sys.path.append('..') 

import numpy as np 
import json 
import h5py

from astropy import constants 
import astropy.units as u
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from hyperion.model import ModelOutput

import caesar 
from caesar.data_manager import DataManager

from simba import simba 
sb = simba() 


def mbb(x,z,F,T,beta):
    k = 1.381e-23; c = 2.998e8; h = 6.626e-34
    x_ = x / (1+z)
    X=1.e-6*x_ 
    nu_rest = (c / X)
    nu_c = (c/100.e-6)
    tau = (nu_rest/nu_c)**beta
    g = (1 - np.exp(-tau)) * (nu_rest**3) / (np.exp((h*nu_rest)/(k*T)) - 1)
    g/=np.max(g)
    return F*g



snap = '078'
rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run'
#rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run_sed'

_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)
z = cs.simulation.redshift

_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]
galaxy_names = {3: 'Smiley', 8: 'Haydon', 51: 'Guillam', 54: 'Alleline',
                94: 'Esterhase', 100: 'Prideaux', 134: 'Bland', 139: 'Lacon'}

Tmbb = {}
Lsol = {}
    
dl = sb.cosmo.luminosity_distance(z).to(u.cm)
x = np.logspace(-1,4,int(1e3))
_nu = (constants.c / (x * u.micron)).to(u.Hz)

for _g in galaxies:
    gidx = _g.GroupID
    with h5py.File('sed_out_hires.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        f850 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,lambda_pivot=850)
        
        f500 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z, 
                      filt_wl=[495,496,500,504,505],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=500)
        f350 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                      filt_wl=[345,346,350,354,355],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=350)
        f250 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                            filt_wl=[245,246,250,254,255],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=250)
        f100 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                            filt_wl=[95,96,100,104,105],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=100)
        f80 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                            filt_wl=[75,76,80,84,85],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=80)
        f50 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                            filt_wl=[45,46,50,54,55],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=50)


    Tmbb[gidx] = np.zeros(spec.shape[0])
    Lsol[gidx] = np.zeros(spec.shape[0])

    for i in np.arange(spec.shape[0]):
        y = np.array([# f50[i].value, f80[i].value, f100[i].value, 
                      f250[i].value, f350[i].value, f500[i].value, f850[i].value])  # mJy
        
        wl = np.array([#50, 80, 100, 
                       250,350,500,850])

        _beta = 2
        popt, pcov = curve_fit(lambda x, F, T: mbb(x,z,F,T,_beta), wl, y, p0=[1e4,40])
        Tmbb[gidx][i] = popt[1]

        y = mbb(x,z,popt[0],popt[1],beta=_beta) * u.mJy
        y = y.to(u.erg  / (u.s * u.cm**2 * u.Hz))
        _out = np.trapz(y[::-1],_nu[::-1]) * (4 * np.pi * dl**2)
        Lsol[gidx][i] = _out.to(u.solLum).value
        

        


Tbins = np.linspace(32,80,35)
Lbins = np.linspace(11,13.8,55)

fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(2,4,figsize=(12,6))
plt.subplots_adjust(hspace=0.3, wspace=0.1)

_percsT = {}
_percsL = {}
for i,(_g,axA,axB) in enumerate(zip(galaxies, 
                                [ax1,ax1,ax2,ax2,ax3,ax3,ax4,ax4],
                                [ax5,ax5,ax6,ax6,ax7,ax7,ax8,ax8])):

    axA.hist(Tmbb[_g.GroupID], bins=Tbins, alpha=0.5, color='C%i'%i, 
             label=galaxy_names[_g.GroupID])



    axB.hist(np.log10(Lsol[_g.GroupID]), bins=Lbins, alpha=0.5, color='C%i'%i)
    
    _percsT[_g.GroupID] = np.percentile(Tmbb[_g.GroupID], q=[16,50,84])

    _percsL[_g.GroupID] = np.percentile(np.log10(Lsol[_g.GroupID]), q=[16,50,84])
    print('$%.2f_{-%.2f}^{+%.2f}$ & $%.2f_{-%.2f}^{+%.2f}$'%\
            (_percsT[_g.GroupID][1], _percsT[_g.GroupID][1] - _percsT[_g.GroupID][0], 
             _percsT[_g.GroupID][2] - _percsT[_g.GroupID][1],
             _percsL[_g.GroupID][1], _percsL[_g.GroupID][1] - _percsL[_g.GroupID][0], 
             _percsL[_g.GroupID][2] - _percsL[_g.GroupID][1]))
   

## inter-percentile range
print(np.median([_percsT[_g.GroupID][2] - _percsT[_g.GroupID][0] for _g in galaxies]))
print(np.median([_percsL[_g.GroupID][2] - _percsL[_g.GroupID][0] for _g in galaxies]))
    
for ax in [ax1,ax2,ax3,ax4]: 
    ax.set_xlim(41,69)
    ax.set_ylim(0,25)
    ax.set_xlabel('$T_{\mathrm{MBB}}$')
    ax.legend(frameon=False)

for ax in [ax5,ax6,ax7,ax8]: 
    ax.set_xlim(12.45,13.15)
    ax.set_ylim(0,31)
    ax.set_xlabel('$L_{\mathrm{IR}} \,/\, \mathrm{L_{\odot}}$')

for ax in [ax1,ax5]: ax.set_ylabel('$N$')
for ax in [ax2,ax3,ax4]: ax.set_yticklabels([])
for ax in [ax6,ax7,ax8]: ax.set_yticklabels([])

# plt.show()
plt.savefig('plots/temp_lum_distribution.png', dpi=300, bbox_inches='tight') 



# i = 0
# _flux = sb.luminosity_to_flux_density(wav,spec[i],z).value 
# 
# y = np.array([f50[i].value, f80[i].value, f100[i].value, 
#               f250[i].value,f350[i].value,
#               f500[i].value,f850[i].value])  # mJy
# 
# wl = np.array([50, 80, 100,
#                250,350,500,850])
# 
# _beta = 1.5
# popt, pcov = curve_fit(lambda x, F, T: mbb(x,z,F,T,_beta), wl, y, p0=[1e4,40]) 
# 
# x = np.logspace(-1,4,int(1e3))
# plt.plot(np.log10(x), np.log10(mbb(x,z,popt[0],popt[1],beta=_beta)))
# 
# plt.scatter(np.log10(wl), np.log10(y))
# plt.plot(np.log10(wav*(1+z)), np.log10(_flux))
# 
# plt.xlim(0.9,3.3)
# plt.ylim(-1,4)
# plt.show()


