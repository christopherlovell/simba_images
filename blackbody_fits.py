import sys 
sys.path.append('..') 

import numpy as np 
import json 
import h5py
import glob

from astropy import constants 
import astropy.units as u
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from hyperion.model import ModelOutput
import yt

import caesar 
from caesar.data_manager import DataManager

from simba import simba 
sb = simba() 

# def mbb(x,z,F,T,beta):
#     k = 1.381e-23; c = 2.998e8; h = 6.626e-34
#     x_ = x / (1+z)
#     X=1.e-6*x_ 
#     nu_rest = (c / X)
#     nu_c = (c/100.e-6)
#     tau = (nu_rest/nu_c)**beta
#     g = (1 - np.exp(-tau)) * (nu_rest**3) / (np.exp((h*nu_rest)/(k*T)) - 1)
#     g/=np.max(g)
#     return F*g


# def mbb(x,z,F,T,beta):
def mbb(x,z,T,beta,lambda_0=100.e-6):
    k = 1.381e-23; c = 2.998e8; h = 6.626e-34
    # x_ = x / (1+z)
    # X=1.e-6*x_ 
    X=1.e-6*x 
    nu_rest = (c / X)
    nu_c = (c/lambda_0)
    tau = (nu_rest/nu_c)**beta
    g = (1 - np.exp(-tau)) * (nu_rest**3) / (np.exp((h*nu_rest)/(k*T)) - 1)
    g/=np.max(g)
    # return F*g
    return g

def mbb_norm(x,z,T,beta,Nbb,lambda_0=100.e-6):
    return Nbb * mbb(x / (1+z),z,T,beta,lambda_0=lambda_0)

# def mbb_pl(x,z,F,T,beta,alpha,Nbb,Npl):
def mbb_pl(x,z,T,beta,alpha,Nbb):#,Npl):
    _x = x / (1+z)

    b1=26.68; b2=6.246; b3=1.905e-4; b4=7.243e-5
    lambda_c = ((b1+b2*alpha)**-2 + (b3+b4*alpha)*T)**-1

    #Npl = (Nbb * mbb(lambda_c,z,T,beta)) / lambda_c**3 # alpha
    #_c = 2.9e8 * 1e6
    Npl = (Nbb * mbb(lambda_c,z,T,beta)) / (lambda_c)**3 # alpha

    # _mbb = mbb(_x,z,F,T,beta)
    _mbb = mbb(_x,z,T,beta)

    # return (Nbb*_mbb) + (Npl * (_x)**alpha * np.exp(-1 * ((_x/lambda_c)**2)))
    return (Nbb*_mbb) + (Npl * (_x/lambda_c)**alpha * np.exp(-1 * ((_x/lambda_c)**2)))



snap = '078'
_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)
z = cs.simulation.redshift

_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]
galaxy_names = {3: 'Smiley', 8: 'Haydon', 51: 'Guillam', 54: 'Alleline',
                94: 'Esterhase', 100: 'Prideaux', 134: 'Bland', 139: 'Lacon'}

Tmbb = {}
Tpeak = {}
lambda_peak = {}
Lsol = {}
Lsol_spec = {}
    
dl = sb.cosmo.luminosity_distance(z).to(u.cm)


fig, ax = plt.subplots(1,1)
for _c,_g in enumerate(galaxies):
    gidx = _g.GroupID
    with h5py.File('sed_out_hires_test.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]; spec = f['%s/SED'%gidx][:]
        f850 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,lambda_pivot=850)
        
        f870= sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z, 
                      filt_wl=[865,866,870,874,875],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=870)
        # f1200 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z, 
        #               filt_wl=[1195,1196,1200,1204,1205],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=1200)
        f500 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z, 
                      filt_wl=[495,496,500,504,505],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=500)
        f350 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                      filt_wl=[345,346,350,354,355],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=350)
        f250 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                            filt_wl=[245,246,250,254,255],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=250)
        f70 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                            filt_wl=[65,66,70,74,75],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=70)
        f100 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                            filt_wl=[95,96,100,104,105],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=100)
        f160 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,z,
                            filt_wl=[155,156,160,164,165],filt_trans=[0.,1.,1.,1.,0.], lambda_pivot=160)

    Tmbb[gidx] = np.zeros(spec.shape[0])
    Lsol[gidx] = np.zeros(spec.shape[0])
    Lsol_spec[gidx] = np.zeros(spec.shape[0])

    f = interp1d(wav, spec, axis=1, kind='quadratic')
    _x = np.linspace(1e1,100,int(1e4))
    _spec_interp = f(_x)
    lambda_peak[gidx] = _x[np.argmax(_spec_interp,axis=1)] # micron
    Tpeak[gidx] = 2.9e3 / lambda_peak[gidx] # K

    # lambda_peak[gidx] = wav[np.argmax(spec,axis=1)] # micron
    # Tpeak[gidx] = 2.9e3 / lambda_peak[gidx] # K

    for i in np.arange(spec.shape[0]):
        _y = np.array([#f70[i].value, f100[i].value, f160[i].value, 
            f250[i].value, f350[i].value, f500[i].value, f850[i].value])#, f870[i].value])#, f1200[i].value])  # mJy
        wl = np.array([#70, 100, 160, 
            250, 350, 500, 850])#, 870])#, 1200]) 
        
        _beta = 2
        _lambda_0 = 80e-6
        popt, pcov = curve_fit(lambda x, T, F: mbb_norm(x,z,T,_beta,F,_lambda_0), wl, _y, p0=[40,40])
        # popt, pcov = curve_fit(lambda x, T, _alpha, Nbb: mbb_pl(x,z,T,_beta,_alpha,Nbb), 
                               # wl, _y, p0=[40,1,1],maxfev=int(1e4))
        
        Tmbb[gidx][i] = popt[0]
        # y = mbb_pl(x,z,popt[0],beta=_beta,alpha=popt[1],Nbb=popt[2]) * u.mJy    
        y = mbb_norm(x,z,popt[0],_beta,popt[1],_lambda_0) * u.mJy    
        y = y.to(u.erg  / (u.s * u.cm**2 * u.Hz))
        x = np.logspace(-1,4,int(1e3)); _nu = (constants.c / (x * u.micron)).to(u.Hz)
        _out = np.trapz(y[::-1],_nu[::-1]) * (4 * np.pi * dl**2)
        Lsol[gidx][i] = _out.to(u.solLum).value

        _mask = (wav>1e1) & (wav<1e3)
        _nu = (constants.c / (wav[_mask] * u.micron)).to(u.Hz)
        _spec = (spec[i,_mask] * (u.erg / u.s)) / (4 * np.pi * dl**2) / _nu

        _out = np.trapz(_spec,_nu) * (4 * np.pi * dl**2)
        Lsol_spec[gidx][i] = _out.to(u.solLum).value

        if (i == 0) & (_c == 0):
            ax.scatter(wl,_y,marker='o')
            x = np.logspace(0,3,int(1e3))
            # ax.plot(x, mbb_pl(x,z,popt[0],_beta,popt[1],popt[2]), color='C%i'%_c)
            ax.plot(x, mbb_norm(x,z,popt[0],_beta,popt[1],_lambda_0), color='C%i'%_c)


ax.set_xlim(10,1000)#; ax.set_ylim(0,50)
plt.show()        


run = '/blue/narayanan/c.lovell/simba/m100n1024/run_sed/snap_078_hires_orthogonal'

Tbins = np.linspace(32,80,35)
Lbins = np.linspace(11,13.8,55)

fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(2,4,figsize=(12,6))
plt.subplots_adjust(hspace=0.3, wspace=0.1)

_percsT = {}
_percsT_peak = {}
_percsL = {}
for i,(_g,axA,axB) in enumerate(zip(galaxies, 
                                [ax1,ax1,ax2,ax2,ax3,ax3,ax4,ax4],
                                [ax5,ax5,ax6,ax6,ax7,ax7,ax8,ax8])):

    axA.hist(Tmbb[_g.GroupID], bins=Tbins, alpha=0.5, color='C%i'%i, 
             label=galaxy_names[_g.GroupID])

    # ## plot true mass-weighted temperature
    # fname = glob.glob('%s/gal_%i/snap078.galaxy*.rtout.sed'%(run,_g.GroupID))[0]

    # m = ModelOutput(fname)
    # _pf = m.get_quantities().to_yt()
    # ad = _pf.all_data()
    # _T = ad.quantities.weighted_average_quantity("temperature", "cell_mass")
    # axA.vlines(_T,0,25, linestyle='dashed', color='C%i'%i)

    axB.hist(np.log10(Lsol[_g.GroupID]), bins=Lbins, alpha=0.5, color='C%i'%i)
    
    _percsT[_g.GroupID] = np.percentile(Tmbb[_g.GroupID], q=[16,50,84])
    _percsT_peak[_g.GroupID] = np.percentile(Tpeak[_g.GroupID], q=[16,50,84])

    _percsL[_g.GroupID] = np.percentile(np.log10(Lsol[_g.GroupID]), q=[16,50,84])
    print('$%.2f_{-%.2f}^{+%.2f}$ & $%.2f_{-%.2f}^{+%.2f}$ & $%.2f_{-%.2f}^{+%.2f}$'%\
            (_percsT[_g.GroupID][1], _percsT[_g.GroupID][1] - _percsT[_g.GroupID][0], 
             _percsT[_g.GroupID][2] - _percsT[_g.GroupID][1],
             _percsT_peak[_g.GroupID][1], _percsT_peak[_g.GroupID][1] - _percsT_peak[_g.GroupID][0], 
             _percsT_peak[_g.GroupID][2] - _percsT_peak[_g.GroupID][1],
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

plt.show()
# plt.savefig('plots/temp_lum_distribution.png', dpi=300, bbox_inches='tight') 



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




for i,_g in enumerate(galaxies):
    gidx = _g.GroupID
    plt.errorbar(_g.sfr, np.median(Lsol[gidx]), yerr=np.std(Lsol[gidx]) ,
                color='C%i'%i, label=galaxy_names[gidx], marker='o')


plt.legend(loc='lower center')
plt.xlabel('SFR / Msol yr^-1')
plt.ylabel('L_IR / Lsol')
plt.show()


## ---- dust temp against luminosity
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5,8))
plt.subplots_adjust(hspace=0)

for i,_g in enumerate(galaxies):
    gidx = _g.GroupID
    ax1.scatter(np.log10(Lsol[gidx]), Tmbb[gidx],# yerr=np.std(Lsol[gidx]) ,
                color='C%i'%i, label=galaxy_names[gidx], s=5)
    ax2.scatter(np.log10(Lsol[gidx]), Tpeak[gidx],# yerr=np.std(Lsol[gidx]) ,
                color='C%i'%i, label=galaxy_names[gidx], s=5)#, marker='*')

    # fname = glob.glob('%s/gal_%i/snap078.galaxy*.rtout.sed'%(run,_g.GroupID))[0]
    # m = ModelOutput(fname)
    # _pf = m.get_quantities().to_yt()
    # ad = _pf.all_data()
    # _T = ad.quantities.weighted_average_quantity("temperature", "cell_mass")
    # ax.scatter(np.log10(np.median(Lsol[gidx])),_T, color='C%i'%i,
    #            marker='o', edgecolors='black', s=30, lw=1)

_x = np.linspace(12,13,100)
ax1.plot(_x,5.57 * (10**_x)**0.0638, color='black')

ax1.set_xticklabels([])
ax1.legend(ncol=1, prop={'size': 9})#loc='lower center')
ax2.set_xlabel('$L_{\mathrm{IR}} \,/\, \mathrm{L_{\odot}}$')
ax1.set_ylabel('$T_{\mathrm{MBB}}$')
ax2.set_ylabel('$T_{\mathrm{peak}}$')

def peak2lambda(x):
    return 2.9e3 / x

def lambda2peak(x):
    return 2.9e3 / x

secax = ax2.secondary_yaxis('right', functions=(peak2lambda, lambda2peak))
secax.set_ylabel('$\lambda_{\mathrm{peak}}$')

for ax in [ax1,ax2]: 
    ax.set_ylim(43,68)
    ax.set_xlim(12.55,13.15)
    ax.grid(alpha=0.4)

plt.show()
# plt.savefig('plots/temp_lum_scatter.pdf', dpi=300, bbox_inches='tight'); plt.close()

