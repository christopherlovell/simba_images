import sys
sys.path.append('..')

import numpy as np
import json
import h5py
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import caesar
from caesar.data_manager import DataManager

from simba import simba
sb = simba()


snap = '078'
rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run'

_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)
z = cs.simulation.redshift
_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]

binlims = np.linspace(0.5,1,30)
gindexes = [3,8,51,54,94,100,134,139]
a_array = np.zeros((4,3))
N_array = np.zeros((4,len(binlims)-1))
wavelengths = [250,350,500,850]

for j,_lambda in enumerate(wavelengths):
    N = np.zeros(len(binlims)-1, dtype=int)

    for i,gidx in enumerate(gindexes):
        _g = cs.galaxies[gidx]
    
        gidx = galaxies[i].GroupID
        with h5py.File('sed_out_hires.h5','r') as f:
            wav = f['%s/Wavelength'%gidx][:]
            spec = f['%s/SED'%gidx][:]
            spec = sb.luminosity_to_flux_density(wav,spec,z)
    
        sidx = np.argmin(np.abs(wav*(1+z) - _lambda))
        
    
        _n,_dummy = np.histogram(spec[:,sidx].value/spec[:,sidx].max().value, bins=binlims)
        N += _n
     
        # plt.hist(spec[:,sidx].value/spec[:,sidx].max().value,histtype='step')
        # print(np.std(spec,axis=0)[sidx] / np.mean(spec,axis=0)[sidx])
    
    
    N_array[j] = N

    bins = binlims[1:] - (binlims[1] - binlims[0])/2
    idx = np.isfinite(np.log(N))
    
    def func(x, a, b, c):
      return (a*x) + c - np.log(b)
    
    popt, _ = curve_fit(func, bins[idx], np.log(N)[idx], 
                        bounds=([-np.inf,-np.inf,-1.00001],[np.inf,np.inf,-0.9999999]))
                        #,p0=[40,1,-1])
    print(popt)
    a_array[j] = popt



x = np.linspace(0,1,100)
fig,ax = plt.subplots(1,1,figsize=(6,5))

colors = [plt.cm.Set2(i) for i in range(len(a_array))]

ax.hlines(0,0.,1.1, linestyle='dashed', color='black')

for i,(_a,_N,label,c) in enumerate(zip(a_array,N_array,wavelengths,colors)):
    ax.plot(1 - x,np.log10(np.exp((_a[0]*x - 1))/_a[1]), c=c, 
            label='$\lambda = %s \, \mathrm{\mu m}$ | $a = %.2f$'%(label,_a[0]), linestyle='dashed')

    _y = np.log10(_N)
    _y[np.where(_y == -np.inf)[0].max()] = 0.
    ax.step(1 - binlims[1:], _y, where='pre', c=c)


ax.legend(frameon=False)
# ax.text(0.05,0.9,'$a = %.2f$'%popt[0], transform=ax.transAxes, size=10)
ax.set_xlim(0.,0.5)
ax.set_ylim(-0.1,2.8)
ax.set_xlabel('$D$') # S_{i} \,/\, S_{i}^{\mathrm{max}}$')
ax.set_ylabel('$\mathrm{log_{10}(Normalised \; frequency)}$')
plt.show()
# fname = 'plots/dimming_distribution.png'; print(fname)
# plt.savefig(fname,dpi=300,bbox_inches='tight')

