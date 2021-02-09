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
z = 2.025
rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run'

_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)
redshift = cs.simulation.redshift
_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]

binlims = np.linspace(0,1,200)
N = np.zeros(len(binlims)-1, dtype=int)
gindexes = [3,8,51,54,94,100,134,139]
for i,gidx in enumerate(gindexes):
    _g = cs.galaxies[gidx]

    gidx = galaxies[i].GroupID
    with h5py.File('sed_out_hires.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        spec = sb.luminosity_to_flux_density(wav,spec,redshift)

    sidx = np.argmin(np.abs(wav - 200))
    

    _n,_dummy = np.histogram(spec[:,sidx].value/spec[:,sidx].max().value, bins=binlims)
    N += _n
 
    # plt.hist(spec[:,sidx].value/spec[:,sidx].max().value,histtype='step')
    # print(np.std(spec,axis=0)[sidx] / np.mean(spec,axis=0)[sidx])


bins = binlims[1:] - (binlims[1] - binlims[0])/2
idx = np.isfinite(np.log(N))

def func(x, a, b, c):
  return (a*x) + c - np.log(b)

popt, _ = curve_fit(func, bins[idx], np.log(N)[idx], 
                    bounds=([-np.inf,-np.inf,-1.00001],[np.inf,np.inf,-1]))
print(popt)


x = np.linspace(0,1,100)
plt.plot(x,np.exp((popt[0]*x - 1))/popt[1])
plt.plot(bins,N)
plt.xlim(0.8,1.0)
plt.xlabel('$S_{i} \,/\, S_{i}^{\mathrm{max}}$')
plt.ylabel('Normalised frequency')
plt.show()
fname = 'plots/dimming_distribution.png'
print(fname)
plt.savefig(fname,dpi=300,bbox_inches='tight')

