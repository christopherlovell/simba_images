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

snap = '078'
_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
# cs.data_manager = DataManager(cs)
z = cs.simulation.redshift

_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]
galaxy_names = {3: 'Smiley', 8: 'Haydon', 51: 'Guillam', 54: 'Alleline',
                94: 'Esterhase', 100: 'Prideaux', 134: 'Bland', 139: 'Lacon'}

frac_spec = np.zeros(8)
frac_lambda = np.zeros(8)
diff = np.zeros(8)
for _c,_g in enumerate(galaxies):
    gidx = _g.GroupID
    with h5py.File('sed_out_hires_test.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]; spec = f['%s/SED'%gidx][:]
        _max = np.argmax(spec,axis=1)
        spec = sb.luminosity_to_flux_density(wav,spec,z)
        spec_max = spec[:,_max]
        lambda_max = wav[_max]
        diff[_c] = "%.2f"%(np.max(spec_max).value - np.min(spec_max).value)
        print(_c, "max: %.2f"%np.min(spec_max).value, "| min: %.2f"%np.max(spec_max).value,
              "| diff:", diff[_c])
        frac_spec[_c] = np.max(spec_max) / np.min(spec_max)
        print("%.2f"%frac_spec[_c])
        frac_lambda[_c] = np.max(lambda_max) - np.min(lambda_max)


print("median frac:", np.median(frac_spec))
print("median absolute:", np.median(diff))
