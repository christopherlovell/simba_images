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

# from hyperion.model import ModelOutput
# import yt

import caesar
from caesar.data_manager import DataManager

from simba import simba
sb = simba()

snap = '078'
_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)
z = cs.simulation.redshift

_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]
galaxy_names = {3: 'Smiley', 8: 'Haydon', 51: 'Guillam', 54: 'Alleline',
                94: 'Esterhase', 100: 'Prideaux', 134: 'Bland', 139: 'Lacon'}

#for _c,_g in enumerate(galaxies):
_c = 0; _g = galaxies[0]
gidx = _g.GroupID
with h5py.File('sed_out_hires_test.h5','r') as f:
    wav = f['%s/Wavelength'%gidx][:]; spec = f['%s/SED'%gidx][:]


fig, ax = plt.subplots(1,1)
[ax.plot(np.log10(wav), np.log10(_s), alpha=0.1, color='black') for _s in spec]

ax.set_xlabel('$\lambda \,/\, \mathrm{\mu m}$')
ax.set_ylabel('$L \,/\, \mathrm{erg \; s^{-1}$')

plt.show()
