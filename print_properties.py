import sys 
sys.path.append('..') 

import glob
import numpy as np 
import json 
import h5py
 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

import astropy.units as u

from scipy.spatial.distance import cosine

from hyperion.model import ModelOutput

import caesar 
from caesar.data_manager import DataManager

from simba import simba 
sb = simba() 


snap = '078'
z = 2.025
rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run'
#rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run_sed'

_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)
redshift = cs.simulation.redshift

_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]

galaxy_names = {3: 'Smiley', 8: 'Haydon', 51: 'Guillam', 54: 'Alleline',
                94: 'Esterhase', 100: 'Prideaux', 134: 'Bland', 139: 'Lacon'}

disc_label=['Yes','No','Yes','No','Yes','No','No','Yes']

print('\\begin{tabular*}{\\tablewidth}{llllll}')
print('\hline\nLabel & $\mathrm{log_{10}}(M_{\star,30 \; \mathrm{kpc}} \,/\, \mathrm{M_{\odot}}) \,^1$ & $\mathrm{log_{10}}(M_{\mathrm{dust}} \,/\, \mathrm{M_{\odot}}) \,^2$ & $\mathrm{SFR \,/\, M_{\odot} \, yr^{-1}} \,^3$ & $\left< S_{850} \\right> \,/\, \mathrm{mJy} \,^4$ & Disc morphology? $\,^5$ \\\\ \n\hline')

for _g,_disc in zip(galaxies,disc_label):
    gidx = _g.GroupID
    with h5py.File('sed_out_hires.h5','r') as f:
        wav = f['%s/Wavelength'%gidx][:]
        spec = f['%s/SED'%gidx][:]
        f850 = sb.calc_mags(wav*u.micron,spec*u.erg/u.s,redshift)


    print('%s & %.2f & %.2f & %.2f & %.2f & %s \\\\[2pt]'%\
            (galaxy_names[_g.GroupID], np.log10(_g.masses['stellar_30kpc']), 
             np.log10(_g.masses['dust']), _g.sfr_100, np.median(f850).value, _disc))


print('\hline')
print('\end{tabular*}')
