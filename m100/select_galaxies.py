# import sys
# sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import json
import caesar

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from simba import simba
sb = simba()

verbose = True

fname = 'galaxy_selection.json'

sfr_lim = 500
mstar_lim = 5.8e8
R = 60  # SCUBA2_beam = 14.8  # 850 micron
N_lcs = 50


outs = np.loadtxt(sb.output_file)
snaps = sb.lightcone_snaps
# zeds = [1./outs[int(snap)] - 1 for snap in snaps]

# n_neighbours = {snap: None for snap in snaps}

out = {}
lc_out = {str(_lc): {} for _lc in np.arange(N_lcs)}

# for i,snap in enumerate(snaps):
snap = '078'

# z = 1./outs[int(snap)] - 1
# print("\nz:",z,snap)

out[snap] = {}

cs = caesar.load(sb.cs_directory+'m100n1024_078.hdf5')
# cs = sb.get_caesar(snap,fname='caesar_%s_)#, fname='m50n1024_%s.hdf5')

a = cs.simulation.scale_factor

sfr = np.array([g.sfr.value for g in cs.galaxies])
mstar = np.array([g.masses['stellar'].value for g in cs.galaxies])
coods_code = np.array([g.pos.in_units('code_length').value for g in cs.galaxies])
coods_pkpc = np.array([g.pos.to('kpc').value for g in cs.galaxies])
coods_cMpc = np.array([g.pos.to('Mpccm').value for g in cs.galaxies])
hidx = np.array([g.parent_halo_index for g in cs.galaxies])


if verbose: print("N:",len(sfr))
mask = (mstar > mstar_lim)
if verbose: print("N (mstar):",np.sum(mask))
mask = mask & (sfr > sfr_lim)
if verbose: print("N (mstar & sfr):",np.sum(mask))
idx_arr = np.where(mask)[0]

if verbose: print("Finding selection neighbours...")
distances = cdist(coods_pkpc[idx_arr], 
                  coods_pkpc[idx_arr])

np.fill_diagonal(distances,np.inf)
idxs = np.array(np.where(distances < R)).T 


for j in np.arange(len(idxs)): 
    if j >= len(idxs): 
        break 
    else: 
        if idxs[j,0] in idxs[j:,1]: 
            didx = np.where(idxs[j,0] == idxs[j:,1])[0] + j
            for d in np.sort(didx)[::-1]: 
                idxs = np.delete(idxs,d,axis=0) 

idx_arr = np.delete(idx_arr,idxs[:,1])
if verbose: print("N:",idx_arr.shape)

for idx in idx_arr:
    out[snap][int(idx)] = {}
    out[snap][int(idx)]['pos'] = list(coods_code[idx])
    out[snap][int(idx)]['hidx'] = int(hidx[idx])


print("Writing comoving selection to %s"%fname)
with open(fname, 'w') as fp:
    json.dump(out, fp,sort_keys=True,indent=4,separators=(',', ': '))


