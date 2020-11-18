"""
Script for creating subsets of particles from gizmo sims

Authors: 
- Sydney Lower
- Chris Lovell
"""
import sys
sys.path.append('..')

import os
import h5py
import caesar
import numpy as np
import json
from simba import simba
sb = simba()

verbose=1

output_dir='/blue/narayanan/c.lovell/simba/m100n1024/out/'

cs = caesar.load(sb.cs_directory+'m100n1024_078.hdf5')
# cs = sb.get_caesar(snap)

# for snap in sb.snaps:
snap_idx = str(sys.argv[1])
# snap = sb.lightcone_snaps[::-1][snap_idx]
snap = str(snap_idx)
print(snap)
# input("Press Enter to continue...")

ds = sb.get_sim_file(snap, snap_str='snap_m100n1024_%s.hdf5')#snapshot_%s.hdf5')

out_folder=output_dir+'snap_'+snap+'/'
if not os.path.exists(out_folder): os.mkdir(out_folder) 

with open('galaxy_selection.json', 'r') as fp:
    _dat = json.load(fp)[snap]
    hidx = [_g['hidx'] for k,_g in _dat.items()]


# only download unique halos
hidx = np.unique(hidx)

ignore_fields = [] 

## ---- write datasets
with h5py.File(ds, 'r') as input_file:

    header = input_file['Header']

    for ptype in ['PartType0','PartType1','PartType4','PartType5']:
        print(ptype)
        
        pidx = int(ptype[8:])   # get particle type index

        if ptype in input_file:  # check particle type present
            for k in input_file[ptype]:   # loop through fields
                
                if k in ignore_fields:
                    if verbose > 1: print(k,'skipped...')
                    continue
                
                if verbose > 0: print(ptype,k)
                
                # load a given field (the bottleneck)
                temp_dset = input_file[ptype][k][:]
                if verbose > 1: print(temp_dset.shape)
        
                for halo in hidx:

                    output = '{}/subset_{:05.0f}.h5'.format(out_folder,halo)

                    # # check if already downloaded...
                    # if os.path.exists(output):
                    #     print("Halo",halo,"already downloaded.")
                    #     continue
    
                    if ptype == 'PartType0': 
                        plist = cs.halos[halo].glist
                    elif ptype == 'PartType4': 
                        plist = cs.halos[halo].slist
                    elif ptype == 'PartType1': 
                        plist = cs.halos[halo].dmlist
                    elif ptype == 'PartType5': 
                        plist = cs.halos[halo].bhlist
                    else: 
                        if verbose > 0: print('No compatible particle type specified')
                        continue
                    
            
                    # write to file
                    with h5py.File(output, 'a') as output_file:

                        ## create header if it doesn't exist
                        if 'Header' not in output_file:
                            output_file.copy(header, 'Header')

                        if ptype not in output_file:
                            output_file.create_group(ptype)
                        
                        if '%s/%s'%(ptype,k) in output_file:
                            if verbose > 1: print("dataset already exists. replacing...")
                            del output_file[ptype][k]
    
                        output_file[ptype][k] = temp_dset[plist]
                        
                        temp = output_file['Header'].attrs['NumPart_ThisFile']
                        temp[pidx] = len(plist)
                        output_file['Header'].attrs['NumPart_ThisFile'] = temp
                        
                        temp = output_file['Header'].attrs['NumPart_Total']
                        temp[pidx] = len(plist)
                        output_file['Header'].attrs['NumPart_Total'] = temp
           
    
    
    ## Force numpart for black holes to zero (required if caesar not run with black holes)
    # for galaxy in gal_idx:
    #     print(galaxy)
    #     output = '{}/subset_{:05.0f}.h5'.format(out_folder,galaxy)
    #     with h5py.File(output, 'a') as output_file:
    # 
    #         temp = output_file['Header'].attrs['NumPart_ThisFile']
    #         temp[5] = 0#len(plist)
    #         output_file['Header'].attrs['NumPart_ThisFile'] = temp
    #         
    #         temp = output_file['Header'].attrs['NumPart_Total']
    #         temp[5] = 0#len(plist)
    #         output_file['Header'].attrs['NumPart_Total'] = temp
    
    
