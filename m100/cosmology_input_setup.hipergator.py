"""
purpose: to set up slurm files and model *.py files from the
positions written by caesar_cosmology_npzgen.py for a cosmological
simulation.

Updated by Chris Lovell for cosma machine (Durham) 12/10/19
"""
# import sys
# sys.path.append('..')

import numpy as np
from subprocess import call
import caesar
import json
from shutil import copyfile

from simba import simba
sb = simba()

#===============================================
#MODIFIABLE HEADER
#===============================================
nnodes=1
nprocs = 100
model_run_name='m100'
COSMOFLAG=0 #flag for setting if the gadget snapshots are broken up into multiples or not
# SPHGR_COORDINATE_REWRITE = True

selection = 'halo' # galaxy
selection_file='galaxy_selection.json'

dir_base = '/blue/narayanan/c.lovell/simba/m100n1024/'
model_dir_base = '%s/run/'%dir_base
hydro_dir_base = '%s/out/'%dir_base

with open(selection_file, 'r') as fp: 
    snaps = list(json.load(fp).keys())[::-1]

#===============================================

# snaps_to_redshift = {}
# scalefactor = np.loadtxt(sb.output_file)
# for snap in range(len(scalefactor)):
#     snaps_to_redshift['{:03.0f}'.format(snap)] = (1./scalefactor[snap])-1.


for snap in ['078']: # snaps:
    
    with open(selection_file, 'r') as fp: 
        _dat = json.load(fp)[snap]
        ids = np.array(list(_dat.keys()),dtype=int)
        pos = np.array([d['pos'] for k,d in _dat.items()])
        hidx = np.array([d['hidx'] for k,d in _dat.items()])
    
    hydro_dir_remote = hydro_dir_base+'snap_'+snap
    model_dir = model_dir_base+'/snap_'+snap+'_jwst'
    
    redshift = 2.025 # snaps_to_redshift[str(snap)]
    tcmb = 2.73*(1.+redshift)

    # 6 for galaxy, 4 for halo


    N = len(ids)
    print("snap:",snap,"| N:",N)
    if N > 0:

        # for i,nh in enumerate(ids):
        for i,nh in enumerate(ids):
            
            model_dir_remote = model_dir+'/gal_%s'%nh
            
            xpos = pos[i][0]
            ypos = pos[i][1]
            zpos = pos[i][2]

            for n in np.arange(nprocs):

                # only write job submission script once...
                if (n == 0): job_flag = 1
                else: job_flag = 0
    
                cmd = "./cosmology_setup_all_cluster.hipergator.sh "+str(nnodes)+' '+model_dir+' '+hydro_dir_remote+' '+model_run_name+' '+str(COSMOFLAG)\
                        +' '+model_dir_remote+' '+hydro_dir_remote+' '+str(xpos)+' '+str(ypos)+' '+str(zpos)+' '+str(nh)+' '+str(int(snap))+' '+str(tcmb)\
                        +' '+str(i)+' '+str(job_flag)+' '+str(N-1)+' '+str(hidx[i])+' '+str(n)+' '+str(nprocs)
            
                call(cmd,shell=True)
       
        np.savetxt(model_dir+'/ids.txt',ids,fmt='%i')

        copyfile('parameters_master.py', model_dir+'/parameters_master.py')

