import json
import numpy as np
import yt
import caesar
from caesar.pyloser import pyloser
from caesar.cyloser import compute_AV


# _dir = '/cosma7/data/dp104/dc-dave2/sim/m100n1024/s50j7k/'
_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')

_dat = json.load(open('m100/galaxy_selection.json','r'))
halos = [cs.halos[n['hidx']] for k,n in _dat['078'].items()]

# subset_dir='/blue/narayanan/c.lovell/simba/m100n1024/out/snap_078/'
# ds = yt.load(subset_dir+'subset_00000.h5')

ds = yt.load(_dir+'snap_m100n1024_078.hdf5')
phot = pyloser.photometry(cs, [halos[0]], ds=ds)

phot.ssp_table_file = \
    '/home/c.lovell/codes/caesar/FSPS_Chab_EL.hdf5'
    # '/cosma7/data/dp004/dc-love2/codes/caesar/FSPS_Chab_EL.hdf5'

phot.init_pyloser()


def transform_coods(theta,phi,tol=20,test=False):

    alpha = np.arcsin(-1 * np.round(np.sin(theta),tol) * np.round(np.sin( phi ),tol))
    beta = np.round(np.arcsin((np.round(np.cos(phi),tol) * np.round(np.sin(theta),tol)) / np.round(np.cos(alpha),tol)),tol)

    beta[np.isnan(beta)] = 0.

    mask = (theta > np.pi/2)
    beta[mask] = np.arccos(     np.round(np.cos(theta[mask]),tol) / np.round(np.cos(alpha[mask]),tol))

    mask = (phi > np.pi/2) & (phi < 3*np.pi/2) & (theta > np.pi/2)
    beta[mask] = -beta[mask]

    if test:
        x = np.round(np.sin(theta),tol) * np.round(np.cos(phi),tol)
        y = np.round(np.sin(theta),tol) * np.round(np.sin(phi),tol)
        z = np.round(np.cos(theta),tol)
        out = np.array([x,y,z]).T
        
        c  = np.round(np.cos(alpha),tol); s  = np.round(np.sin(alpha),tol)
        Rx = np.array([[1.0,0.0,0.0],[0.0,  c, -s],[0.0,  s,  c]])
 
        c  = np.round(np.cos(beta),tol); s  = np.round(np.sin(beta),tol)
        Ry = np.array([[  c,0.0, s],[0.0,1.0,0.0],[ -s,0.0,  c]])
        V = np.array([0,0,1])#.T
 
        out2 = np.round(np.vstack(np.dot(np.dot(Ry,Rx),V)).T,tol)
        print(np.round(out,2),"\n\n",np.round(out2,2),"\n")

        match = np.round(out,5) == np.round(out2,5)
        print(np.sum(match),np.product(match.shape))

    return alpha,beta


theta = np.array([90,90, 90, 90,0,180]) * np.pi/180.
phi = np.array([0, 90,180,270,0,  0]) * np.pi/180.
alpha,beta = transform_coods(theta,phi)
A_V_out = {i: {_g.GroupID: None for _g in phot.groups}\
             for i in np.arange(len(alpha))}

for i,(_a,_b) in enumerate(zip(alpha,beta)):
    phot.obj.AV_star = compute_AV(phot,_a,_b)
    phot.Av_per_group()
    for j,_g in enumerate(phot.groups):
        A_V_out[i][_g.GroupID] = phot.groups[j].group_Av



