import numpy as np
import yt
import caesar
from caesar.pyloser import pyloser
from caesar.cyloser import compute_AV


_dir = '/cosma7/data/dp104/dc-dave2/sim/m100n1024/s50j7k/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')

ds = yt.load(_dir+'snap_m100n1024_078.hdf5')

phot = pyloser.photometry(cs, [cs.halos[0]], ds=ds)

phot.ssp_table_file = \
    '/cosma7/data/dp004/dc-love2/codes/caesar/FSPS_Chab_EL.hdf5'

phot.init_pyloser()



def transform_coods(theta,phi):
    # theta += np.pi
    # phi += np.pi
    x = np.round(np.sin(theta),2) * np.round(np.cos(phi),2)
    y = np.round(np.sin(theta),2) * np.round(np.sin(phi),2)
    z = np.round(np.cos(theta),2)    
    print(x,y,z)

    alpha = np.arctan( (np.round(np.sin(phi),2) * np.round(np.sin(theta),2)) / np.round(np.cos(theta),2) )
    alpha[(z<0) & (y>=0)] += np.pi
    alpha[(z<0) & (y<0)] -= np.pi
    alpha[(z==0) & (y>0)] = np.pi/2
    alpha[(z==0) & (y<0)] = -np.pi/2
    beta = np.arctan( (np.round(np.sin(theta),2) * np.round(np.cos(phi),2)) / np.round(np.cos(theta),2) )
    # beta[(z<0) & (x>=0)] += np.pi      # beta[(z<0) & (x<0)] -= np.pi  
    # beta[(z==0) & (x>0)] = np.pi/2     # beta[(z==0) & (x<0)] = -np.pi/2  
    alpha[np.isnan(alpha)] = 0.
    beta[np.isnan(beta)] = 0.

    for _a,_b in zip(alpha,beta):
        c  = np.cos(_a)
        s  = np.sin(_a)
        Rx = np.array([[1.0,0.0,0.0],
                       [0.0,  c, -s],
                       [0.0,  s,  c]])#.round(2)
    
        c  = np.cos(_b)
        s  = np.sin(_b)
        Ry = np.array([[  c,0.0, -s],
                       [0.0,1.0,0.0],
                       [  s,0.0,  c]]).round(2)
     
        vals = np.array([0.,0.,1.]) 
        vals = np.dot(Rx, vals)
        vals = np.dot(Ry, vals)
        print(vals.round(2)) 
    
    
    alpha *= 180./np.pi
    beta  *= 180./np.pi   
    return alpha,beta


theta = np.array([90,90, 90, 90,0,180]) * np.pi/180.
phi = np.array([0, 90,180,270,0,  0]) * np.pi/180.
alpha,beta = transform_coods(theta,phi)
A_V_out =[None] * len(theta)
for i,(_a,_b) in enumerate(zip(alpha,beta)):
    phot.obj.AV_star = compute_AV(phot,_a,_b)
    phot.Av_per_group()
    A_V_out[i] = phot.groups[0].group_Av


