import numpy as np
import glob

import matplotlib.pyplot as plt

from hyperion.model import ModelOutput

import sphviewer as sph
from sphviewer.tools import QuickView


run = '/blue/narayanan/c.lovell/simba/m100n1024/run_sed/snap_078_hires_orthogonal'

groupID = 3
fname = glob.glob('%s/gal_%i/snap078.galaxy*.rtout.sed'%(run,groupID))[0]
m = ModelOutput(fname)
_pf = m.get_quantities().to_yt()
ad = _pf.all_data()

_kpc= 3.08568025e+21
_temp = np.array(ad.to_dataframe('temperature'))
_mass = np.array(ad.to_dataframe('cell_mass'))
_density = np.array(ad.to_dataframe('density'))
_radius = np.array(ad.to_dataframe('radius')) / _kpc
_x = np.array(ad.to_dataframe('x')) / _kpc
_y = np.array(ad.to_dataframe('y')) / _kpc
_z = np.array(ad.to_dataframe('z')) / _kpc
_coods = np.squeeze(np.array([_x,_z,_y])).T

radius = 15 # kpc
mask = _radius < radius
print("radius:",radius)
print("mass weighted temp:", np.sum((_temp * _mass)[mask])/np.sum(_mass[mask]))




extent = 15
img = [None,None]
for i,_weight in enumerate([_temp,_temp*(_mass/1.99e38)]):
    Pd = sph.Particles(_coods, _weight)# halog_dust[mask] * 1e10)
    C = sph.Camera(x=0,y=0,z=0,r='infinity',zoom=1,extent=[-extent,extent,-extent,extent],xsize=512, ysize=512)
    S = sph.Scene(Pd, Camera=C)
    R = sph.Render(S)    
    # R.set_logscale()
    img[i] = R.get_image() 

#if vmax is None:
#    vmax = img.max()
#
#if vmin > vmax:
#    print("vmin larger than vmax! Setting lower")
#    vmin = img.max() / 10
#print("vmin,vmax:",vmin,vmax)

#cNorm  = colors.LogNorm(vmin=vmin,vmax=vmax)
#sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

fig,ax = plt.subplots(1,1)
im = ax.imshow(img[1]/img[0], extent=[-extent,extent,-extent,extent])
#             cmap=cmap, norm=cNorm)
fig.colorbar(im, ax=ax, shrink=0.5)

plt.show()

