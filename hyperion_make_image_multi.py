import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13

import functools

from hyperion.model import ModelOutput
from hyperion.util.constants import pc
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel,Gaussian2DKernel
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo

import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datetime import datetime
from datetime import timedelta

from multiprocessing import Pool
import glob
import pdb

def get_image(filename,dist):
    try:
        m = ModelOutput(filename)
        return m.get_image(inclination='all',distance=luminosity_distance,units='Jy')
    except OSError:
        print("OS Error in reading in: "+filename)
        pass


for gidx in ['3','8','51','54','94','100','134','139']:
    _gal = int(gidx)

    _dir = '/blue/narayanan/c.lovell/simba/m100n1024/run/snap_078/gal_%s/*.rtout.image'%_gal
    files = glob.glob(_dir)
    image_limit = 30 #kpc
    
    wav = 850
    nprocesses=16
        
    z = 2.025
    luminosity_distance = cosmo.luminosity_distance(z).to(u.cm).value
    
    pool = Pool(processes=nprocesses)
    
    t1=datetime.now()
    _partial = functools.partial(get_image, dist=luminosity_distance)
    result= pool.map(_partial,files)
    t2 = datetime.now()
    print('Execution time to get image = '+str(t2-t1))
    pool.close()
    
    iwav = np.argmin(np.abs( (wav/(1+z)) - result[0].wav))
    print("wl:",result[0].wav[iwav] * (1+z))
    
    
    #pop out the None's passed out by the try/except in the multiporcess
    #https://www.geeksforgeeks.org/python-remove-none-values-from-list/
    result = np.array([i for i in result if i])
    
    orientations = result[0].val.shape[0]
    dim = result[0].val.shape[1]
    
    for _orientation in np.arange(orientations): 
    
        fname = 'm100_g%s_o%s'%(_gal,_orientation)
        print(fname)
    
        im_array = np.zeros([dim,dim])
        for r in range(len(result)):
            im_array += result[r].val[_orientation,:,:,iwav]
                    
        im_array/=len(result)
    
        np.savetxt('arrays/%s.txt'%fname, im_array)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('$\mathrm{kpc}$')
        ax.set_ylabel('$\mathrm{kpc}$')
    
    
        imgplot =plt.imshow(np.log10(im_array), 
                            vmax=np.log10(np.max(im_array)),
                            vmin=-10,
                            cmap=plt.cm.plasma, origin='lower', 
                            extent=[-image_limit, image_limit, -image_limit, image_limit])
        
        # cax = fig.add_axes([0.82, 0.1, 0.03, 0.7])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(imgplot, cax=cax, orientation='vertical')
        cbar.set_label('$\mathrm{log_{10}\,Jy}$')
        
        fig.savefig('images/%s.png'%fname, bbox_inches='tight',dpi=300)
        plt.close()
    
    
