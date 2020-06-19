import numpy as np
import glob

import matplotlib as mpl 
# mpl.use('nbAgg')
import matplotlib.pyplot as plt

from hyperion.model import ModelOutput
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

wav = 850  # micron
_dir = '/home/c.lovell/data/ufrc/simba/m25n512/run/snap_159/'
ids = np.loadtxt('%s/ids.txt'%_dir,dtype=int)




z = 2.025
luminosity_distance = cosmo.luminosity_distance(z).to(u.cm)

for _i,_id in enumerate(ids):

    if _i != 14:
        continue

    f = glob.glob('%s/gal_%i/snap159.galaxy*.rtout.image'%(_dir,_id))
    if len(f) < 1:
        print(_i,_id,"No image!")
    else:
        print(_i,_id,f)
        m = ModelOutput(f[0])
    
        # Get the image from the ModelOutput object
        image = m.get_image(units='Jy', distance=luminosity_distance.value)
        # image = m.get_image(units='ergs/s')#, distance=luminosity_distance.value)
        
        fig,ax = plt.subplots(1,1)
        
        # Find the closest wavelength
        iwav = np.argmin(np.abs(wav - image.wav))
        
        # Calculate the image width in kpc
        w = image.x_max * u.cm
        w = w.to(u.kpc)
        
        np.savetxt('arrays/gal_%i_850.txt'%_id, image.val[0,:,:,iwav])
        print(image.val.shape)
        print(image.val[0,:5,:5,iwav])

        # plot the beast
        cax = ax.imshow(np.log10(image.val[0, :, :, iwav]), cmap=plt.cm.viridis,
                        origin='lower', extent=[-w.value, w.value, -w.value, w.value])
        
        
        # Finalize the plot
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel('x (kpc)')
        ax.set_ylabel('y (kpc)')
        
        # plt.colorbar(cax, label='log Luminosity (ergs/s)', format='%.0e')
        plt.colorbar(cax, label='$\mathrm{log_{10} \; Flux \; (Jy)}$', format='%.0e')
        
        # plt.show()
        fig.savefig('images/gal_%i_850.png'%_id, bbox_inches='tight', dpi=250)
        plt.close() 
 
