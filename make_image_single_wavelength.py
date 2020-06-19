import numpy as np
import glob

import matplotlib as mpl 
# mpl.use('nbAgg')
import matplotlib.pyplot as plt

from hyperion.model import ModelOutput
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from simba import simba
sb = simba()


wav = 850  # micron
_dir = '/home/c.lovell/data/ufrc/simba/m25n512/run/snap_159/'
ids = np.loadtxt('%s/ids.txt'%_dir,dtype=int)

filt_wl, filt_trans = sb.scuba850_filter('model850.txt')

z = 2.025
luminosity_distance = cosmo.luminosity_distance(z).to(u.cm)

for _i,_id in enumerate(ids):

    if _i != 14:
        continue

    f_sed = glob.glob('%s/gal_%i/snap159.galaxy*.rtout.sed'%(_dir,_id))
    f_img = glob.glob('%s/gal_%i/snap159.galaxy*.rtout.image'%(_dir,_id))
    
    if len(f_img) < 1:
        print(_i,_id,"No image!")
    else:
        print(_i,_id,"\n",f_sed,"\n",f_img)
        m = ModelOutput(f_sed[0])
   
        wl,lum = m.get_sed(inclination='all',aperture=-1)
        wl = np.asarray(wl)*u.micron
        lum = np.asarray(lum)*(u.erg/u.s)

        mag_850 = sb.calc_mags(wl.copy(),lum.copy(),z,filt_wl=filt_wl,filt_trans=filt_trans).value[0]
        
        # plt.loglog(wl,lum[0])
        # plt.show()
        print("S_850: %.4f mJy"%mag_850)

        m = ModelOutput(f_img[0])
        image = m.get_image(units='Jy', distance=luminosity_distance.value, component='total')
        # image = m.get_image(units='ergs/s')#, distance=luminosity_distance.value)
        


        wl = image.wav
        wl *= (1.+z)
        # Find the closest wavelength
        iwav = np.argmin(np.abs(wav - wl))
        
        # Calculate the image width in kpc
        w = image.x_max * u.cm
        w = w.to(u.kpc)
        
        np.savetxt('arrays/gal_%i_850.txt'%_id, image.val[0,:,:,iwav])
        print(image.val.shape)
        print(image.val[0,:5,:5,iwav])
        
        
        fig,ax = plt.subplots(1,1)

        cax = ax.imshow(np.log10(image.val[0, :, :, iwav]), cmap=plt.cm.viridis,
                        origin='lower', extent=[-w.value, w.value, -w.value, w.value])
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel('x (kpc)')
        ax.set_ylabel('y (kpc)')
        
        # plt.colorbar(cax, label='log Luminosity (ergs/s)', format='%.0e')
        plt.colorbar(cax, label='$\mathrm{log_{10} \; Flux \; (Jy)}$')# , format='%.0e')
        
        # plt.show()
        # fig.savefig('images/gal_%i_850.png'%_id, bbox_inches='tight', dpi=250)
        plt.close() 
 
