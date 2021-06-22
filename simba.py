import glob
import numpy as np
from scipy.spatial.distance import cdist
from scipy.integrate import quad

import caesar

from hyperion.model import ModelOutput
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from astropy import constants

from scipy.integrate import simps

from pyphot.astropy import UnitFilter # unit

class simba:
    def __init__(self):
        self.snaps = ['134','129','124','114','105','097','090',
                      '087','084','081','078','076',
                      '074','071',#'069',
                      '068','066',
                      '062','056',#'050',
                      '049','046',
                      '042','036','033','030','028',
                      '026']


        self.lightcone_snaps = np.array([str(s).zfill(3) for s in np.arange(20,145,2)[::-1]])

        self.sim_extension = 'm25n512'
        # self.sim_directory='/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/output/'
        # self.cs_directory=self.sim_directory+'Groups/'
        self.sim_directory='/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
        self.cs_directory=self.sim_directory+'Groups/'

        self.output_file='outputs_boxspace50.txt'
        self.cosmo = cosmo


    def get_sim_file(self,snap,snap_str=None,verbose=False):
        if snap_str is None:
            snap_str = "snap_%s_%s.hdf5"%self.sim_extension
            
        if verbose: print("snap_str:",snap_str)
        return self.sim_directory+(snap_str%snap)


    def get_caesar(self,snap,fname=None,verbose=False):
        if fname is None:
            fname = self.sim_extension+'_%s.hdf5'
        
        if verbose: print("fname:",fname)
        fname = self.cs_directory+(fname%snap)
        return  caesar.load(fname)


    def get_galaxy_id(self,directory):
        run = glob.glob('{0}/snap*.galaxy*.rtout.sed'.format(directory))
    
        if len(run) > 1:
            raise ValueError('More than one spectrum in directory')
        elif len(run) == 0:
            raise ValueError('No output spectrum in this directory')
    
    
        return run[0][len(directory)+15:-10]

    def luminosity_to_flux_density(self,wl,spec,z):

        wl *= u.micron
        spec *= u.erg / u.s

        dl = self.cosmo.luminosity_distance(z)
        dl = dl.to(u.cm)

        wl *= (1.+z)  # shift by redshift
        
        nu = constants.c.cgs/(wl.to(u.cm))
        nu = nu.to(u.Hz)

        # spec /= wl.to(u.AA) # erg s^-1 AA^-1
        spec /= nu # erg s^-1 Hz^-1

        fd = spec / (4.*np.pi*dl**2.) # erg s^-1 cm^-2 Hz^-1
        return fd.to(u.mJy)
    

#     """
#     Deperecated version for individual spec files
#     """
#     def get_spectrum(self,directory,stype='out',fname=None):
#         if fname is None:
#             run = glob.glob('{0}/snap*.galaxy*.rt{1}.sed'.format(directory,stype))
#         else:
#             run = glob.glob('{0}/{1}'.format(directory,fname))
#     
#         if len(run) > 1:
#             raise ValueError('More than one spectrum in directory')
#         elif len(run) == 0:
#             raise ValueError('No output spectrum in this directory')
#     
#         m = ModelOutput(run[0])
#         wav,lum = m.get_sed(inclination='all',aperture=-1)
#     
#         # set units
#         wav  = np.asarray(wav)*u.micron
#         lum = np.asarray(lum)*u.erg/u.s
#     
#         return(wav,lum)

    
    def get_spectrum(self,fname,gal_id,stype='out'):
        """
        For combined spec files
        """
        # if fname is None:
        #     run = glob.glob('{0}/snap*.galaxy*.rt{1}.sed'.format(directory,stype))
        # else:
        #     run = glob.glob('{0}/{1}'.format(directory,fname))
    
        # if len(run) > 1:
        #     raise ValueError('More than one spectrum in directory')
        # elif len(run) == 0:
        #     raise ValueError('No output spectrum in this directory')
   
        if gal_id is None:
            m = ModelOutput(filename=fname)
        else:
            m = ModelOutput(filename=fname,group=gal_id)
        wav,lum = m.get_sed(inclination='all',aperture=-1)
    
        # set units
        wav  = np.asarray(wav)*u.micron
        lum = np.asarray(lum)*u.erg/u.s
    
        return(wav,lum)
    
    def scuba850_filter(self,fdir='data/model850.txt'):
        dat = np.loadtxt(fdir)
        nu = dat[:,0] * 1e9
        transmission = dat[:,14]
        # transmission /= transmission.max()
        wav = 2.99792458e8 / nu # m
        wav *= 1e6 # micron
        return wav[::-1], transmission[::-1]

    @staticmethod
    def calc_df(y,volume,bin_edges):
        hist, dummy = np.histogram(y, bins=bin_edges)
        hist = np.float64(hist)
        phi = (hist / volume) / (bin_edges[1] - bin_edges[0])
        phi_sigma = (np.sqrt(hist) / volume) /\
                    (bin_edges[1] - bin_edges[0]) # Poisson errors
    
        return phi, phi_sigma, hist
    
    @staticmethod
    def blending(coods,y,R=0.240,verbose=True):
        """
        Args:
        coods (array, (N,3))
        y (array, N)
        R (float) kpc
        """
        _c = np.array(coods)[:,[0,1]]
        distances = cdist(_c,_c)
        np.fill_diagonal(distances,np.inf)
        idxs = np.array(np.where(distances < R))

        if verbose==True: print("Blend count:",len(idxs[0]))
        
        for i,idx in enumerate(idxs.T[::-1]):
            if idx[0] not in idxs[1][::-1][:i]: # check index not already acounted for
                y[idx[1]] += y[idx[0]]          # add up SFRs
                y = np.delete(y,idx[0])         # delete old value of SFR
    
        return y
   

    def calc_mags(self,wl,lum,z,filt_wl=[845,846,850,854,855],
                  filt_trans=[0.,1.,1.,1.,0.],lambda_pivot=850):
        """
        
        Args:
            wl (arr, float): Angstrom
            lum (arr, float): erg s^-1

        """
        filt_wl = filt_wl * u.micron

        dl = self.cosmo.luminosity_distance(z)
        dl = dl.to(u.cm)

        wl *= (1.+z)  # shift by redshift
        # nu = constants.c.cgs/(wl.to(u.cm))
        # nu = nu.to(u.Hz)

        # lum /= nu
        lum /= wl.to(u.AA) # erg s^-1 AA^-1

        flux = lum / (4.*np.pi*dl**2.) # erg s^-1 cm^-2 AA^-1
        # flux *= (1+z)

        # filt_nu = (2.99792458e8 * u.m / u.s) / filt_wl
        pivot_wl = lambda_pivot * u.micron
        pivot_nu = constants.c / pivot_wl

        #tophat = UnitFilter(filt_wl, filt_trans, name='tophat', dtype='energy', unit='micron')
        tophat = UnitFilter(filt_wl, filt_trans, name='tophat', dtype='energy', unit='angstrom')
        flux_tophat = tophat.get_flux(wl.to(u.AA), flux)# flux_unit='fnu',

        # flam to fnu (erg s^-1 cm^-2 Hz^-1)
        flux_tophat = flux_tophat * pivot_wl / pivot_nu
        return flux_tophat.to(u.mJy)


    def _volume_differential_comoving(self,z_low,z_upp,N=100):
        z_arr = np.linspace(z_low, z_upp, N)
        dVC = self.cosmo.differential_comoving_volume(z_arr).to(u.Mpc**3 / u.deg**2).value
        return simps(dVC,z_arr)
    
    
    def comoving_phi(self,mags,zeds,vol,bin_edges,snaps=None,verbose=False):
        if snaps is None:
            snaps = self.snaps
    
        phi = np.array([self.calc_df(mags[snap],vol,bin_edges)[0] for snap in snaps])
    
        z_integ_lims = np.array(zeds)[1:] - (np.diff(zeds) / 2)
        z_integ_lims = np.insert(z_integ_lims, 0, np.max([0,zeds[0] - np.diff(zeds)[0]]))
        z_integ_lims = np.concatenate((z_integ_lims, [zeds[-1] + np.diff(zeds)[-1] / 2]))
        if verbose: print("z_integ_lims:",z_integ_lims)
    
        # whole_sky = (4 * 180 * 180 / np.pi) * u.deg**2
        # phi_deg = [(np.diff(Planck15.comoving_volume([z_integ_lims[i],z_integ_lims[i+1]])) /\
        #                        whole_sky).value[0] for i in np.arange(len(zeds))]
    
        phi_deg = [self._volume_differential_comoving(z_integ_lims[i],z_integ_lims[i+1]) for i in np.arange(len(zeds))]
    
        # dN / dS (mJy^-1 deg^-2)
        phi = [a*b for a,b in zip(phi,phi_deg)]
        return np.sum(phi,axis=0)
    

    def calc_cumulative(self,mags,bin_edge,snaps=None):
        # if snaps is None:
        #     snaps = self.lightcone_snaps

        # _mags = np.vstack([mags[snap] for snap in snaps])
        return np.array([np.sum(mags > S) for S in bin_edge]).astype(float)



    def extract_output(self,fname,gal_id,out_dir='.'):
        """
        Create a new output file to be used by Hyperion `ModelOutput` class

        Args:
        gal_id (str): galaxy ID string, top level in the parent file
        """
        with h5py.File(fname, 'r') as h5_in: 
            with h5py.File('%s/%s.h5'%(out_dir, gal_id), 'w') as h5_out: 
                for k in h5_in[gal_id].keys(): 
                    f.copy('%s/%s'%(gal_id, k), h5_out) 


    def calc_phi(self,S,volume,binlimits=None):
        if binlimits is None:
            _n,binlims = np.histogram(S)
        else:
            _n,binlims = np.histogram(S,bins=binlimits)
    
        bins = binlims[:-1] + (binlims[1:] - binlims[:-1])/2
        return (_n/volume)/(binlims[1] - binlims[0]), bins
 
   

class Schechter():

    def __init__(self, Dstar=1e-2, alpha=-1.4, log10phistar=4):
        self.sp = {}
        self.sp['D*'] = Dstar
        self.sp['alpha'] = alpha
        self.sp['log10phistar'] = log10phistar


    def _integ(self, x,a,D):
        return 10**((a+1)*(x-D)) * np.exp(-10**(x-D))


    def binPhi(self, D1, D2):
        args = (self.sp['alpha'],self.sp['D*'])
        gamma = quad(self._integ, D1, D2, args=args)[0]
        return gamma * 10**self.sp['log10phistar'] * np.log(10)


    def _CDF(self, D_lowlim, normed = True, inf_lim=30):
        log10Ls = np.arange(self.sp['D*']+5.,D_lowlim-0.01,-0.01)
        CDF = np.array([self.binPhi(log10L,inf_lim) for log10L in log10Ls])
        if normed: CDF /= CDF[-1]

        return log10Ls, CDF


    def sample(self, volume, D_lowlim, inf_lim=100):
        D, cdf = self._CDF(D_lowlim, normed=False, inf_lim=inf_lim)
        n = np.random.poisson(volume * cdf[-1])
        ncdf = cdf/cdf[-1]
        D_sample = np.interp(np.random.random(n), ncdf, D)

        return D_sample


