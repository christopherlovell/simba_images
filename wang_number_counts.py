import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

from simba import Schechter, simba
#from simba import simba
sb = simba()

dat = {}
dat['250'] = {}
dat['250']['S'] = np.array([0.95,1.56,2.58,4.25,7.01,11.56,19.06,31.42,51.80,85.41,140.81])
dat['250']['dNdS'] = np.array([5247.40,8670.32,11872.1,15274.6,19670.4,21567.6,20242.8,
                               13876.4,7198.93,3277.45,693.836]) ## ---- S^2.5 dN / dS [ ster^-1]
# dat['250']['error'] = np.array([15.8654,29.6727,50.5201,83.3767,137.666,
#              209.740,295.650,356.156,373.247,366.430,245.308])

dat['350'] = {}
dat['350']['S'] = np.array([0.95,1.56,2.58,4.25,7.01,11.56,19.06,31.42,51.80,85.41,140.81])
dat['350']['dNdS'] = np.array([5547.49,8550.90,10501.0,11935.8,13523.4,13996.3,
                               10061.0,5576.15,1199.82,163.872,173.459])
# error 16.3128,29.4676,47.5132,73.7033,114.147,168.962,208.431,225.772,152.378,81.9362,122.654        

for _wl in ['250','350']:
    dat[_wl]['dNdS'] /= ((dat[_wl]['S'] * 1e-3)**2.5)  ## ---- dN / dS [ster^-1]
    dat[_wl]['dNdS'] /= 3282.8  ## ---- dN / dS [deg^-2]
    dat[_wl]['dNdlogS'] = (dat[_wl]['S']*1e-3) * dat[_wl]['dNdS'] * np.log(10)


def schech_func(bins, Dstar=0.568, 
                alpha=-1.4, log10phistar=3):

    model = Schechter(Dstar=Dstar, alpha=alpha, log10phistar=log10phistar)

    db = (bins[1] - bins[0])
    # binlimits = np.append(bins-db/2,bins[-1]+db/2)
    binlimits = np.append(bins,bins[-1]+db)

    return np.log10([model.binPhi(b1,b2)/(b2-b1) for b1,b2 in \
                     zip(binlimits[:-1],binlimits[1:])])

_wl = '350'
_p,_dummy = curve_fit(schech_func, np.log10(dat[_wl]['S']), 
                      np.log10(dat[_wl]['dNdlogS']), p0=[1.5,-1.4,8])
print("Wavelength: %s\n"%_wl,"Parameters:",_p)


bins = np.linspace(0,2.3,20)
_phi = schech_func(bins, *_p)
#_phi = schech_func(bins, *[0.5,-1.4,3])

plt.plot(bins, _phi)
plt.scatter(np.log10(dat[_wl]['S']), np.log10(dat[_wl]['dNdlogS']))

plt.show()

