import sys
sys.path.append('..')

import numpy as np
import json
import h5py
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import caesar
from caesar.data_manager import DataManager

from simba import simba
sb = simba()


snap = '078'
rt_directory = '/blue/narayanan/c.lovell/simba/m100n1024/run'

_dir = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m100n1024/'
cs = caesar.load(_dir+'Groups/m100n1024_078.hdf5')
cs.data_manager = DataManager(cs)
z = cs.simulation.redshift
_dat = json.load(open('m100/galaxy_selection.json','r'))
galaxies = [cs.galaxies[int(k)] for k in _dat['078'].keys()]

binlims = np.linspace(0.,1,50)
gindexes = [3,8,51,54,94,100,134,139]

wavelengths = [30,70,100,200,250,450,850]
# wavelengths = wavelengths.append(np.linspace(20,1000,100))
wavelengths = np.hstack([np.array(wavelengths), np.linspace(20,1000,100)])
a_array = np.zeros((len(wavelengths),2)) # 3))
N_array = np.zeros((len(wavelengths),len(binlims)-1))

for j,_lambda in enumerate(wavelengths):
    N = np.zeros(len(binlims)-1, dtype=int)

    for i,gidx in enumerate(gindexes):
        _g = cs.galaxies[gidx]
    
        gidx = galaxies[i].GroupID
        with h5py.File('sed_out_hires_test.h5','r') as f:
            wav = f['%s/Wavelength'%gidx][:]
            spec = f['%s/SED'%gidx][:]
            spec = sb.luminosity_to_flux_density(wav,spec,z)
    

        # sidx = np.argmin(np.abs(wav*(1+z) - _lambda))
        sidx = np.argmin(np.abs(wav-_lambda)) ## rest frame
        _n,_dummy = np.histogram(1 - spec[:,sidx].value/spec[:,sidx].max().value, bins=binlims)
        N += _n
        # plt.hist(spec[:,sidx].value/spec[:,sidx].max().value,histtype='step')
        # print(np.std(spec,axis=0)[sidx] / np.mean(spec,axis=0)[sidx])
    
    
    N_array[j] = N
    bins = binlims[1:] - (binlims[1] - binlims[0])/2
    idx = np.isfinite(np.log(N))
   
    if np.sum(idx) > 1:
        def func(x, a, b):
          # return (a*x) + c - np.log(b)
          return (-a*x) + np.log(a) - np.log(b)
    
        popt, _ = curve_fit(func, bins[idx], np.log(N)[idx], 
                   bounds=([-np.inf,-np.inf],[np.inf,np.inf]),p0=[60,1])
               #bounds=([-np.inf,-np.inf,-1.00001],[np.inf,np.inf,-0.9999999]),p0=[60,1,-1])
        print(popt)
        a_array[j] = popt
    else:
        a_array[j] = [2.93459109e+02, 3.68203788e-02] # [-2.93459109e+02, 1.25470219e-04]



def f(x,p1,p2,p3,p4):
    return p1*x**3 + p2*x**2 + p3*x + p4

_pa = np.polyfit(wavelengths[7:],np.log10(a_array[7:,0]), 3)

## example usage
_wl = 250
print("a:",10**f(_wl,*_pa))
# print("b:",10**f(_wl,*_pb))


## plot parameter dependence on wavelength
fig, ax2 = plt.subplots(1,1,figsize=(5,4))
ax1 = fig.add_axes([0.39, 0.58, 0.28, 0.28])
# fig, axs = plt.subplot_mosaic([['A', 'A'], ['A', 'A'], ['B', 'B']], constrained_layout=True)
# ax1 = axs['B']; ax2 = axs['A']
_x = np.linspace(10,900)
ax1.plot(wavelengths[7:], np.log10(a_array[7:,0]))
ax1.plot(_x,f(_x, *_pa))
ax1.set_xlabel('$\lambda_{\mathrm{rest}} \,/\, \mathrm{\mu m}$')
ax1.set_ylabel('$\mathrm{log_{10}}(a)$')
print("a:",_pa)
# ax1.text(0.1, 0.5,'$%.2fx^3 + %.2fx^2 + %.2fx + %.2f$'%_p.tolist(), transform=ax1.transAxes)

# for ax in [ax1,ax2]: 
ax1.set_xlim(20,860)
# plt.show()

## plot dimming distributions (make sure you use a small `wavelengths` array)
x = np.linspace(0,1,100)
# fig,ax = plt.subplots(1,1,figsize=(6,5))

colors = [plt.cm.Set2(i) for i in range(len(a_array[:7]))]


ax2.hlines(0,0.,1.1, linestyle='dashed', color='black')

for i,(_a,_N,label,c) in enumerate(zip(a_array[:7],N_array[:7],wavelengths[:7],colors)):
    print(i)
    ax2.plot(x,np.log10(np.exp((-_a[0]*x))*_a[0]/_a[1]),c=c,label=int(label), linestyle='dashed')
    # ax.plot(x,np.log10(np.exp((_a[0]*x))/_a[1]), c=c, 
    # label='%-5s | $a = %.1f$'%(int(label),_a[0]), linestyle='dashed')

    _y = np.log10(_N)
    if len(np.where(_y == -np.inf)[0]) > 0:
        _y[np.where(_y == -np.inf)[0].min()] = 0.
    ax2.step(binlims[1:], _y, where='pre', c=c)
    ax1.scatter(label,np.log10(_a[0]),color=c,s=15,marker='x',zorder=10)


ax2.legend(frameon=False, title='$\lambda_{\mathrm{rest}} \,/\, \mathrm{\mu m}=$')
# ax.text(0.05,0.9,'$a = %.2f$'%popt[0], transform=ax.transAxes, size=10)
ax2.set_xlim(0.,1.0)
ax2.set_ylim(-0.1,2.8)
ax2.set_xlabel('$D\,[\mathrm{Dimming}]$') # S_{i} \,/\, S_{i}^{\mathrm{max}}$')
ax2.set_ylabel('$\mathrm{log_{10}(Normalised \; frequency)}$')
plt.show()
# fname = 'plots/dimming_distribution.pdf'; print(fname)
# plt.savefig(fname,dpi=300,bbox_inches='tight'); plt.close()





# x = np.linspace(0,1,100)
# fig,ax = plt.subplots(1,1,figsize=(6,5))
# colors = [plt.cm.Set2(i) for i in range(len(a_array))]
# 
# for i,(_a,_N,label,c) in enumerate(zip(a_array,N_array,wavelengths,colors)):
#     ax.step(1 - binlims[1:], _N, where='pre', c=c)
# 
# plt.show()
