import numpy as np

from scipy.stats import truncnorm, binned_statistic

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator

from simba import Schechter


def calc_phi(S,volume,binlimits=None):
    if binlimits is None:
        _n,binlims = np.histogram(S)
    else:
        _n,binlims = np.histogram(S,bins=binlimits)

    bins = binlims[:-1] + (binlims[1:] - binlims[:-1])/2
    _y = (_n/volume)/(binlims[1] - binlims[0])
    return _y, bins


## sample from number count distribution
wl_observer = 250
V = 50
# a = 9.58
model = Schechter(Dstar=1.50, alpha=-1.91, log10phistar=3.56)  # 250 mu-metre
# model = Schechter(Dstar=1.46, alpha=-2.31, log10phistar=3.17)  # 350 mu-metre
S = 10**model.sample(volume=V, D_lowlim=0, inf_lim=3)

## sample from redshift distribution
_mean = 2; _std = 1.0
clip_a = 0; clip_b = 5
_a, _b = (clip_a - _mean) / _std, (clip_b - _mean) / _std
zeds = truncnorm.rvs(_a, _b, loc=_mean, scale=_std, size=len(S))

wl_rest = wl_observer / (1+zeds)





def f(x,p1,p2,p3,p4):
    return p1*x**3 + p2*x**2 + p3*x + p4

_pa = [ 8.86550033e-09, -1.80296967e-05,  1.16724363e-02,  7.38651389e-02]
# _pa = [2.40195197e-10, -5.57233507e-06, 6.84216592e-03, 5.27412031e-01] 
# _pa = [ 7.25462147e-09, -1.54154045e-05,  1.05483936e-02,  1.49646973e-01]
# _pb = [-5.28000446e-09,  1.33246908e-05, -1.08837687e-02, -1.04244066e+00]

a = 10**f(wl_rest,*_pa)
# b = 10**f(wl_rest,*_pb)

### sample from dimming distribution
#def inv_cdf(x, A=43.455):
#    c = np.log(x * (np.exp(A) -1)) / -A
#    # c = ((np.log(x * (np.exp(A-1) - np.exp(-1)))) + 1) / A
#    # while np.sum(c < 0.5) > 0:
#    #     x = np.random.rand(np.sum(c < 0.5))
#    #     c[c<0.5] = ((np.log(x * (np.exp(A-1) - np.exp(-1)))) + 1) / A
#
#    return c
# 
# dimming = inv_cdf(np.random.rand(len(S)), A=a) 

dimming = np.random.exponential(scale=1/a, size=len(a))
max_dim = 0.9
while np.sum(dimming > max_dim) > 0:
    dimming[dimming > max_dim] = \
            np.random.exponential(scale=1/a[dimming > max_dim], size=np.sum(dimming > max_dim))


new_S = S * (1-dimming)

# _a = 9.579 # change for wavelength
# orientation = 1 - inv_cdf(np.random.rand(len(S)), A=_a)
# new_S = S * (1 - orientation)

binlimits = np.linspace(0,2.3,40)
db = (binlimits[1] - binlimits[0])
bins = binlimits[:-1] + db/2 
_phi = np.log10([model.binPhi(b1,b2)/(b2-b1) for b1,b2 in \
                 zip(binlimits[:-1],binlimits[1:])])
plt.plot(bins, _phi, label='model')


_phi,bins = calc_phi(np.log10(S),V,binlimits)
plt.plot(bins, np.log10(_phi), label='binned')


plt.legend()
plt.show()



## plot dimming as a function of redshift
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(5,12))
plt.subplots_adjust(hspace=0.)

binlims = np.linspace(0,5,10)
bins = binlims[:-1] + np.diff(binlims)[0]/2
dim_dist = [binned_statistic(zeds,dimming,statistic=lambda y: np.percentile(y, p), bins=binlims)[0] \
        for p in [16,50,84]]

_im = ax2.hexbin(zeds, dimming, gridsize=(40,30), cmap='Blues')
cax = fig.add_axes([0.15, 0.45, 0.02, 0.14])
fig.colorbar(_im, cax=cax, label='$N$')
ax2.errorbar(bins, dim_dist[1], yerr=[dim_dist[1] - dim_dist[0],dim_dist[2] - dim_dist[1]], 
             linestyle='none', color='grey', marker='o', 
             markeredgewidth=1, markeredgecolor='black')

ax2.set_xlabel('$z$'); ax2.set_ylabel('$D \;[\mathrm{Dimming}]$'); ax2.set_ylim(0,0.48)
ax2.set_xticklabels([])

## plot redshift distribution for different selections
slim = 30
binlims = np.linspace(0,5,30); bins = binlims[:-1] + np.diff(binlims)[0]/2
ax1.hist(zeds, histtype='step', density=True, bins=binlims,
        label='$\mathrm{All} \; (\mu = %i; \sigma=%i)$'%(_mean,_std))
ax1.hist(zeds[S > slim], histtype='step', density=True, label='$S > S_{\mathrm{lim}}$', bins=binlims)
ax1.hist(zeds[new_S > slim], histtype='step', density=True, 
         label='$S_{\mathrm{dimmed}} > S_{\mathrm{lim}}$', bins=binlims)
ax1.legend()
#ax1.set_xticklabels('')
ax1.xaxis.tick_top(); ax1.set_xlabel('$z$'); ax1.xaxis.set_label_position('top') 
ax1.set_ylabel('$N(\mathrm{normalised})$')
# ax1.text(0.03, 0.94, '$\mu = %.1f, \sigma=%.1f$'%(_mean,_std), transform=ax1.transAxes)
ax1.text(0.73, 0.64, '$S_{\mathrm{lim}} = %.1f$'%(slim), transform=ax1.transAxes)

for ax in [ax1,ax2,ax3]: ax.grid(alpha=0.4)
for ax in [ax1,ax2,ax3]: ax.set_xlim(0,5)

## plot completeness as a function of redshift
slims = np.array([4,8,16,32,64])# slims = np.array([10,15,20,25,30,35])
cm = plt.get_cmap('cividis') 
cNorm  = matplotlib.colors.Normalize(vmin=slims.min(), vmax=slims.max())
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)

for slim in slims:
    binlims = np.linspace(0,5,10); bins = binlims[:-1] + np.diff(binlims)[0]/2
    _dist = binned_statistic(zeds,(new_S,S), statistic=lambda y: np.sum(y > slim), bins=binlims)[0]
    ax3.plot(bins,_dist[0]/_dist[1],label='$S_{lim} = %.1f$'%slim, color=scalarMap.to_rgba(slim))

ax3.set_ylim(0.25,0.999)
ax3.set_xlabel('z'); ax3.set_ylabel('Completeness')
cax2 = fig.add_axes([0.25, 0.13, 0.5, 0.015])
cbar = fig.colorbar(scalarMap, label='$S_{\mathrm{lim}}$', cax=cax2, orientation='horizontal')
cax2.xaxis.set_label_position('top')
cax2.xaxis.set_ticks_position('top')
plt.show()
# plt.savefig('plots/redshift_completeness.pdf',dpi=200,bbox_inches='tight'); plt.close()


## from exponential fit script
#a,b = 4.34552217e+01, 3.43867717e+16


# orientation = inv_cdf(np.random.rand(len(S)), A=a)
# new_S = S * orientation
# 
# _phi,bins = calc_phi(np.log10(new_S),V)
# plt.plot(bins, np.log10(_phi))
# 
# plt.show()
# 
# 
# edge_on = orientation < np.quantile(orientation,0.1)
# _p0 = (np.sum(edge_on)/len(S))
# print('edge on fraction:%.4f'%_p0)
# 
# slim = 0.3
# S_selection = new_S > slim
# _p1 = (np.sum(S_selection & edge_on) / np.sum(S_selection))
# _p2 = (np.sum((S > slim) & edge_on) / np.sum(S > slim))
# print('fraction of S selection edge on:%.4f'%_p1)
# 
# print('percentage reduction in edge-on galaxies in selection:%.4f'%(1 - _p1 / _p2))

## ---- plot orientation distribution
Ns = 5
Slim_array = np.array([4,8,16,32,64])  
# np.array([5e-1,1,2,4,8]) # np.logspace(-0.3,1.0,Ns)
binlimits = np.linspace(0,1.0,21)
bins = binlimits[1:] - ((binlimits[1] - binlimits[0])/2)

cmap = plt.cm.get_cmap('cividis', len(Slim_array))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Slim_array.min(), vmax=Slim_array.max()))
sm.set_clim(0,Ns)

fig, ax = plt.subplots(1,1,figsize=(6,5))

for i, slim in enumerate(Slim_array):
    _N = list(np.histogram(dimming[S > slim], bins=binlimits)[0])
    _NB = list(np.histogram(dimming[new_S > slim], bins=binlimits)[0])

    _N.append(_N[-1])
    _NB.append(_NB[-1])

    ax.step(binlimits, np.array(_NB) / np.array(_N), color=cmap(i/Ns), where='post')


ax.set_xlim(0,1.0)
ax.set_ylim(0,1)
ax.set_xlabel('$D$ [Dimming]')
ax.set_ylabel('$N_{\mathrm{dim}} \,/\, N$')
ax.grid(alpha=0.2)
cbar = fig.colorbar(sm)
cbar.set_ticks(np.arange(Ns)+0.5)
cbar.set_ticklabels(Slim_array)
cbar.set_label('$S_{\mathrm{lim}}$')
plt.show()
# fname = 'plots/dimming_fraction.pdf'; print(fname)
# plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close()


## ---- plot luminosity function with completeness (350 mu-metre)
fig, ax = plt.subplots(1,1, figsize=(6,5))

cmap = plt.cm.get_cmap('viridis')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))#LogNorm(vmin=1e-2, vmax=1))

binlimits = np.linspace(-0.3,2.3,180)
#binlimits = np.hstack([binlimits,np.linspace(1.8,2.3,20)])

_phi,bins = calc_phi(np.log10(new_S), V, binlimits=binlimits)
ax.plot(10**bins, np.log10(_phi), label='un-dimmed', color='black', linestyle='dashed')

_phi,bins = calc_phi(np.log10(S), V, binlimits=binlimits)
ax.plot(10**bins, np.log10(_phi), label='un-dimmed', color='red', linestyle='dashed')

ax.set_xscale('log')

slim = binlimits[100] # [95] # [80]
_N = np.histogram(np.log10(S[new_S > 10**slim]), bins=binlimits[binlimits>=slim])[0]
_NT = np.histogram(np.log10(S[S > 10**slim]), bins=binlimits[binlimits>=slim])[0]

for _c,_bl in zip(_N/_NT,binlimits[binlimits>=slim]):
    _x = np.linspace(_bl,_bl+np.diff(binlimits)[0],2)
    _bins = _x[1:] - ((_x[1] - _x[0])/2)
    _y = np.log10(calc_phi(np.log10(new_S),V,binlimits=_x)[0])
    ax.fill_between(10**_x, np.hstack([_y,_y]), y2=-1, color=cmap(_c))

y_upp = 4.3 #4.6 # 4.3
ax.set_ylim(2.6,y_upp) # (1,y_upp)
ax.set_xlim(10,80) # (5,90) # (8,80)

_x_completeness = 10**binlimits[binlimits>=slim][np.min(np.where((_N/_NT) > 0.95))]
ax.text(_x_completeness*1.05, y_upp*0.91, '$S_{95} = %.2f \, \mathrm{mJy}$'%(_x_completeness))
ax.vlines(_x_completeness, -1, y_upp*0.94, linestyle='dotted', color='black')
ax.text(10**slim * 1.05, y_upp * 0.96, '$S_{\mathrm{lim}} = %.2f \, \mathrm{mJy}$'%10**slim)
ax.vlines(10**slim, -1, y_upp, linestyle='-.', color='black')

cbar = fig.colorbar(sm)
cbar.set_label('Completeness')

formatter = ScalarFormatter()
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_minor_formatter(formatter)

ax.set_xlabel('$\mathrm{log_{10}}(S \,/\, \mathrm{mJy})$')
ax.set_ylabel('$\phi \,/\, (\mathrm{deg^{-2} \; dex^{-1}})$')
ax.text(0.7,0.9,'$\lambda_{\mathrm{obs}} = 250 \, \mathrm{\mu m}$',transform=ax.transAxes)

plt.show()
# fname = 'plots/lf_completeness.pdf'; print(fname)
# plt.savefig(fname, dpi=300, bbox_inches='tight')


# ## ---- plot luminosity function with completeness
# fig, ax = plt.subplots(1,1, figsize=(6,5))
# 
# cmap = plt.cm.get_cmap('viridis')
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))#LogNorm(vmin=1e-2, vmax=1))
# 
# binlimits = np.linspace(-0.3,2.3,79)
# _phi,bins = calc_phi(np.log10(S),V,binlimits=binlimits)
# ax.plot(10**bins, np.log10(_phi), label='un-dimmed', color='black', linestyle='dashed')
# 
# ax.set_xscale('log')
# 
# slim = binlimits[44]
# _N = np.histogram(np.log10(S[new_S > 10**slim]), bins=binlimits[binlimits>=slim])[0]
# _NT = np.histogram(np.log10(S[S > 10**slim]), bins=binlimits[binlimits>=slim])[0]
# 
# for _c,_bl in zip(_N/_NT,binlimits[binlimits>=slim]):
#     _x = np.linspace(_bl,_bl+np.diff(binlimits)[0],10)
#     _bins = _x[1:] - ((_x[1] - _x[0])/2)
#     ax.fill_between(10**_bins, np.log10(calc_phi(np.log10(S),V,binlimits=_x)[0]), 
#                     y2=-1, color=cmap(_c))
# 
# y_upp = 4.3
# ax.set_ylim(3,y_upp)
# ax.set_xlim(8,80)
# 
# _x_completeness = 10**binlimits[binlimits>=slim][np.min(np.where((_N/_NT) > 0.95))]
# ax.text(_x_completeness*1.05, y_upp*0.96, '$S_{95} = %.2f \, \mathrm{mJy}$'%(_x_completeness))
# ax.vlines(_x_completeness, -1, y_upp*0.97, linestyle='dotted', color='black')
# ax.text(10**slim * 1.05, y_upp * 0.99, '$S_{\mathrm{lim}} = %.2f \, \mathrm{mJy}$'%10**slim)
# ax.vlines(10**slim, -1, y_upp, linestyle='-.', color='black')
# 
# cbar = fig.colorbar(sm)
# cbar.set_label('Completeness')
# 
# formatter = ScalarFormatter()
# formatter.set_scientific(False)
# ax.xaxis.set_major_formatter(formatter)
# ax.xaxis.set_minor_formatter(formatter)
# 
# ax.set_xlabel('$\mathrm{log_{10}}(S \,/\, \mathrm{mJy})$')
# ax.set_ylabel('$\phi \,/\, (\mathrm{deg^{-2} \; dex^{-1}})$')
# 
# plt.show()
# # fname = 'plots/lf_completeness.png'; print(fname)
# # plt.savefig(fname, dpi=300, bbox_inches='tight')



## ---- completeness as a function of flux density limit
fig, ax = plt.subplots(1,1,figsize=(6,5))

Slim_array = np.linspace(0.3,2.0,20)
binlimits = np.linspace(0.3,2.3,79)

wavelengths = [250,350,500,850]
a_array = [9.579,17.24,26.12,42.37]

for _a,label in zip(a_array,wavelengths):
    orientation = 1 - inv_cdf(np.random.rand(len(S)), A=_a)
    new_S = S * (1 - orientation)

    completeness = np.zeros(len(Slim_array))
    for i,slim in enumerate(Slim_array):
        _N = np.histogram(np.log10(S[new_S > 10**slim]), bins=binlimits[binlimits>=slim])[0]
        _NT = np.histogram(np.log10(S[S > 10**slim]), bins=binlimits[binlimits>=slim])[0]
        if len(_N) > 0:
            completeness[i] = binlimits[binlimits>=slim][np.min(np.where((_N/_NT) > 0.99))]
        
    ax.plot(Slim_array, completeness, label='$\lambda = %s \; \mathrm{\mu m}$'%label)


ax.plot(Slim_array, Slim_array, linestyle='dashed', color='black')
ax.set_xlabel('$\mathrm{log_{10}}(S_{\mathrm{lim}})$')
ax.set_ylabel('$\mathrm{log_{10}}(S_{99})$')
ax.set_xlim(Slim_array.min(), Slim_array.max())
ax.set_ylim(Slim_array.min(), Slim_array.max() + 0.3)
ax.grid(alpha=0.2)
ax.legend()

plt.show() 
# fname = 'plots/slim_completeness.png'; print(fname)
# plt.savefig(fname, dpi=300, bbox_inches='tight')




## ---- plot $b$ as a function of Slim
Slim_array = np.logspace(0.3,2.0,20)
a_array = [9.579,17.24,26.12,42.37]
wavelengths = [250,350,500,850]

colors = [plt.cm.Set2(i) for i in range(len(a_array))]

fig, ax = plt.subplots(1,1,figsize=(5,4))

for _a,label,c in zip(a_array,wavelengths, colors):
    orientation = inv_cdf(np.random.rand(len(S)), A=_a)
    new_S = S * orientation
    edge_on = orientation < np.quantile(orientation,0.2)

    _out = np.zeros(len(Slim_array))
    for i, slim in enumerate(Slim_array):
        S_selection = new_S > slim
        _p1 = np.sum(S_selection & edge_on)
        _p2 = np.sum((S > slim) & edge_on)
        _out[i] = (_p2 - _p1)/_p2
    
    ax.plot(Slim_array, _out, label='$\lambda = %s \; \mathrm{\mu m}$'%label, 
            c=c, lw=3)


ax.set_xlabel('$S_{\mathrm{lim}} \,/\, \mathrm{mJy}$')
ax.set_ylabel('$b$')
ax.set_xlim(Slim_array.min(), Slim_array.max())
ax.set_ylim(0,1)
ax.legend(frameon=False)
ax.grid(alpha=0.2)

plt.show()
# fname = 'plots/orientation_fraction.png'; print(fname)
# plt.savefig(fname, dpi=300, bbox_inches='tight')



# ## ---- do for a grid
# A_array = np.linspace(2,15,20)
# Slim_array = np.logspace(-0.3,1.0,20)
# 
# xv, yv = np.meshgrid(A_array, Slim_array)
# 
# _out = np.zeros((len(Slim_array), len(A_array)))
# 
# 
# for i in range(xv.shape[0]):
#     for j in range(yv.shape[1]):
# 
#         orientation = inv_cdf(np.random.rand(len(S)), A=xv[i,j])
#         new_S = S * orientation
#         edge_on = orientation < np.quantile(orientation,0.1)
# 
#         S_selection = new_S > yv[i,j]
#         
#         _p1 = np.sum(S_selection & edge_on)
#         _p2 = np.sum((S > yv[i,j]) & edge_on)
#         
#         _out[i,j] = (_p2 - _p1) / _p2
#         
#         # print("%.4f"%_out[i,j], "%.4f"%_p1, "%.4f"%_p2, 
#         #       "%.2f"%yv[i,j], "%.2f"%xv[i,j])
# 
# 
# 
# 
# def extents(f):
#   delta = f[1] - f[0]
#   return [f[0] - delta/2, f[-1] + delta/2]
# 
# plt.imshow(_out * 100, aspect='auto', interpolation='none',
#            extent=extents(A_array) + extents(Slim_array))#, origin='lower')
# plt.colorbar(label='$b$ (-% difference in fraction \n of edge-on galaxies)')
# plt.xlabel('$a$')
# plt.ylabel('$S_{\mathrm{lim}}$')
# plt.vlines(a,Slim_array.min(),Slim_array.max(),color='red', linestyle='dotted')
# 
# plt.show()
# fname = 'plots/orientation_grid.png'
# print(fname)
# plt.savefig(fname, dpi=300, bbox_inches='tight')


