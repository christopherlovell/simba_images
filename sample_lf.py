import numpy as np
from scipy.integrate import quad

import matplotlib.pyplot as plt



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
        n2 = self.binPhi(D_lowlim, inf_lim)*volume
    
        # --- Not strictly correct but I can't think of a better approach
        n = np.random.poisson(volume * cdf[-1])
        ncdf = cdf/cdf[-1]
        D_sample = np.interp(np.random.random(n), ncdf, D)
    
        return D_sample


def calc_phi(S,volume):
    _n,binlims = np.histogram(S)
    bins = binlims[:-1] + (binlims[1:] - binlims[:-1])/2
    return (_n/volume)/(binlims[1] - binlims[0]), bins

V = 2e2
model = Schechter(Dstar=np.log10(3.7), alpha=-1.4, log10phistar=3)
S = 10**model.sample(volume=V, D_lowlim=-1, inf_lim=3)
_phi,bins = calc_phi(np.log10(S),V)
plt.plot(bins, np.log10(_phi))


## from exponential fit script
a,b = 4.34552217e+01, 3.43867717e+16

def inv_cdf(x, A=43.455):
    c = np.log(x * (np.exp(A-1) - np.exp(-1)))
    return (c + 1) / A

orientation = inv_cdf(np.random.rand(len(S)), A=60)
new_S = S * orientation

_phi,bins = calc_phi(np.log10(new_S),V)
plt.plot(bins, np.log10(_phi))

plt.show()


edge_on = orientation < np.quantile(orientation,0.1)
_p0 = (np.sum(edge_on)/len(S))
print('edge on fraction:%.4f'%_p0)

slim = 0.3
S_selection = new_S > slim
_p1 = (np.sum(S_selection & edge_on) / np.sum(S_selection))
_p2 = (np.sum((S > slim) & edge_on) / np.sum(S > slim))
print('fraction of S selection edge on:%.4f'%_p1)

print('percentage reduction in edge-on galaxies in selection:%.4f'%(1 - _p1 / _p2))


## do for a grid
A_array = np.linspace(30,60,50)
Slim_array = np.logspace(-0.3,1.0,70)

xv, yv = np.meshgrid(A_array, Slim_array)

_out = np.zeros((len(Slim_array), len(A_array)))


for i in range(xv.shape[0]):
    for j in range(yv.shape[1]):

        orientation = inv_cdf(np.random.rand(len(S)), A=xv[i,j])
        new_S = S * orientation
        edge_on = orientation < np.quantile(orientation,0.1)

        S_selection = new_S > yv[i,j]
        
        _p1 = np.sum(S_selection & edge_on)
        _p2 = np.sum((S > yv[i,j]) & edge_on)
        
        _out[i,j] = (_p2 - _p1) / _p2
        
        # print("%.4f"%_out[i,j], "%.4f"%_p1, "%.4f"%_p2, 
        #       "%.2f"%yv[i,j], "%.2f"%xv[i,j])




def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

plt.imshow(_out * 100, aspect='auto', interpolation='none',
           extent=extents(A_array) + extents(Slim_array))#, origin='lower')
plt.colorbar(label='$b$ (-% difference in fraction \n of edge-on galaxies)')
plt.xlabel('$a$')
plt.ylabel('$S_{\mathrm{lim}}$')
plt.vlines(a,Slim_array.min(),Slim_array.max(),color='red', linestyle='dotted')

plt.show()
# fname = 'plots/orientation_grid.png'
# print(fname)
# plt.savefig(fname, dpi=300, bbox_inches='tight')


