#import matplotlib.inline
import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.stats as stats
from scipy.integrate import solve_ivp
import seaborn as sns
sns.set_context('talk')

from functools import partial

def plot_mp(t, n, a, m, label='', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(t, n, lw=3, label='wildtype ' + label)
    ax.plot(t, a, lw=3, label='aneuploidy ' + label)
    ax.plot(t, m, lw=3, label='mutant ' + label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.legend()
    return ax

@numba.jit # 2-fold faster
def get_rates(n, a, m, λn, λa, λm, μn, μa, μm, v, w):
    return np.array([
        λn * n,    
        μn * n,     
        v * n,
        w * n,
        λa * a, 
        μa * a,
        w * a,
        λm * m,    
        μm * m,
    ])
    
@numba.jit # 2-fold faster
def draw_time(rates):
 #   print(rates,'r')
    total_rate = rates.sum()
    
#    print(rates.sum(),'hello')
    if total_rate > 0:
        return np.random.exponential(1/total_rate)
    if total_rate == 0:
        return 1

# @numba.jit # jit causes errors with multinomial
def draw_reaction(rates):
    if rates.sum() > 0:
        rates /= rates.sum()
        return np.random.multinomial(1, rates).argmax()
    if rates.sum() == 0:
        return 9

#rates = get_rates(3, 100, βm, βp, γp)
#rates /= rates.sum()
    
updates = np.array([
    [1, 0, 0],  # growth wild type
    [-1, 0, 0], # death wild type
    [-1, 1, 0], #mutation w->a
    [-1, 0, 1], #mutation w->m
    [0, 1, 0],  # growth aneuploidy type
    [0, -1, 0],  # death aneuploidy type
    [0, -1, 1], # #mutation a->m
    [0, 0, 1],  # growth mutant type
    [0, 0, -1],   # death mutant type
    [0, 0, 0]
])
    
def gillespie_step(n, a, m, λn, λa, λm, μn, μa, μm, v, w):
    rates = get_rates(n, a, m, λn, λa, λm, μn, μa, μm, v, w)
    Δt = draw_time(rates)
    ri = draw_reaction(rates)
    Δn, Δa, Δm= updates[ri]
    return Δt, Δn, Δa, Δm , rates[0]


def gillespie_ssa(λn, λa, λm, μn, μa, μm, v, w, n0, tmax, t0=0,  a0=0, m0=0, t_steps=1000):
    
    ΔT = (tmax-t0)/t_steps
    i = 0
    next_t = t0
    # init
    info=0
    t = t0
    n, a, m = n0, a0, m0
    Δn, Δa, Δm = 0, 0, 0
    T = []
    N = []
    A = []
    M = []
    # loop over recording times
    while n + a + m > 0:
    #      i.append[+1]
    #      next_t.append[+dt]
#        for i, next_t in enumerate(times):
            # simulate until next recording time
        while t < next_t:
            Δt, Δn, Δa, Δm, info = gillespie_step(n, a, m, λn, λa, λm, μn, μa, μm, v, w)
            t, n, a, m = t+Δt, n+Δn, a+Δa, m+Δm
        # record the previous state for the time point we just passed
        #print(t,n,a,m,'important',n0, info, next_t)
        N.append(n - Δn)
        A.append(a - Δa)
        M.append(m - Δm)
        #states[:, i] = n - Δn, a - Δa, m - Δm
        i += 1
        next_t += ΔT
        T.append(next_t)
        
        if m > int(-3*np.log(10)/np.log(μm/λm)) + 1: break
            
    if M[-1]==0: return 0
    if M[-1]>0: return 1

λn, λa, λm, μn, μa, μm, v, w = 1-0.01, 1-0.001, 1+0.1, 1, 1, 1, 10**(-2), 10**(-7)

###############################################################################
reps = 50
T = 1000

fig, ax = plt.subplots()

txt='EvolutionaryRescueDataExponentialNeutralAneuploidy.txt'
file = open(txt, "w")

num = []
mum = []
eum = []

for N in [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]:
    
    S = [gillespie_ssa(λn, λa, λm,  μn,  μa,  μm, v, w, n0=N, tmax=T) for _ in range(reps)] # accelerate with multiprocessing

    print(2*np.mean(S),stats.sem(S),N)
    
    num.append(N)
    mum.append(2*np.mean(S))
    eum.append(stats.sem(S))
    
    file.write(str(N)+' '+str(2*np.mean(S))+' '+str(stats.sem(S))+' ')

file.close()

plt.errorbar(num,mum,eum)
plt.show()


