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

def funsum(n,a,m):
    
    if n+a+m==0:
        
        return 1
    
    if n+a+m>0:
        
        return n+a+m
        
        

@numba.jit # 2-fold faster
def get_rates(n, a, m, λn, λa, λm, μn, μa, μm, v, w):
    return np.array([
        λn * n,    
        μn * n,     
        v * n,
        w * n,
        (λa - μa*np.log(funsum(n,a,m)/K))*a, 
        λa * a,
        w * a,
        (λm - μm* np.log(funsum(n,a,m)/K))*m,    
        λm * m,
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
    times = np.linspace(t0, tmax, t_steps) # recording times: time points in which to record the state
    states = np.empty((updates.shape[1], t_steps), dtype=int) # recorded states
    
    # init
    info=0
    t = t0
    n, a, m = n0, a0, m0
    Δn, Δa, Δm = 0, 0, 0
    # loop over recording times
    for i, next_t in enumerate(times):
        # simulate until next recording time
        while t < next_t:
            Δt, Δn, Δa, Δm, info = gillespie_step(n, a, m, λn, λa, λm, μn, μa, μm, v, w)
            t, n, a, m = t+Δt, n+Δn, a+Δa, m+Δm
        # record the previous state for the time point we just passed
        print(t,n,a,m,'important',n0, info)
        states[:, i] = n - Δn, a - Δa, m - Δm
        #print(t,next_t,m,p,c1,c2)
    # return array equivalent to [[times, mRNA, protein] for t in times]
    return np.concatenate((times.reshape(1, -1), states), axis=0)

global K 

λn, λa, λm, μn, μa, μm, v, w, K = 1, 1, 1, 0.99, 0.01, 0.1, 10**(-2), 10**(-7), 200

###############################################################################
reps = 400
T = 400

fig, ax = plt.subplots()

txt='EvolutionaryRescueDataComplete_n'+str(100)+'_K'+str(K)+'.txt'
file = open(txt, "w")

num = []
mum = []
eum = []

for N in [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]:
    
    
    tmp = np.array([gillespie_ssa(λn, λa, λm,  μn,  μa,  μm, v, w, n0=N, tmax=T) for _ in range(reps)]) # accelerate with multiprocessing
    
    
    t = tmp[0, 0, :] # time is the same in all replicates, see first line of gillespie_ssa
    n = tmp[:, 1, :]
    a = tmp[:, 2, :]
    m = tmp[:, 3, :]
        
    S = []
    
    for x in a:
        if x[-1]==0:
            S.append(0)
        if x[-1]>0:
            S.append(1)

            
    for x in m:
        if x[-1]==0:
            S.append(0)
        if x[-1]>0:
            S.append(1)
            
    print(2*np.mean(S),stats.sem(S),N)
    
    num.append(N)
    mum.append(2*np.mean(S))
    eum.append(stats.sem(S))
    
    file.write(str(N)+' '+str(2*np.mean(S))+' '+str(stats.sem(S))+' ')

file.close()

plt.errorbar(num,mum,eum)
plt.show()


