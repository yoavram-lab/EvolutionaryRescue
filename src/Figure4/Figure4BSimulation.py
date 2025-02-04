import mpmath as mp
import numpy as np
import numba
import seaborn as sns
from random import choices

sns.set_context('talk')

@numba.jit # 2-fold faster
def get_rates(s, a, m, λs, λa, λm, μs, μa, μm, u, v):
    return np.array([
        λs * s,    
        μs * s,     
        u * λs * s,
        v * λs * s,
        λa * a, 
        μa * a,
        v * λa * a,
        λm * m,    
        μm * m,
        0,
    ])
    
updates = np.array([
    [1, 0, 0],  # growth sensitive
    [-1, 0, 0], # death sensitive
    [0, 1, 0], #mutation s->a
    [0, 0, 1], #mutation s->m
    [0, 1, 0],  # growth aneuploidy type
    [0, -1, 0],  # death aneuploidy type
    [0, 0, 1], # #mutation a->m
    [0, 0, 1],  # growth mutant type
    [0, 0, -1],   # death mutant type
    [0, 0, 0]
])
    
def τ_leap(s, a, m, λs, λa, λm, μs, μa, μm, u, v, τ):

    rates = get_rates(s, a, m, λs, λa, λm, μs, μa, μm, u, v)
    
    try:
        adj_rates = np.random.poisson(rates * τ)
    except ValueError:
        print(rates, τ, s, a, m)
        raise
       
    adj_rates = np.random.poisson(rates * τ) 
    Δs, Δa, Δm = updates.T @ adj_rates
   
    return τ, Δs, Δa, Δm


def gillespie_ssa(λs, λa, λm, μs, μa, μm, u, v, s0, tmax, t0=0,  a0=0, m0=0, t_steps=1000):
    
    ΔT = (tmax-t0)/t_steps
        
    next_t = t0
    
    t = t0
    
    s, a, m = s0, a0, m0
    
    Δs, Δa, Δm = 0, 0, 0
    
    T = []
    S = []
    A = []
    M = []
    
    Lim = int(-3*np.log(10)/np.log(μm/λm)) + 1
    # loop over recording times
    while s + a + m > 0:
        while t < next_t:
            Δt, Δs, Δa, Δm = τ_leap(s, a, m, λs, λa, λm, μs, μa, μm, u, v, τ)
            t, s, a, m = t + Δt, max(s + Δs,0), max(a + Δa,0), max(m + Δm,0)
        # record the previous state for the time point we just passed
        S.append(s - Δs)
        A.append(a - Δa)
        M.append(m - Δm)
        next_t += ΔT
        T.append(next_t)
        
        if m > Lim : break
                
    if M[-1] == 0 : return 0
    if M[-1] > 0 : return 1

τ = 1/100

λs, λa, λm, μs, μa, μm, v = 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09, 10**(-7)

ds = λs - μs
da = λa - μa
dm = λm - μm
Nms = abs(ds)/(v*λs)*λm/dm

###############################################################################

def NaS(λs, λa, λm,  μs,  μa,  μm, u, v):
    
    ds = λs - μs
    da = λa - μa
    dm = λm - μm
    
    Ts = 1/np.sqrt(4*v*λa*λa*dm/λm)
    
    Ret = 0
    
    if da*Ts<-1:
        
        Ret = abs(da)/(v*λa)*λm/dm 
        
    if abs(da*Ts)<1:
        
        Ret = 2*λa*Ts
        
    if da*Ts>1:
        
        Ret = λa/da
        
    return abs(ds)/(u*λs)*Ret

###############################################################################
reps = 100
T = 1000

txt='ThresholdPopulationSizeVersusRatio.txt'
file = open(txt, "w")
    
for u in [10**(0), 10**(-1), 10**(-2), 10**(-3),10**(-4), 10**(-5)]:
    
    S = []
    Sum = []
    n = 0
    N = int(min(NaS(λs, λa, λm, μs, μa,  μm, u, v),Nms))
    
    while n < reps:
        S.append(gillespie_ssa(λs, λa, λm, μs, μa, μm, u, v, s0=N, tmax=T))
        n += 1
        print(n,N)
        
    Sum = S
    m = -1/N*mp.log(1-np.mean(Sum))
    print(m, np.mean(Sum))
    ### Bootstrapping ###
    M = []
    for i in range(reps):
        S_i = choices(S,k=len(S))
        M.append(1/(1 - mp.exp(mp.log(1 - np.mean(S_i))/N)))
    
    file.write(str(u/v)+' '+str(m)+' '+str(np.mean(M)-np.percentile(M, 2.5))+' '+str(np.percentile(M, 92.5)-np.mean(M))+' ')
        
file.close()