import mpmath as mp
import numpy as np
import numba
import seaborn as sns
sns.set_context('talk')
from random import choices

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


def gillespie_ssa(λs, λa, λm, μs, μa, μm, u, v, s0, a0, tmax, t0=0, m0=0, t_steps=1000):
    
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

τ = 1/10

λs, λa, λm, μa, μm, u, v = 0.1, 0.0899, 0.1, 0.09, 0.09, 10**(-2), 10**(-7)
c, us = 0.07, 10**(-3)

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

def NaS2(λs, λa, λm,  μs,  μa,  μm, u, v):
    
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
        
    return Ret
###############################################################################
reps = 100
T = 1000

#fig, ax = plt.subplots()

txt='DataGenerationFractionAneuploidy.txt'
file = open(txt, "w")

da = λa - μa
dm = λm - μm
f = (us*λs)/abs(c)

for μs in [0.11,0.13,0.15,0.17]:
    
    ds = λs - μs
    Nms = abs(ds)/(v*λs)*λm/dm
    
    Sum2 = []
    n = 0
    F = (1-f)*(u*λs)/abs(ds)+f
    N1 = int(min(NaS(λs, λa, λm, μs, μa, μm, u, v),Nms))
    N2 = int(NaS2(λs, λa, λm, μs, μa, μm, u, v))
    
    while n < reps:
        Sum2.append(gillespie_ssa(λs, λa, λm, μs, μa, μm, u, v, w0=0, a0=N2, tmax=T))
        n += 1
        
    m2 = -1/(N2/f)*mp.log(1-np.mean(Sum2))
    
    ########
    Sum1 = []
    n = 0
    
    while n < reps:
        Sum1.append(gillespie_ssa(λs, λa, λm, μs, μa, μm, u, v, w0=N1, a0=0, tmax=T))
        n += 1
               
    m1 = -1/N1*mp.log(1-np.mean(Sum1))    
    
    
    M = []
    for i in range(reps):
        S_1 = choices(Sum1,k=len(Sum1))
        S_2 = choices(Sum2,k=len(Sum2))

        M1 = -1/N1*mp.log(1-np.mean(S_1))
        M2 = -1/(N2/f)*mp.log(1-np.mean(S_2))
        M.append(M1/M2)
    
    file.write(str(λs-μs)+' '+str(m1/m2)+' '+str(m1/m2-np.percentile(M, 2.5))+' '+str(np.percentile(M, 92.5)-m1/m2)+' ')

file.close()


