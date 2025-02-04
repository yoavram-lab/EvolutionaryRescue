import numpy as np
import matplotlib.pyplot as plt
import numba
import seaborn as sns

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
    
@numba.jit # 2-fold faster
def draw_time(rates):
    total_rate = rates.sum()
    
    if total_rate > 0:
        return np.random.exponential(1/total_rate)
    if total_rate == 0:
        return 1

def draw_reaction(rates):
    if rates.sum() > 0:
        rates /= rates.sum()
        return np.random.multinomial(1, rates).argmax()
    if rates.sum() == 0:
        return 9

    
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
    
def gillespie_step(s, a, m, λs, λa, λm, μs, μa, μm, u, v):
    rates = get_rates(s, a, m, λs, λa, λm, μs, μa, μm, u, v)
    Δt = draw_time(rates)
    ri = draw_reaction(rates)
    Δw, Δa, Δm= updates[ri]
    return Δt, Δw, Δa, Δm

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
    
    t = t0
    s, a, m = s0, a0, m0
    Δs, Δa, Δm = 0, 0, 0
    
    MT = []
    m4 = 0
    ok=0
    
    Lim = int(-3*np.log(10)/np.log(μm/λm)) + 1
    # loop over recording times
    while s + a + m > 0: 
                    
        Δt, Δs, Δa, Δm = τ_leap(s, a, m, λs, λa, λm, μs, μa, μm, u, v, τ)
        
        #print(n,a,m,'1',n0,Δa,Δm)
        
        if m == 0 and Δm > 0: 
            for i in range(Δm):                 
                next_t_4 = t
                t4, m4 = t + Δt, m + 1

                if ok==1: break
                while m4 > 0:             
                        Δt4, ΔN4, ΔA4, Δm4 = gillespie_step(0, 0, m4, λs, λa, λm, μs, μa, μm, u, v)#τ_leap(0, 0, m4, λn, λa, λm, μn, μa, μm, v, w, τ)
                        t4, m4 = t4 + Δt4, m4 + Δm4
                        if m4 > Lim:
                            MT.append(t + Δt)   
                            ok=1
                            break  
                    
                
                                
        t, s, a, m = t + Δt, max(s + Δs, 0), max(a + Δa, 0), 0
            
        if ok==1: break          
                            
    if len(MT) == 0: return 0
    if len(MT) > 0: return min(MT)

N, λs, λa, λm, μs, μa, μm, u, v = 10**7, 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09,  10**(-2), 10**(-7)

τ = 1/10
###############################################################################
reps = 100
T = 1000

fig, ax = plt.subplots()

txt='ReboundProbability.txt'
file = open(txt, "w")

#Select txt='ReboundProbabilityStationary.txt' for λa=0.089999
#Select txt='ReboundProbabilityLargeDa.txt' for λa=0.095
#Select txt='ReboundProbabilityNoAneuploidy.txt' for u=0


num = []
mum = []
eum = []

Time = [gillespie_ssa(λs, λa, λm,  μs,  μa,  μm, u, v, s0=N, tmax=T) for i in range(reps)]

for t in Time: 
    print(t)
    file.write(str(t)+' ')

file.close()