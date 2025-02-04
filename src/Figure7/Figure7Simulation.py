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
        u * s * λs,
        v * s * λs,
        λa * a, 
        μa * a,
        v * a * λa,
        λm * m,    
        μm * m,
        0,
    ])
    
updates = np.array([
    [1, 0, 0],  # growth sensitive
    [-1, 0, 0], # death sensitive
    [-1, 1, 0], #mutation s->a
    [-1, 0, 1], #mutation s->m
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

    Δs, Δa, Δm = updates.T @ adj_rates
   
    return τ, Δs, Δa, Δm

def gillespie_ssa(λs, λa, λm, μs, μa, μm, u, v, s0, tmax, t0=0,  a0=0, m0=0, t_steps=1000):
    
    t = t0
    s, a, m = s0, a0, m0
    Δs, Δa, Δm = 0, 0, 0
    
    while s + a + m > 0: 
              
        #print(n,a,m)      
        Δt, Δs, Δa, Δm = τ_leap(s, a, m, λs, λa, λm, μs, μa, μm, u, v, τ)
        t, s, a, m = t + Δt, max(s + Δs, 0), max(a + Δa, 0), max(m + Δm, 0)
            
        if m>=s0: break          
                            
    if m == 0: return 0
    if m >= s0: return t


λs, λa, λm, μs, μa, μm, u, v = 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09, 10**(-2), 10**(-7)

τ = 1/10
###############################################################################
reps = 100
T = 1000

fig, ax = plt.subplots()

txt='ProliferationTime.txt'
file = open(txt, "w")

num = []
mum = []
eum = []

for N in [10**4,10**5,10**6,10**7,3*10**7,10**8,3*10**8,10**9]:
    file.write(str(N)+' ')
    
    nS = 0
    S = []

    while nS < reps:
        print(N,nS)
        x = gillespie_ssa(λs, λa, λm,  μs,  μa,  μm, u, v, s0=N, tmax=T)
        
        if x > 0: 
            
            nS+=1
            S.append(x)
            
    for s in S:
        file.write(str(s)+' ')
 
    file.write(' ')
    
file.close()
