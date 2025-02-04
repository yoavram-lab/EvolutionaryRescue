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
    
    Δs, Δa, Δm = updates.T @ adj_rates
   
    return τ, Δs, Δa, Δm

def gillespie_ssa(λs, λa, λm, μs, μa, μm, u, v, s0, tmax, t0=0,  a0=0, m0=0, t_steps=1000):
    
    t = t0
    s, a, m = s0, a0, m0
    Δs, Δa, Δm = 0, 0, 0
    
    MT = []
    # loop over recording times
    while s + a + m > 0: 
                    
        Δt, Δs, Δa, Δm = τ_leap(s, a, m, λs, λa, λm, μs, μa, μm, u, v, τ)
        t, s, a, m = t + Δt, max(s + Δs, 0), max(a + Δa, 0),  max(m + Δm, 0)  
                    
        if m > s0: 
            MT.append(t)
            break          
                            
    if len(MT) == 0: return 0
    if len(MT) > 0: return min(MT)

N, λs, λa, λm, μs, μa, μm, u, v = 10**7, 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09, 10**(-2),10**(-7)

τ = 1/1000
###############################################################################
reps = 100
T = 1000

fig, ax = plt.subplots()

#Select txt='KaplanMeierDataverysmallN.txt' for N=10**6
#Select txt='KaplanMeierDatalargeN.txt' for N=10**10


txt='KaplanMeierDatasmallN.txt'
file = open(txt, "w")

num = []
mum = []
eum = []

Time = [gillespie_ssa(λs, λa, λm,  μs,  μa,  μm, u, v, w0=N, tmax=T) for i in range(reps)]

for t in Time: 
    print(t)
    file.write(str(t)+' ')

file.close()

plt.errorbar(num,mum,eum)
plt.show()


