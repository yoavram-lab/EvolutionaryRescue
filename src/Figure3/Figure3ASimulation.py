import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.stats as stats
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

τ = 1/10

#Stationary aneuploids
λw, λa, λm, μw, μa, μm, u, v = 0.1, 0.09-10**(-5), 0.1, 0.14, 0.09, 0.09, 10**(-2), 10**(-7)
###############################################################################
reps = 100
T = 1000

fig, ax = plt.subplots()

#For tolerant aneuploids select λa=0.0899, txt='ProbvN.txt' and for N in [10**5,3*10**5,10**6,3*10**6,10**7,3*10**7,10**8]
#For resistant aneuploids select λa=0.095, txt='ProbvNgg0.txt' and for N in [10**1,10**2,10**3,10**4,10**5]
#For no aneuploids select u=0, txt='ProbvNnoAneuploidy.txt' and for N in [10**5,3*10**5,10**6,3*10**6,10**7,3*10**7,10**8]

txt='ProbvNee0.txt'
file = open(txt, "w")
    
for N in [10**3,10**4,10**5,10**6,10**7]:

    S = [gillespie_ssa(λw, λa, λm, μw, μa, μm, u, v, w0=N, tmax=T) for _ in range(reps)] # accelerate with multiprocessing
    
    file.write(str(N)+' '+str(np.mean(S))+' '+str(stats.sem(S))+' ')
    
file.close()