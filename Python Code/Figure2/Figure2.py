#import matplotlib.inline
import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.stats as stats
from scipy.integrate import solve_ivp
import seaborn as sns
from statistics import variance

sns.set_context('talk')

from functools import partial

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
    
@numba.jit # 2-fold faster
def draw_time(rates):
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
    
def gillespie_step(s, a, m, λs, λa, λm, μs, μa, μm, u, v):
    rates = get_rates(s, a, m, λs, λa, λm, μs, μa, μm, u, v)
    Δt = draw_time(rates)
    ri = draw_reaction(rates)
    Δs, Δa, Δm= updates[ri]
    return Δt, Δs, Δa, Δm

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
    
    states = [t0,s0,a0,m0]
    
    t = t0
    s, a, m = s0, a0, m0
    Δs, Δa, Δm = 0, 0, 0
    
    # loop over recording times
    while s + a + m > 0: 
                    
        Δt, Δs, Δa, Δm = τ_leap(s, a, m, λs, λa, λm, μs, μa, μm, u, v, τ)
        t, s, a, m = t + Δt, max(s + Δs, 0), max(a + Δa, 0),  max(m + Δm, 0)
        
        states= np.concatenate((states,[t, s, a, m]),axis=0)
        
        if t > 500: break
    
    return states

λA = [0, 0.065, 0.08999, 0.095]
U = [0 , 10**(-2), 10**(-2), 10**(-2)]
Text = ['TauLeapMeanTimeDiagramNoAneuploidy.pdf','TauLeapMeanTimeDiagramSmallda.pdf','TauLeapMeanTimeDiagramdazero.pdf','TauLeapMeanTimeDiagramlargeda.pdf']

for i in {0,1,2,3}:
    
    λa, u = λA[i], U[i]
    
    #Model parameters
    λs, λm, μs, μa, μm, v = 0.1, 0.1, 0.14, 0.09, 0.09,10**(-7)
    #Sensitive fitness
    ds = λs - μs
    #Aneuploid fitness
    da = λa - μa
    
    τ = 1/10
    ###############################################################################
    reps = 1
    T = 1000
    
    num = []
    mum = []
    eum = []
    
    #Initial population size
    N = 10**7
        
    fig, ax = plt.subplots()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    
    
    ax.set_yscale('log')
    
    def linesstart(λs, λa, λm,  μs,  μa,  μm, u, v):
        
        tmp = gillespie_ssa(λs, λa, λm,  μs,  μa,  μm, u, v, s0=N, tmax=T) 
        
        A=[]
        B=[]
        C=[]
        D=[]
        
        for j in range(len(tmp)):
            if j%4==0:
                A.append(tmp[j])
            if j%4==1:
                B.append(tmp[j])
            if j%4==2:
                C.append(tmp[j])
            if j%4==3:
                D.append(tmp[j])
        
        plt.plot(A,B,color='g',alpha=0.25)
        plt.plot(A,C,color='b',alpha=0.25)
        plt.plot(A,D,color='r',alpha=0.25)
        
    def lines(λs, λa, λm,  μs,  μa,  μm, u, v):
        
        tmp = gillespie_ssa(λs, λa, λm,  μs,  μa,  μm, u, v, s0=N, tmax=T) 
        
        A=[]
        B=[]
        C=[]
        D=[]
        
        for j in range(len(tmp)):
            if j%4==0:
                A.append(tmp[j])
            if j%4==1:
                B.append(tmp[j])
            if j%4==2:
                C.append(tmp[j])
            if j%4==3:
                D.append(tmp[j])
        
        plt.plot(A,B,color='g',alpha=0.25)
        plt.plot(A,C,color='b',alpha=0.25)
        plt.plot(A,D,color='r',alpha=0.25)
        
    linesstart(λs, λa, λm,  μs,  μa,  μm, u, v) 
    [lines(λs, λa, λm,  μs,  μa,  μm, u, v) for _ in range(9)]
    
    A=[10**9,10**10]
    if i in {1}:
        plt.plot(A,A,color='g',alpha=1, label='sensitive')
        plt.plot(A,A,color='b',alpha=1, label='aneuploid')
        plt.plot(A,A,color='r',alpha=1, label='mutant')
    else:
        plt.plot(A,A,color='g',alpha=1, label='')
        plt.plot(A,A,color='b',alpha=1, label='')
        plt.plot(A,A,color='r',alpha=1, label='')
        
    
    if i in {2,3}:
        plt.xlabel(r'Time, $t$ (days)',fontsize=16)
        plt.ylabel(r'Number of cells',fontsize=16)
    
    plt.yticks([10**(0),10**(2),10**(4),10**(6),10**(8)], [r'$10^{0}$',r'$10^{2}$', r'$10^{4}$',r'$10^{6}$',r'$10^{8}$'],fontsize=16)
    plt.xlim([0,500])
    plt.ylim([1,10**8])
    
    
    plt.legend(loc=1,fontsize=14,frameon=False)
    
    plt.tight_layout()
    
    plt.savefig(Text[i])
    plt.show()
