import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import seaborn as sns
from scipy.optimize import fsolve

sns.set_context('talk')


# s = sensitive
# a = aneuploidy
# m = mutant

# Proliferation time
def proliferationtime(n, λm, μm):
    
    dm = λm - μm
    
    return np.log(n*dm)/dm

# Evolutionary rescue time
def EvolutionaryRescueTime(n, λs, λa, λm, μs, μa, μm, u, v):
    
    pm = (λm - μm)/λm
        
    pa = (λa - μa - v*λa + np.sqrt((λa - μa - v*λa)**2 + 4*λa**2*v*pm))/(2*λa)
    
    ps = (λs - μs - u*λs - v*λs + np.sqrt((λs - μs - u*λs - v*λs)**2 + 4*λs**2*(u*pa + v*pm)))/(2*λs)
       
    ds = λs - μs
    da = λa - μa
        
    return mp.quad(lambda t: mp.exp(v*n*λs*pm*(1 - mp.exp((ds)*t))/(ds) - u*v*λs*λa*pm*n/(ds - da)*((mp.exp((ds)*t)-1)/(ds) - (mp.exp(da*t) - 1)/da))/(1 - (1 - ps)**n), [0, 10000])

# Expected size of the aneuploid population at time t
def aneuploidypop(t, n, λs, λa, λm, μs, μa, μm, u, v):
    
    ds = λs - μs
    da = λa - μa
    
    return n*u*λs/(ds-da)*(mp.exp(ds*t)-mp.exp(da*t))

# probability density function of Gumbel distribution
def gumbel(t, n, λs, λa, λm, μs, μa, μm, u, v):
    
    dm = λm - μm
    pm = dm/λm
    
    return mp.exp(-pm*n*mp.exp(-dm*t))


def CDFcode(t, n, λs, λa, λm, μs, μa, μm, u, v):
    
    dm = λm - μm
    pm = dm/λm
    
    return v*λa*pm*mp.quad(lambda s: aneuploidypop(s, n, λs, λa, λm, μs, μa, μm, u, v)*gumbel(t-s, n, λs, λa, λm, μs, μa, μm, u, v), [0, t])
    
# roots of this function are solutions of equation D3 
def func(x, λs, λa, λm, μs, μa, μm, u, v):
    
    ds = λs - μs - λs*u - λs*v
    da = λa - μa
    dm = λm - μm

    return  u*v*λs*λa/(ds-da)*((np.exp(ds*x)-np.exp(dm*x))/(ds-dm)-(np.exp(da*x)-np.exp(dm*x))/(da-dm)) + v*λs*(np.exp(ds*x)-np.exp(dm*x))/(ds-dm)-1
        
def func1(x):
    
    ds = λs - μs - λs*u - λs*v
    da = λa - μa
    dm = λm - μm

    return  u*v*λs*λa/(ds-da)*((np.exp(ds*x)-np.exp(dm*x))/(ds-dm)-(np.exp(da*x)-np.exp(dm*x))/(da-dm)) + v*λs*(np.exp(ds*x)-np.exp(dm*x))/(ds-dm)-1
        
# 1-Heaviside function
def Heaviside(x,tau):
    
    return 1 - np.heaviside(x-tau, 1)

# Model parameters
λs, λa, λm, μs, μa, μm, u, v = 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09, 10**(-2), 10**(-7)



Time = np.geomspace(1,10**5,200)

fig = plt.figure(frameon=False)
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_xscale('log')

N = 10**6

CDF = [mp.exp(-CDFcode(t, N, λs, λa, λm, μs, μa, μm, u, v)) for t in Time]

plt.plot(Time, CDF, color='g', linewidth = 2.0, label=r'$N=10^6$')

N = 10**7

tau = float(proliferationtime(N, λm, μm) + EvolutionaryRescueTime(N, λs, λa, λm, μs, μa, μm, u, v))

def cdf(x,tau):
    return 1-np.exp(-x/tau)

CDF = [mp.exp(-CDFcode(t, N, λs, λa, λm, μs, μa, μm, u, v)) for t in Time]

plt.plot(Time, CDF, color='r', linewidth = 2.0, label=r'$N=10^7$')



###############################################################################
# Finding the root of equation D3    
root = fsolve(func1, 1)[0]
    
HEAVISIDE = [Heaviside(t,root) for t in Time]

plt.plot(Time, HEAVISIDE, color='b', linewidth = 2.0, label=r'$N=10^{10}$')

###############################################################################

# Plot of stochastic simulations

txt=str("KaplanMeierDatasmallN.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]

for j in range(len(lines)-1):
    A.append(float(lines[j]))
    
C = []

for t in [1,3,10,3*10,100,3*100,1000,3*1000,10000,3*10000,100000]:
    
    B = []
    for x in A:
        if x<t and x>0:
            B.append(x)
    
    C.append(1-len(B)/100)
      
D = []
for x in C:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
       
A = [1,3,10,3*10,100,3*100,1000,3*1000,10000,3*10000,100000]
ax.errorbar(A, C, D, fmt = 'ro', alpha=0.5, label='')

###############################################################################

txt=str("KaplanMeierDataverysmallN.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]

for j in range(len(lines)-1):
    A.append(float(lines[j]))
    
C = []

for t in [1,3,10,3*10,100,3*100,1000,3*1000,10000,3*10000,100000]:
    
    B = []
    for x in A:
        if x<t and x>0:
            B.append(x)
    
    C.append(1-len(B)/100)
      
D = []
for x in C:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
       
A = [1,3,10,3*10,100,3*100,1000,3*1000,10000,3*10000,100000]
ax.errorbar(A, C, D, fmt = 'go', alpha=0.5, label='')

###############################################################################

txt=str("KaplanMeierDatalargeN.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]

for j in range(len(lines)-1):
    A.append(float(lines[j]))
    
C = []

for t in [1,3,10,3*10,100,3*100,1000,3*1000,10000,3*10000,100000]:
    
    B = []
    for x in A:
        if x<t and x>0:
            B.append(x)
    
    C.append(1-len(B)/100)
      
D = []
for x in C:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
       
A = [1,3,10,3*10,100,3*100,1000,3*1000,10000,3*10000,100000]
ax.errorbar(A, C, D, fmt = 'bo', alpha=0.5, label='')

###############################################################################

plt.xlabel(r'Time,  $t$ (days)',fontsize=13)
ax.set_ylabel('Probability that mutant population\n has not reached size $N$ at time $t$\n$P(m_t\leq N)$',fontsize=13)

plt.yticks([0,0.25,0.5,0.75,1], [r'$0$',r'$0.25$',r'$0.5$',r'$0.75$',r'$1$'],fontsize=14)

plt.legend(loc=3,fontsize=14,frameon=False)
plt.tight_layout()

plt.savefig('ProliferationTimeCDFN.pdf')
plt.show()