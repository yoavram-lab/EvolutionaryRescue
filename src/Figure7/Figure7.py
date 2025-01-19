#import matplotlib.inline
#from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import seaborn as sns
from random import choices
sns.set_context('talk')

# s = sensitive
# a = aneuploidy
# m = mutant

def proliferationtime(n, λm, μm):
    
    dm = λm - μm
    
    return (np.log(n*dm/μm))/dm

def EvolutionaryRescueTime(n, λs, λa, λm, μs, μa, μm, u, v):
    
    pm = (λm - μm)/λm
        
    pa = (λa - μa - v*λa + np.sqrt((λa - μa - v*λa)**2 + 4*λa**2*v*pm))/(2*λa)
    
    ps = (λs - μs - u*λs - v*λs + np.sqrt((λs - μs - u*λs - v*λs)**2 + 4*λs**2*(u*pa + v*pm)))/(2*λs)
       
    ds = λs - μs
    da = λa - μa
    
    return mp.quad(lambda t: t*(v*n*λs*pm*mp.exp((ds)*t)+u*v*n*λs*λa*pm*(mp.exp((ds)*t)-mp.exp((da)*t))/(ds-da))*mp.exp(v*n*λs*pm*(1 - mp.exp((ds)*t))/(ds) - u*v*λs*λa*pm*n/(ds - da)*((mp.exp((ds)*t)-1)/(ds) - (mp.exp(da*t) - 1)/da))/(1 - (1 - ps)**n), [0, np.infty])


def func(x, λs, λa, λm, μs, μa, μm, u, v):
    
    ds = λs - μs 
    da = λa - μa
    dm = λm - μm

    return  u*v*λs*λa/(ds-da)*((np.exp(ds*x)-np.exp(dm*x))/(ds-dm)-(np.exp(da*x)-np.exp(dm*x))/(da-dm)) + v*λs*(np.exp(ds*x)-np.exp(dm*x))/(ds-dm)-1
    
#Model parameters
λs, λa, λm, μs, μa, μm, u, v = 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09, 10**(-2), 10**(-7)

d =  (λm - μm) - (λs - μs)
dm = λm - μm
proliferationtimelimit = (np.log(d/(v*λs)))/dm

X= np.linspace(0,2000,1000)
sol = 0
last = func(0, λs, λa, λm, μs, μa, μm, u, v)

for x in X:
    if func(x, λs, λa, λm, μs, μa, μm, u, v)*last < 0: 
        root = x
        break
    last = func(x, λs, λa, λm, μs, μa, μm, u, v)

NumberCells = []
MeanTime = []
txt=str("ProliferationTime.txt")
text_file = open(txt, "r")
lines = text_file.read().split("  ")
lower_error = []
upper_error = []
for I in range(len(lines)-1):
    line = lines[I]
    linestwo = line.split(" ")
    if float(linestwo[0])>=10**7:
        NumberCells.append(float(linestwo[0]))
        S = []
        for i in range(1,len(linestwo)):
            S.append(float(linestwo[i]))
        MeanTime.append(np.mean(S))
        
        M = []
        for i in range(100):
            S_i = choices(S,k=100)
            M.append(np.mean(S_i))
            
        lower_error.append(np.mean(S)-np.percentile(M, 2.5))    
        upper_error.append(np.percentile(M, 92.5)-np.mean(S))  
        
        print(float(linestwo[0]),np.mean(S),np.mean(M),'mean',len(linestwo),len(lines))
    
N = np.geomspace(9*10**6,10**(9)+10**8,100)

fig = plt.figure(frameon=False)
ax = plt.subplot(111)

plt.axhline(y = root, color = 'b', linestyle = '--', linewidth = 2, alpha = 0.5) 
plt.axhline(y = proliferationtimelimit, color = 'g', linestyle = '-', linewidth = 1) 


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_xscale('log')
ax.set_yscale('log')

Time = [proliferationtime(n, λm, μm) + EvolutionaryRescueTime(n, λs, λa, λm, μs, μa, μm, u, v) for n in N]

plt.plot(N, Time, color='k', linewidth = 2.0, label='approximation')

asymmetric_error = [lower_error, upper_error]
ax.errorbar(NumberCells, MeanTime, asymmetric_error, fmt='ro',label='simulations') 

plt.xlabel(r'Initial number of cancer cells, $N$',fontsize=13)
ax.set_ylabel('Mean recurrence time, 'r'$\tau_a^r$ (days)',fontsize=13)

plt.xlim([9*10**6,10**(9)+10**8])
plt.ylim([10**3,10**(4)])


plt.xticks([10**7,10**8,10**9], [r'$10^7$',r'$10^8$',r'$10^9$'],fontsize=14)
plt.yticks([1000,10000], [r'$1000$',r'$10000$'],fontsize=14)

plt.legend(loc=1,fontsize=14,frameon=False)
plt.tight_layout()

plt.savefig('ProliferationTime.pdf')
plt.show()