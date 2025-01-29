#import matplotlib.inline
import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.stats
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import seaborn as sns
sns.set_context('talk')

from functools import partial

# s = sensitive
# a = aneuploidy
# m = mutant
       
λs, λa, λm, μs, μa, μm, v = 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09, 10**(-7)
s, us = 7.3*10**(-3), 10**(-3)

fig = plt.figure(frameon=False)
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')


Col = ['r','b','g']

Col1 = ['r--','b--','g--']

Y = [-0.01,-0.005,-0.001]

u = np.geomspace(10**(-3),10**(-1),1000)
N = [U/us*(abs(s)/abs(λs - μs)) for U in u]
x = [us/U for U in u]
plt.plot(x, N, color='k', linewidth=1.5)   

#Stochastic simulations plot
    
color = ['ro','bo','go']

txt=str("DataGenerationFractionAneuploidyratiomutations.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]
B=[]
lower_error=[]
upper_error=[]


for j in range(len(lines)-1):
    if j%4==0:
        A.append(float(lines[j]))
    if j%4==1:
        B.append(float(lines[j]))
    if j%4==2:
        lower_error.append(float(lines[j]))
    if j%4==3:
        upper_error.append(float(lines[j]))
        
asymmetric_error = [lower_error, upper_error]
    
ax.errorbar(A, B, asymmetric_error, fmt = color[0], alpha=0.5)

###############################################################################

plt.xlabel(r'Ratio of mutation rates,  $\tilde{u}/u$',fontsize=14)
plt.ylabel(r'Ratio of threshold tumor sizes,  $\frac{\tilde{N}_a^*}{N_a^*}$',fontsize=14)

ax.set_xscale('log')
ax.set_yscale('log')

plt.xlim([0.009,1.1])
plt.tight_layout()
plt.savefig('ratio_uPlot.pdf')
plt.show()
###############################################################################