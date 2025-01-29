import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')


# s = sensitive
# a = aneuploidy
# m = mutant
        
λs, λa, λm, μa, μm, u, v = 0.1, 0.0899, 0.1, 0.09, 0.09, 10**(-2), 10**(-7)
s, us = 7.3*10**(-3), 10**(-3)

dm = λm - μm

fig = plt.figure(frameon=False)
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

Col = ['r','b','g']

Col1 = ['r--','b--','g--']

Y = [-0.01,-0.005,-0.001]
X = np.linspace(0.11,0.17,10000)
N = [u/us*(abs(s)/abs(λs - x)) for x in X]
xaxis = [λs - x for x in X]
plt.plot(xaxis, N, color='k', linewidth=1.5)   

#Stochastic simulations plot
    
color = ['ro','bo','go']


txt=str("DataGenerationFractionAneuploidy.txt")
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

plt.xlabel(r'Sensitive growth rate, $\Delta_s$',fontsize=14)
plt.ylabel(r'Ratio of threshold tumor sizes,  $\frac{\tilde{N}_a^*}{N_a^*}$',fontsize=14)

ax.set_yscale('log')
plt.xticks([-0.07,-0.05,-0.03,-0.01], [r'$-0.07$',r'$-0.05$',r'$-0.03$',r'$-0.01$'],fontsize=14)
plt.yticks([10**(-1),10**(0),10**(1),], [r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$'],fontsize=14)
plt.xlim([-0.071,-0.009])
plt.tight_layout()
plt.savefig('RatiodsPlot.pdf')
plt.show()
###############################################################################