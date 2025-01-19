import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

# s = sensitive
# a = aneuploid
# m = mutant
 
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

def prob(N, λs, λa, λm,  μs,  μa,  μm, u, v):
    
    return 1 - np.exp(-N/NaS(λs, λa, λm,  μs,  μa,  μm, u, v))
        
#Model parameters
λs, λa, λm, μs, μa, μm, u, v = 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09, 10**(-2),10**(-7)

ds = λs - μs
dm = λm - μm

Ts = 1/np.sqrt(4*v*λa*λa*dm/λm)

fig = plt.figure(figsize=(7,5.5),frameon=False)
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

Col = ['k','b','g','m']

Col1 = ['r--','b--','g--']

PopSize = np.geomspace(10**0,10**9,10000)
Nms = abs(ds)/(v*λs)*λm/dm

#Partially resistant aneuploids
Prob = [prob(x, λs, 0.091, λm,  μs,  μa,  μm, u, v) for x in PopSize]

plt.plot(PopSize, Prob, color=Col[2], linewidth=1.5, label=r'partially resistant') 

#Stationary aneuploids
Prob = [prob(x, λs, 0.09-10**(-5), λm,  μs,  μa,  μm, u, v) for x in PopSize]

plt.plot(PopSize, Prob, color='r', linewidth=1.5, label=r'stationary') 

#Tolerant aneuploids
Prob = [prob(x, λs, λa, λm,  μs,  μa,  μm, u, v) for x in PopSize]

plt.plot(PopSize, Prob, color=Col[0], linewidth=1.5, label=r'tolerant') 

#No aneuploids
Prob_wAneuploidy = [1-np.exp(-x/Nms) for x in PopSize]
plt.plot(PopSize, Prob_wAneuploidy,'-', linewidth=1.5, label=r'no aneuploidy') 

s1 = (λm - μm)/λm
s2 = (λa - μa - v*λa + np.sqrt((λa - μa - v*λa)**2 + 4*λa*λa*v*s1))/(2*λa)
pw = (λs - μs - u*λs - v*λs + np.sqrt((λs - μs - u*λs - v*λs)**2 + 4*λs*λs*(u*s2 + v*s1)))/(2*λs)

plt.axvline(x=NaS(λs, λa, λm,  μs,  μa,  μm, u, v), color=Col[0],linestyle="--", linewidth=1)

###############################################################################

λa = 0.091
s1 = (λm - μm)/λm
s2 = (λa - μa - v*λa + np.sqrt((λa - μa - v*λa)**2 + 4*λa*λa*v*s1))/(2*λa)
pw = (λs - μs - u*λs - v*λs + np.sqrt((λs - μs - u*λs - v*λs)**2 + 4*λs*λs*(u*s2 + v*s1)))/(2*λs)
plt.axvline(x=NaS(λs, λa, λm,  μs,  μa,  μm, u, v), color='g',linestyle="--", linewidth=1)

λa = 0.09-10**(-5)
s1 = (λm - μm)/λm
s2 = (λa - μa - v*λa + np.sqrt((λa - μa - v*λa)**2 + 4*λa*λa*v*s1))/(2*λa)
pw = (λs - μs - u*λs - v*λs + np.sqrt((λs - μs - u*λs - v*λs)**2 + 4*λs*λs*(u*s2 + v*s1)))/(2*λs)
plt.axvline(x=NaS(λs, λa, λm,  μs,  μa,  μm, u, v), color='r',linestyle="--", linewidth=1)

#Ploting stochastic simulations
    
color = ['ro','bo','go']
    
txt=str("ProbvN.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]
B=[]
C=[]

for j in range(len(lines)-1):
    if j%3==0:
        A.append(float(lines[j]))
    if j%3==1:
        B.append(float(lines[j]))
    if j%3==2:
        C.append(float(lines[j]))
        
D = []
for x in B:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
    
ax.errorbar(A, B, D, fmt = 'ko', alpha=0.5)

txt=str("ProbvNnoAneuploidy.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]
B=[]
C=[]

for j in range(len(lines)-1):
    if j%3==0:
        A.append(float(lines[j]))
    if j%3==1:
        B.append(float(lines[j]))
    if j%3==2:
        C.append(float(lines[j]))
        
D = []
for x in B:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
    
ax.errorbar(A, B, D, fmt = color[1], alpha=0.5)

txt=str("ProbvNgg0.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]
B=[]
C=[]

for j in range(len(lines)-1):
    if j%3==0:
        A.append(float(lines[j]))
    if j%3==1:
        B.append(float(lines[j]))
    if j%3==2:
        C.append(float(lines[j]))
        
D = []
for x in B:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
    
ax.errorbar(A, B, D, fmt = 'go', alpha=0.5)
###############################################################################
txt=str("ProbvNee0.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]
B=[]
C=[]

for j in range(len(lines)-1):
    if j%3==0:
        A.append(float(lines[j]))
    if j%3==1:
        B.append(float(lines[j]))
    if j%3==2:
        C.append(float(lines[j]))
        
D = []
for x in B:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
    
ax.errorbar(A, B, D, fmt = 'ro', alpha=0.5)

###############################################################################

plt.xlabel(r'Initial number of cancer cells, $N$',fontsize=14)
plt.ylabel(r'Probability of recurrence, $p_{rescue}$',fontsize=14)

ax.set_xscale('log')

plt.xticks([10**1,10**3,10**5,10**7,10**9], [r'$10$',r'$10^3$',r'$10^5$',r'$10^7$',r'$10^9$'],fontsize=14)

ax2 = ax.secondary_xaxis('top')
ax2.spines['top'].set_visible(False)

ax2.tick_params(axis='x', colors='white')
ax2.set_xticks([NaS(λs, 0.0899, λm,  μs,  μa,  μm, u, v),NaS(λs, 0.09-10**(-5), λm,  μs,  μa,  μm, u, v),NaS(λs, 0.091, λm,  μs,  μa,  μm, u, v)], [r'$N_a^*$',r'$N_a^*$',r'$N_a^*$'],fontsize=14)

ax2.get_xticklabels()[0].set_color("black")
ax2.get_xticklabels()[1].set_color("red")
ax2.get_xticklabels()[2].set_color("green")

plt.yticks([0.0,0.5,1], [r'$0$',r'$0.5$',r'$1$'],fontsize=14)
plt.xlim([10**(0),10**(9)])
legend = plt.legend(loc=2,fontsize=12,frameon=False)#,ncol = len(ax.lines))
legend.get_title().set_fontsize('14')

plt.tight_layout()
plt.savefig('ProbvNPlot.pdf')
plt.show()