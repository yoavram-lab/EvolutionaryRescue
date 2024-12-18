import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')


# s = sensitive
# a = aneuploidy
# m = mutant
 
#Tumor threshold Na*
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

#Probability of rebound
def ReboundProbability(t, λs, λa, λm,  μs,  μa,  μm, u, v, N):
    
    ds = λs - μs
    da = λa - μa
    dm = λm - μm

    pm = dm/λm
    
    return 1 - (1 - np.exp(-(u*v*λs*λa*N*pm*((np.exp(ds*t)-1)/ds-(np.exp(da*t)-1)/da)/(ds-da)+v*λs*N*pm*(np.exp(ds*t)-1)/ds)))

#Model parameters
N, λs, λm, μs, μa, μm, u, v = 10**7, 0.1, 0.1, 0.14, 0.09, 0.09, 10**(-2),10**(-7)

fig = plt.figure(frameon=False)
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

Col = ['b','r','k','g']

Col1 = ['r--','b--','g--']

time = np.geomspace(0.01,100000,1000)

λa = 0.089999 #no aneuploidy, i.e. u=0 

Prob = [ReboundProbability(t, λs, λa, λm,  μs,  μa,  μm, 0, v, N) for t in time]

plt.plot(time, Prob, color=Col[0], linewidth=1.5, label=r'no aneuploidy') 

λa = 0.0899 #tolerant aneuploidy

Prob = [ReboundProbability(t, λs, λa, λm,  μs,  μa,  μm, u, v, N) for t in time]

plt.plot(time, Prob, color=Col[2], linewidth=1.5, label=r'tolerant')

λa = 0.089999 #stationary aeuploidy

Prob = [ReboundProbability(t, λs, λa, λm,  μs,  μa,  μm, u, v, N) for t in time]

plt.plot(time, Prob, color=Col[1], linewidth=1.5, label=r'stationary')

λa = 0.095 #partially resistant aneuploidy

Prob = [ReboundProbability(t, λs, λa, λm,  μs,  μa,  μm, u, v, N) for t in time]

plt.plot(time, Prob, color=Col[3], linewidth=1.5, label=r'partially resistant')

#Plot stochastic simulations
    
color = ['bo','ro','ko','go']

txt=str("ReboundProbability.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]

for j in range(len(lines)-1):
    A.append(float(lines[j]))
    
C = []

for t in [1,10,100,1000,10000]:
    
    B = []
    for x in A:
        if x<t and x>0:
            B.append(x)
    
    C.append(1 - len(B)/100)
      
D = []
for x in C:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
   
A = [1,10,100,1000,10000]
ax.errorbar(A, C, D, fmt = color[2],clip_on=False, alpha=0.5)

###############################################################################
txt=str("ReboundProbabilityNoAneuploidy.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]

for j in range(len(lines)-1):
    A.append(float(lines[j]))
    
C = []

for t in [1,10,100,1000,10000]:
    
    B = []
    for x in A:
        if x<t and x>0:
            B.append(x)
    
    C.append(1 - len(B)/100)
      
D = []
for x in C:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
    
A = [1,10,100,1000,10000]
ax.errorbar(A, C, D, fmt = color[0],clip_on=False, alpha=0.5)

###############################################################################
txt=str("ReboundProbabilityLargeDa.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]

for j in range(len(lines)-1):
    A.append(float(lines[j]))
    
C = []

for t in [1,10,100,1000]:
    
    B = []
    for x in A:
        if x<t and x>0:
            B.append(x)
    
    C.append(1 - len(B)/100)
      
D = []
for x in C:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
       
A = [1,10,100,1000]
ax.errorbar(A, C, D, fmt = color[3],clip_on=False, alpha=0.5)
###############################################################################
txt=str("ReboundProbabilityStationary.txt")
text_file = open(txt, "r")
lines = text_file.read().split(" ")

A=[]

for j in range(len(lines)-1):
    A.append(float(lines[j]))
    
C = []

for t in [1,10,100,1000,10000]:
    
    B = []
    for x in A:
        if x<t and x>0:
            B.append(x)
    
    C.append(1 - len(B)/100)
      
D = []
for x in C:
    
    D.append(1.96*np.sqrt(x*(1-x)/100))
       
A = [1,10,100,1000,10000]
ax.errorbar(A, C, D, fmt = color[1],clip_on=False, alpha=0.5)

###############################################################################

plt.xlabel(r'Time, $t$ (days)',fontsize=12)
plt.ylabel('Probaibility that a rescue lineage\n has not appeared by time $t$\n$1-P(rescue,t)$',fontsize=12)

ax.set_xscale('log')

plt.xticks([0.1,1,10 ,100, 1000, 10000, 100000], [r'$0.1$', r'$1$', r'$10$', r'$100$', r'$1000$', r'$10000$', r'$100000$'],fontsize=14)
plt.xticks(rotation=45)

plt.yticks([0.0,0.5,1], [r'$0$',r'$0.5$',r'$1$'],fontsize=14)
plt.xlim([0.9,11000])
plt.ylim([0,1.1])

legend = plt.legend(loc=3,fontsize=12,frameon=False)
legend.get_title().set_fontsize('10')

plt.tight_layout()
plt.savefig('ReboundProbability.pdf')
plt.show()