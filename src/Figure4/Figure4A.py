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
        
#Model parameters
λs, λm, μs, μa, μm, u, v = 0.1, 0.1, 0.14, 0.09, 0.09, 10**(-2), 10**(-7)

ds = λs - μs
dm = λm - μm

fig = plt.figure(frameon=False)
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

Col = ['k']

Col1 = ['r--','b--','g--']

Y = [-0.03]

for i in [0]:
    X = np.linspace(-0.03,0.01,10000)
    N = [NaS(λs, x + μa, λm, μs, μa,  μm, u, v) for x in X]
    plt.plot(X, N, color=Col[i], linewidth=1.75)   

Nms = abs(ds)/(v*λs)*λm/dm
plt.axhline(y = Nms, color = 'k', linestyle= '--', linewidth=1)

#Plot stochastic simulations 
    
color = ['ro','bo','go']

txt=str("ThresholdPopulationSize.txt")
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
    
ax.errorbar(A, B, asymmetric_error, fmt = color[i], alpha=0.5)

###############################################################################

plt.xlabel(r'Aneuploid growth rate, $\Delta_a$',fontsize=14)
plt.ylabel(r'Threshold tumor size, $N_a^*$',fontsize=14)

plt.xticks([-0.03,-0.015,0,0.01], [r'$-0.03$',r'$-0.015$',r'$0$',r'$0.01$'],fontsize=14)
plt.yticks([10**1,10**3,10**5,Nms,10**9], [r'$10^1$',r'$10^3$',r'$10^5$',r'$N_m^*$',r'$10^9$'],fontsize=14)

ax.set_yscale('log')


plt.xlim([-0.03,0.01])
plt.tight_layout()
plt.savefig('ThresholdPopulationSizePlot.pdf')
plt.show()
###############################################################################