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

#Exact Tumor threshold Na*
def NaS2(λs, λa, λm, μs,  μa,  μm, u, v):
    
    ds = λs - μs
    da = λa - μa
    dm = λm - μm

    pm = dm/λm
    pa = (da - v*λa*pm + np.sqrt((da - v*λa*pm)**2 + 4*λa*λa*v*pm))/(2*λa)
    ps = (ds - u*λs*pa - v*λs*pm + np.sqrt((ds - u*λs*pa - v*λs*pm)**2 + 4*λs*λs*(u*pa + v*pm)))/(2*λs)
        
    return 1/ps
        
#Model parameters
λs, λa, λm, μs, μa, μm, u, v = 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09, 10**(-2),10**(-7)

ds = λs - μs
dm = λm - μm

fig = plt.figure(frameon=False)
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

Col = ['r','b','g']

Col1 = ['r--','b--','g--']

Nms = abs(ds)/(v*λs)*λm/dm

for i in [1]:
    X = np.geomspace(10**(-5), 10**(0),10000)
    N = [NaS(λs, λa, λm, μs, μa, μm, x, v) for x in X]
    N2 = [NaS2(λs, λa, λm, μs, μa, μm, x, v) for x in X]

    plt.plot(X/v, N, color='k', linewidth=1.5)   
    plt.plot(X/v, N2, color='b', linewidth=1.25)   


plt.axhline(y = Nms, color = 'k', linestyle= '--', linewidth=1)

#Plot stochastic simulations
    
color = ['ro','bo','go']

txt=str("ThresholdPopulationSizeVersusRatio.txt")
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
        B.append(1/float(lines[j]))
    if j%4==2:
        lower_error.append(float(lines[j]))
    if j%4==3:
        upper_error.append(float(lines[j]))
        
asymmetric_error = [lower_error, upper_error]
    
ax.errorbar(A, B, asymmetric_error, fmt = color[0], alpha=0.5)

###############################################################################

plt.xlabel(r'$u/v$',fontsize=14)
plt.ylabel(r'Threshold tumor size, $N_a^*$',fontsize=14)

ax.set_xscale('log')
ax.set_yscale('log')

plt.yticks([10**5,10**6,10**7,Nms,10**8,10**9], [r'$10^5$',r'$10^6$',r'$10^7$',r'$N_m^*$',r'$10^8$',r'$10^9$'],fontsize=14)

plt.xlim([10**2,10**7])
plt.tight_layout()
plt.savefig('ThresholdPopulationSizeVersusRatioPlot.pdf')
plt.show()
###############################################################################