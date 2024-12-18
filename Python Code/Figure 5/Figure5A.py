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

# n = naive
# a = aneuploidy
# m = mutant
 
def NaS(λw, λa, λm,  μw,  μa,  μm, u, v):
    
    dw = λw - μw
    da = λa - μa
    dm = λm - μm
    
    Ts = 1/np.sqrt(4*v*λa*λa*dm/λm)
    
    Ret = 0
    
    if da*Ts<-1:
        
        Ret = abs(da)/(v*λa)*λm/dm # abs(dw)*(1-u/abs(da))/v*λm/dm
        
    if abs(da*Ts)<1:
        
        Ret = 2*λa*Ts
        
    if da*Ts>1:
        
        Ret = λa/da
        
    return abs(dw)/(u*λw)*Ret

def NaS2(λw, λa, λm,  μw,  μa,  μm, u, v):
    
    dw = λw - μw
    da = λa - μa
    dm = λm - μm
    
    Ts = 1/np.sqrt(4*v*λa*λa*dm/λm)
    
    Ret = 0
    
    if da*Ts<-1:
        
        Ret = abs(da)/(v*λa)*λm/dm # abs(dw)*(1-u/abs(da))/v*λm/dm
        
    if abs(da*Ts)<1:
        
        Ret = 2*λa*Ts
        
    if da*Ts>1:
        
        Ret = λa/da
        
    return Ret
        
λw, λa, λm, μa, μm, u, v = 0.1, 0.0899, 0.1, 0.09, 0.09, 10**(-2), 10**(-7)
s, us = 7.3*10**(-3), 10**(-3)
print(u/us*(abs(s)/abs(λw - 0.14)),'ratio')

#dw = λw - μw
dm = λm - μm

#Ts = np.sqrt(4*v*λa*λa*dm/λm)
#print(Ts,r'Ts')

fig = plt.figure(frameon=False)
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#Nm = 10**(4)

Col = ['r','b','g']

Col1 = ['r--','b--','g--']

Y = [-0.01,-0.005,-0.001]
f = us*λw/abs(s)
print(f)
#F = (1-f)*(u*λw)/abs(dw)+f
X = np.linspace(0.11,0.17,10000)
N = [u/us*(abs(s)/abs(λw - x)) for x in X]
#N = [((1-f)+f*abs(λw-x)/(u*λw))**(-1) for x in X]
xaxis = [λw - x for x in X]
plt.plot(xaxis, N, color='k', linewidth=1.5)   


#Stochastic simulations plot
    
color = ['ro','bo','go']

#for i in [0,1,2]:   
#     
#    txt=str("P_estTauLeapSim_da_"+str(i)+".txt")
#    text_file = open(txt, "r")
#    lines = text_file.read().split(" ")
#    
#    A=[]
#    B=[]
#    C=[]
#    
#    for j in range(len(lines)-1):
#        if j%3==0:
#            A.append(0.14-float(lines[j]))
#        if j%3==1:
#            B.append(float(lines[j]))
#        if j%3==2:
#            C.append(float(lines[j]))
#            
#    D = []
#    for x in B:
#        
#        D.append(1.96*np.sqrt(x*(1-x)/100))
#        
#    ax.errorbar(A, B, D, fmt = color[i], alpha=0.5)

txt=str("DataGenerationFractionAneuploidyDaniel.txt")
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

#pm = (λm - μm)/λm

#S1 = w - 2*np.sqrt(λa*w*pm)
#S2 = w + 2*np.sqrt(λa*w*pm)

#S = 2*w*pm + μa + w - 2*np.sqrt(w*pm*(w*pm + μa + w)) - μa

#print(S,μa)

#plt.xticks([-0.05,-0.025,0,0.025,0.05], [r'$-0.05$',r'$-0.025$',r'$0$',r'$-0.025$',r'$0.05$'],fontsize=14)


#plt.xticks([-10**(-1),-1/Ts ,0, 1/Ts, 10**(-1)], [r'$-0.1$',r'$\Delta_{a-}^*$' ,r'$0$',r'$\Delta_{a+}^*$',r'$0.1$'],fontsize=14)
#plt.gca().get_xticklabels()[1].set_color("grey")
#plt.gca().get_xticklabels()[3].set_color("grey")


#plt.axvline(S1,ymin=10**(-3),ymax=1, linewidth=0.5, color='black')
#plt.axvline(S2,ymin=10**(-3),ymax=1, linewidth=0.5, color='black')

#ax.set_xscale('log')
ax.set_yscale('log')
plt.xticks([-0.07,-0.05,-0.03,-0.01], [r'$-0.07$',r'$-0.05$',r'$-0.03$',r'$-0.01$'],fontsize=14)
plt.yticks([10**(-1),10**(0),10**(1),], [r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$'],fontsize=14)

#plt.ylim([10**(-6),10**(-5)])

plt.xlim([-0.071,-0.009])
#plt.suptitle(r'$b=1\mu m$, $a=0.1\mu m$, $D=0.5\mu m^2/s$', fontsize=14)
#plt.legend(prop={'size': 10}, loc=3)
#legend = plt.legend(title="Wildtype\ngrowth rate", loc=1,fontsize=12,frameon=False)#,ncol = len(ax.lines))
#legend.get_title().set_fontsize('12')

#ax.text(-0.025, 1.2, 'A', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
plt.tight_layout()#(pad=2.0)
plt.savefig('RatiodsPlot.pdf')
plt.show()
###############################################################################