import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7.4,5.5),frameon=False)
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

def prob1cellMelanoma(λs, λa, λm,  μs,  μa,  μm, u, v):
    
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
        
    return (abs(ds)/(u*λs)*Ret)**(-1)


def probNcellMelanoma(N, λs, λa, λm,  μs,  μa,  μm, u, v):
    
    return 1 - np.exp(-N*prob1cellMelanoma(λs, λa, λm,  μs,  μa,  μm, u, v))
    
u = 10**(-2)
v = 10**(-7)
λa1, μa1 = 0.0845, 0.076
λa2, μa2 = 0.0845, 0.1015
λa3, μa3 = 0.0845, 0.1115

λs, μs = 0.0845, 0.1215

λm, μm = 0.0845, 0.076
dm = λm - μm

Ts1 = 1/np.sqrt(4*v*λa1*λa1*dm/λm)
Ts2 = 1/np.sqrt(4*v*λa2*λa2*dm/λm)
Ts3 = 1/np.sqrt(4*v*λa3*λa3*dm/λm)

da1 = λa1 - μa1
da2 = λa2 - μa2
da3 = λa3 - μa3

N = np.geomspace(10**0, 10**11, 1000)

plt.plot(N,probNcellMelanoma(N, λs, λa1, λm,  μs,  μa1,  μm, u, v),color='r', clip_on=False, label='TNBC-SA609 \nclone A')
plt.plot(N,probNcellMelanoma(N, λs, λa2, λm,  μs,  μa2,  μm, u, v),color='g', clip_on=False, label='TNBC-SA1035 \nclone H')
plt.plot(N,probNcellMelanoma(N, λs, λa3, λm,  μs,  μa3,  μm, u, v),color='b', clip_on=False, label='TNBC-SA535 \nclone H')


plt.axvline(x=prob1cellMelanoma(λs, λa1, λm,  μs,  μa1,  μm, u, v)**(-1), color='r', linestyle='--', linewidth=1)
plt.axvline(x=prob1cellMelanoma(λs, λa2, λm,  μs,  μa2,  μm, u, v)**(-1), color='g', linestyle='--', linewidth=1)
plt.axvline(x=prob1cellMelanoma(λs, λa3, λm,  μs,  μa3,  μm, u, v)**(-1), color='b', linestyle='--', linewidth=1)

λs, λa, λm, μs, μa, μm, u, v = 0.1, 0.0899, 0.1, 0.14, 0.09, 0.09, 10**(-2),10**(-7)

ds = λs - μs
da = λa - μa
dm = λm - μm

Ts = 1/np.sqrt(4*v*λa*λa*dm/λm)

plt.plot(N,probNcellMelanoma(N, λs, λa, λm,  μs,  μa,  μm, u, v),color='k', clip_on=False, label='Melamona A375')
plt.axvline(x=prob1cellMelanoma(λs, λa, λm,  μs,  μa,  μm, u, v)**(-1), color='k', linestyle='--', linewidth=1)


ax.set_xscale('log')
plt.ylim([0,1])
plt.xlabel(r'Initial number of cancer cells, $N$',fontsize=14)
plt.ylabel(r'Probability of recurrence, $p_{rescue}$',fontsize=14)
plt.xticks([10**1,10**3,10**5,10**7,prob1cellMelanoma(λs, λa3, λm,  μs,  μa3,  μm, u, v)**(-1),10**11], [r'$10$',r'$10^3$',r'$10^5$',r'$10^7$',r'$N_a^*$',r'$10^{11}$'],fontsize=14)

ax.get_xticklabels()[4].set_color("blue")

ax2 = ax.secondary_xaxis('top')
ax2.spines['top'].set_visible(False)

ax2.tick_params(axis='x', colors='white')


ax2.set_xticks([prob1cellMelanoma(λs, λa1, λm,  μs,  μa1,  μm, u, v)**(-1),prob1cellMelanoma(λs, λa, λm,  μs,  μa,  μm, u, v)**(-1),prob1cellMelanoma(λs, λa2, λm,  μs,  μa2,  μm, u, v)**(-1)], [r'$N_a^*$',r'$N_a^*$',r'$N_a^*$'],fontsize=11.25)

ax2.get_xticklabels()[0].set_color("red")
ax2.get_xticklabels()[1].set_color("black")
ax2.get_xticklabels()[2].set_color("green")



plt.yticks([0, 0.5, 1], [r'$0$', r'$0.5$', r'$1$'],fontsize=14)
plt.xlim([10**(0),10**(11)])
legend = plt.legend(loc=2,fontsize=8,frameon=False)
legend.get_title().set_fontsize('8')
plt.tight_layout()
plt.savefig("PDXModelProb.pdf")
plt.show()