import numpy as np
import matplotlib.pyplot as plt


# Get current size
fig_size = plt.rcParams["figure.figsize"]
#print("Current size:", fig_size)
#Set figure width to 8 and height to 4.8
fig_size[0] = 10   #povečamo našo slikico (menda je osnova(6.4,4.8)
fig_size[1] = 4.8
plt.rcParams["figure.figsize"] = fig_size
#print("Current size:", fig_size)

t=np.arange(0,256,1)
def r(t): #prenosna funkcija
    tau=30
    r=1/(tau)*np.exp(-t/tau)
    return r




plt.figure(0)


plt.figure(1)
def S(t):
    S=10000*np.exp(-0.3125*t) + 10000*np.exp(-80 + 0.3125*t)
    return S
phi=(S(t)/(S(t)+(10**-2)))
plt.plot(t,phi,label='1)')
def S(t):
    S=1000000*(np.exp(-0.3125*t) + np.exp(-80 + 0.3125*t))
    return S

phi=(S(t)/(S(t)+(10**2)))
plt.plot(t,phi,label='2)')
def S(t):
    S=4000000*(np.exp(-0.3125*t) + np.exp(-80 + 0.3125*t))
    return S

phi=(S(t)/(S(t)+(10**4)))
plt.plot(t,phi,label='3)')

def S(t):
    S=10000000*(np.exp(-0.3125*t) + np.exp(-80 + 0.3125*t))
    return S

phi=(S(t)/(S(t)+(10**5.5)))
plt.plot(t,phi,label='4)')
plt.xlabel('f')
plt.legend(loc=0,frameon=False)
plt.title('Wienerjev filter')
plt.savefig('lincoln_filter.png')