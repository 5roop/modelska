import numpy as np
from scipy.fftpack import fft, ifft


file=open("lincoln_L30_N30.pgm","r")
row=[]
for line in file:
    fields = line.split(" ")
    row.append(fields)
file.close()


row.pop(0),row.pop(0),row.pop(0)
vrsta=[]

for vrstica in row:
    for j in vrstica:
        if j!='\n':
            vrsta.append(j)


x=np.zeros((256,313))
k=-1
for i in range(len(vrsta)):
    if i%313==0:
        k=k+1
    x[k][i%313]=vrsta[i]

y=np.transpose(x) # to moremo dekonvoluirat po vrsticah


t=np.arange(0,256,1)
def r(t): #prenosna funkcija
    tau=30
    r=1/(tau)*np.exp(-t/tau)
    return r

def S(t):
    S=4000000*(np.exp(-0.3125*t) + np.exp(-80 + 0.3125*t))
    return S

phi=(S(t)/(S(t)+(10**4)))


yf=np.zeros((313,256))
print(yf)
for i in range(313):
    U=phi*fft(y[i])/fft(r(t))
    u=ifft(U)
    u=np.abs(u)
    for j in range(256):
        u[j]=int(u[j])
        if u[j] >255:
            u[j]=255
        yf[i][j]=u[j]
    #print((u))
    #print(yf[i])


file=open("Lincoln4.pmg","w")
file.write('P2 \n313 256\n255\n')
for i in range(256):
    for j in range(313):
        file.write(str(int(yf[j][i])))
        file. write(' ')
    file.write('\n')
file.close()