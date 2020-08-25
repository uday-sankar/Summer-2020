import matplotlib.pyplot as plt
import numpy as np
import time

def F(a,b):
    return a**2-b

def Numrov(x,q0,q1,psi1,f1,dx,eps):
    q2=dx**2*f1*psi1+2*q1-q0
    f2=F(x,eps)
    psi2=q2/(1-f2*dx**2/12)
    return q2,f2,psi2

def run_eq(X,q,f,psi,dx,eps):
    #print(eps)
    for i in range(len(X)-2):
        x=X[i+1]
        f1=f[-1]
        psi1=psi[-1]
        q1=q[-1]
        q0=q[-2]
        q2,f2,psi2=Numrov(x,q0,q1,psi1,f1,dx,eps)
        q.append(q2)
        f.append(f2)
        psi.append(psi2)

def run_mult(range_eps):
    data=[]
    for eps in range_eps:
        X,psi,f,q,dx=initials(eps)
        run_eq(X,q,f,psi,dx,eps)
        max_v=max(psi)#maximum
        min_v=abs(min(psi))#absolute value of minimum
        if min_v>max_v:
            psi=psi/min_v
        else:
            psi=psi/max_v
        data.append([X,psi])
    return data

def initials(eps=1,Xmin=-6,Xmax=6,psi_0=1e-30,psi_1=1e-30,div=1e5):
    '''
    Xmin,Xmax=minimum and maximum of the range
    div denotes the number of divisions for X
    '''
    X=np.linspace(Xmin,Xmax,int(div))
    dx=X[2]-X[1]
    f_0=(X[0]**2-eps)
    f_1=(X[1]**2-eps)
    q_0=psi_0*(1-dx**2*f_0/12)
    q_1=psi_1*(1-dx**2*f_1/12)
    psi=[psi_0,psi_1]
    f=[f_0,f_1]
    q=[q_0,q_1]
    return X,psi,f,q,dx

def Eigen_finder2(eps_init,num_steps,div=.1):
    '''
    starts with an eigen values
    in all epsilon corresponding to non iegen energies the psi goes to infinty after 
    the origin.Then find the  fractional difference between the higest point near origin and last value of psi
    Then this function tries to minimize this fractional difference by changing the epsilon. 
    i,e it tries to find a psi which has least value near the end of our range(this our boundary condition)
    '''
    for i in range(num_steps):# an arbitary number of steps
        E_range=[eps_init-div,eps_init,eps_init+div]#Our epsilon range 3 psi values for which we try the plotting
        d=run_mult(E_range)#getting the values
        X=d[0][0]#X is same for all 
        imin=np.where((X>-2) & (X<-1.9))[0][0]# getting the range of indices near origin
        imax=np.where((X>2)&(X<2.1))[0][0]#here our range is (-2,2)
        Grad=[]#array to store the fractional difference
        '''
        Method:
        1) find the fractional differencees for 3 epsilon values
        2)fins the minimum amoung the fractional differences
        3) I .Shifts the epsilon ranges toward the epsilon which gave minimum difference
          II .If the difference is less for the current epsilon the epsilon range 
          is futher divided into more finer intervales
        '''
        for i in range(len(E_range)):
            Y=d[i][1]#psi values are diffrent for different epsilons
            max_psi=0# The maximum ner origin
            for y in Y[imin:imax]: #finding the maximum near origin(our range(-2,2))
                if abs(y)>max_psi:
                    max_psi=abs(y)
            g=abs(Y[-1]/max_psi)# the maximum
            Grad.append(g)
        if Grad[2]<Grad[0]:# comparing the fractional difference between different epsilons
            if Grad[2]<Grad[1]:# if the fractional difference is lesser for the higher epsilon 
                eps_init=eps_init+div#then shifts epsilon range to higer epsilon
            else:
                div=div/2# if its lest for current epsilon the range is further divided into finer intervals
        else:
            if Grad[0]<Grad[1]:
                eps_init=eps_init-div
            else:
                div=div/2
    return eps_init

def Eigen_Helper(min_e,max_e,int_gap):
    '''
    In this method we divide an interval and get many points in between the interval.
    Then optimize the values to get eigen energies
    '''
    Eps=[]#denotes the array to store possible eigen epsilons
    
    M_ra=np.linspace(min_e,max_e,int_gap)#the main range
    for i in range(len(M_ra)):
        print(M_ra[i])
        C=True
        eps=Eigen_finder2(M_ra[i],15)#optimizing to get eigen epsilons
        eps=round(eps,2)
        for j in range(len(Eps)):# to avoid repetition of eigen values
            if Eps[j]==eps:
                C=False
        if C:
            Eps.append(eps)
        print(eps)
    return Eps

def plot_mult(Epsrange):
    pl=run_mult(Epsrange)
    i=0
    for eps in Epsrange:
        plt.plot(pl[i][0],pl[i][1],label=eps)
        i+=1
# In[18]:
h=1
w=1
int_gap=10
t_in=time.time()
EPS=Eigen_Helper(1,11,int_gap)
# In[19]:
print(EPS)
# In[20]:
Energy=[]
for e in EPS:
    E=e*h*w/2
    Energy.append(E)
print(Energy)
tl=time.time()
print("time for computing=",tl-t_in)
plot_mult(EPS)



