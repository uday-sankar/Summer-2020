#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from math import exp


# In[2]:


def V(x):
    return (m*(w*x)**2)/2


# In[3]:


def F(x,e):#using the equation ddpsi=(u^2-epsilon)*psi
    return 2*m*(V(x)-e)/h**2

def Numrov(x,q0,q1,psi1,f1,dx,eps):# function implementing numerov's algorith
    q2=(dx**2)*f1*psi1+2*q1-q0
    f2=F(x,eps)
    psi2=q2/(1-f2*dx**2/12)
    return q2,f2,psi2

def Numrov2(x,q2,q1,psi1,f1,dx,eps):# function implementing numerov's algorith
    q2=(dx**2)*f1*psi1+2*q1-q0
    f2=F(x,eps)
    psi2=q2/(1-f2*dx**2/12)
    return q2,f2,psi2

def run_eq(X,q,f,psi,eps):#function to find all the psi values
    for i in range(len(X)-2):
        x=X[i+1]
        f1=f[-1]
        psi1=psi[-1]
        q1=q[-1]
        q0=q[-2]
        dx=X[i+1]-X[i]
        q2,f2,psi2=Numrov(x,q0,q1,psi1,f1,dx,eps)
        q.append(q2)
        f.append(f2)
        psi.append(psi2)


# In[4]:


def run_mult(range_eps):
    data=[]
    for eps in range_eps:
        X,psi,f,q=initials(eps,Xaxmin,Xaxmax)
        run_eq(X,q,f,psi,eps)
        data.append([X,psi])
    return data

def initials(eps=1,Xmin=-5,Xmax=5,psi_0=1e-30,psi_1=1e-30,div=10**4):
    '''
    Xmin,Xmax=minimum and maximum of the range
    div denotes the number of divisions for X
    '''
    X=np.linspace(Xmin,Xmax,div)
    dx=X[1]-X[0]
    f_0=(X[0]**2-eps)
    f_1=(X[1]**2-eps)
    q_0=psi_0*(1-dx**2*f_0/12)
    q_1=psi_1*(1-dx**2*f_1/12)
    psi=[psi_0,psi_1]
    f=[f_0,f_1]
    q=[q_0,q_1]
    return X,psi,f,q


# In[5]:


def initialsBk(eps=1,Xmin=-5,Xmax=5,psi_0=1e-30,psi_1=1e-30,div=10**4):
    '''
    Xmin,Xmax=minimum and maximum of the range
    div denotes the number of divisions for X
    '''
    X=np.linspace(Xmin,Xmax,div)
    X=X[::-1]
    dx=X[1]-X[0]
    f_0=(X[0]**2-eps)
    f_1=(X[1]**2-eps)
    q_0=psi_0*(1-dx**2*f_0/12)
    q_1=psi_1*(1-dx**2*f_1/12)
    psi=[psi_0,psi_1]
    f=[f_0,f_1]
    q=[q_0,q_1]
    return X,psi,f,q


# In[21]:


def Eigen_finder2(eps_init,num_steps,div=0.1):
    '''
    starts with an eigen values
    in all epsilon corresponding to non iegen energies the psi goes to infinty after
    the origin.Then find the  fractional difference between the higest point near origin and last value of psi
    Then this function tries to minimize this fractional difference by changing the epsilon.
    i,e it tries to find a psi which has least value near the end of our range(this our boundary condition)
    '''
    #when the potential overtakes the energy the function will experince an exponential decay    
    X=np.linspace(Xaxmin,Xaxmax,10**4)#X is same for all
    Poten=V(X)
    if min(Poten)>eps_init:
        return Eigen_finder2(eps_init+div,10,div)
    else:
        Decay_potential=eps_init
    #when the potential overtakes the energy the function will experince an exponential decay 
    #we take only consider the active x range for taking the maximum wave function
    active_range=np.where(V(X)<=Decay_potential)#when potential is greater than the energy of the state that function will be a decaying function
    imin=active_range[0][0]# getting the range of indices near origin
    imax=active_range[0][-1]#here our range is (-2,2)
    j=0.0
    while j<=num_steps:# an arbitary number of steps
        E_range=[eps_init-div,eps_init,eps_init+div]#Our epsilon range 3 psi values for which we try the plotting
        d=run_mult(E_range)#getting the values
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
            max_psi=max(Y[imin:imax])# The maximum ner origin
            min_psi=abs(min(Y[imin:imax]))
            if min_psi>max_psi:
                max_psi=min_psi
            g=abs(Y[-1]/max_psi)# the maximum
            Grad.append(g)
        Grad=np.copy(Grad)
        if Grad.argmin()==1:
        	div=div/2# if explo-factor is least for current epsilon, the Energy range is further divided into finer intervals
        	j+=1
        else:#else the energy range is shifter towards the epsilon with lower explosion factor
        	eps_init=E_range[Grad.argmin()]
    return eps_init


# In[18]:


#explo_array
def explo_array(E_array,Xmin,Xmax):
    X=np.linspace(Xmin,Xmax,10**4)#X is same for all
    Poten=V(X)
    eps_init=E_array[0]
    if min(Poten)>=eps_init:
        Decay_potential=(min(Poten)+w)
    else:
        Decay_potential=eps_init
    #when the potential overtakes the energy the function will experince an exponential decay 
    #we take only consider the active x range for taking the maximum wave function
    active_range=np.where(V(X)<=Decay_potential)#when potential is greater than the energy of the state that function will be a decaying function
    imin=active_range[0][0]# getting the index at the begining of the active range
    imax=active_range[0][-1]# getting the index at the endof the active range
    Sol=run_mult(E_array)#running for all the energies
    Explo_Farray=[]#the explosion factor array
    Optim_E=[]#the array storing the approximate optimal energy 
    for s in Sol:
        y=s[1]
        Peak_max=max(y[imin:imax])
        Peak_min=abs(min(y[imin:imax]))
        if Peak_min>Peak_max:
            Peak_max=Peak_min
        Explo_Farray.append(abs(y[-1]/Peak_max))#y[-1]/Peak_max=explosion-factor
    for j in range(len(Explo_Farray)-2):
        if Explo_Farray[j]>Explo_Farray[j+1]:
            if Explo_Farray[j+1]<Explo_Farray[j+2]:
                Optim_E.append(E_array[j+1])
    return Optim_E


# In[9]:


def Eigen_Helper(min_e,max_e,int_gap):
    '''
    In this method we divide an interval and get many points in between the interval.
    Then optimize the values to get eigen energies
    '''
    Eps=[]#denotes the array to store possible eigen epsilons

    M_ra=np.arange(min_e,max_e,int_gap)#the main range
    dE=M_ra[1]-M_ra[0]
    M_ra=explo_array(M_ra,Xaxmin,Xaxmax)
    for i in range(len(M_ra)):
        print("\nInitail Guess->",M_ra[i])
        C=True
        eps=Eigen_finder2(M_ra[i],20,dE/10)#optimizing to get eigen epsilons
        eps=round(eps,2)
        for j in range(len(Eps)):# to avoid repetition of eigen values
            if Eps[j]==eps:
                C=False
        if C:
            Eps.append(eps)
        print("Correct eigen value after optimizing->",eps)
    return Eps


# In[10]:


def Normalize(x,y,norml_Val=1):#function to normalize the function
    '''
    UDX->uniformly descretized x axis
    this means that x axis is uniformly divided hence the interval between any 2 consecutieve points
    in X axis is same.
    '''
    A=0
    for i in range(len(x)-1):
        dx=(x[i+1]-x[i])
        a=abs(dx*(y[i]+y[i+1])/2)
        A=A+a
    norm_y=y/A
    return norm_y


# In[11]:


def Run_both(eps=1,Xmin=-10,Xmax=10,psi_0=1e-30,psi_1=1e-30,div=10**4):
    '''
    This function splirts the x axis into two and runs two function one from negative till x=0
    another from positive.
    xmin has to be a negatieve number and Xmax a positieve number, othewise this step donot work
    This gives a functiojn which satisfies boundary condition irrespective of the eigen value
    So FOR SOLVING EIGEN VALUES THIS FUNCTION SHOULD NOT BE USED
    '''
    X,psi,f,q=initials(eps,Xmin,Xmax,psi_0,psi_1,div)
    X_b,psi_b,f_b,q_b=initialsBk(eps,Xmin,Xmax,psi_0,psi_1,div)
    run_eq(X,q,f,psi,eps)
    run_eq(X_b,q_b,f_b,psi_b,eps)
    N_psi=Merger(X,psi,X_b,psi_b)
    return X,N_psi


# In[12]:


def Merger(X,psi,X_b,psi_b):
    '''
    Given the both forward and backward solution this function merges both of them together.
    An explotion near the end of the forward. othewise this tep is not needed
    X_b starts from 10 till -10(towards negatieve direction)
    '''
    psi_b=np.copy(psi_b[::-1])
    psi_m=np.copy(psi)# the final merged psi
    last_min=0#stores the postion of last minimum in the psi array this indicates the minimum before the explotion
    max_psi=max(psi)
    min_psi=min(psi)
    if abs(min_psi)>max_psi:
        max_psi=min(psi)
    if max_psi==psi[-1]:
        for p in range(len(psi)-2):
            if abs(psi_m[p])>abs(psi_m[p+1]):
                if abs(psi_m[p+1])<abs(psi_m[p+2]):
                    last_min=p+1
        psi_m[last_min:]=psi_b[last_min:]#merging both of them
        psi_n=Normalize(X,psi_m)
    else:
        psi_n=Normalize(X,psi_m)
    return psi_n


# In[13]:


def Plot_Eq(E_range,Xmin=-10,Xmax=10):
    print(E_range)
    for e in E_range:
        x,Psi=Run_both(e,Xmin,Xmax)
        plt.legend()
        print(e)
        ax.plot(x,Psi+e,label=e)


# In[14]:


h=1
m=1
w=.3
int_gap=.1
print("This function tries to find energy values when an energy interval\nand a precision value is given, \nif you don't have any what to type in precision value just type '!' when asked for it \n")


# In[16]:


w=float(input("Enter the angular frequency(omega) of oscilator:"))
E_search=[0,5]
E_search[0]=float(input("Enter the lower bound of the energy region to be searched:"))
E_search[1]=float(input("Enter the upper bound of the energy region to be searched:"))
int_gap=input("Enter the precision for checking eigen energies(if you are unaware just type !):")
if int_gap=='!':
    int_gap=w*h/2
else:
    int_gap=float(int_gap)
Xaxmin=float(input("Enter the lower bound of x axis to be covered:"))
Xaxmax=float(input("Enter the upper bound of x axis to be covered:"))


# In[22]:


EPS=Eigen_Helper(E_search[0],E_search[1],int_gap)


# In[24]:


X=np.linspace(Xaxmin,Xaxmax,10**3)
fig=plt.figure()
ax=plt.axes(ylim=(0,EPS[-1]+2*w))
ax.set_title("Solution for Quantum Harmonic Oscilator")
ax.plot(X,V(X))
Plot_Eq(EPS,Xaxmin,Xaxmax)
ax.legend()
plt.show()
i=1


# In[56]:






