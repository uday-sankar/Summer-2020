#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys
import time


# In[3]:


def Poten_H(x):
    V=(m*(w*x)**2)/2
    return V


# In[4]:


def F_H(x,E):
    v=Poten_H(x)
    return 2*m*(v-E)/h**2


# In[5]:


def Numerov(x,q0,q1,psi1,f1,dx,E):
    q2=dx*dx*f1*psi1+2*q1-q0
    f2=F_H(x+dx,E)
    psi2=q2/(1-f2*dx**2/12)
    return q2,f2,psi2


# In[6]:


def Initialize(X,E,psi0=1e-5,psi1=10e-5):
    '''
    Initializes all the variables needed for finding the 
    solution using Numerov Method
    q->phi
    f->(V-E)
    psi->Our solution to the schrodinger equation
    '''
    dx=X[1]-X[0]
    f0=F_H(X[0],E)
    f1=F_H(X[1],E)
    q0=psi0*(1-dx**2*f0/12)
    q1=psi1*(1-dx**2*f1/12)
    psi=[psi0,psi1]
    f=[f1]
    q=[q0,q1]
    return psi,f,q


# In[135]:


def run_eq(X,e,psi,f,q):
    '''
    The function which does the integration of the
    differential equation
    '''
    for i in range(len(X)-2):
        x=X[i+1]
        f1=f[0]
        psi1=psi[-1]
        q1=q[1]
        q0=q[0]
        dx=X[i+2]-X[i+1]
        q2,f2,psi2=Numerov(x,q0,q1,psi1,f1,dx,e)
        q[0]=q1
        q[1]=q2
        f[0]=f2
        psi.append(psi2)
    psi_n=np.copy(psi)
    return psi_n


# In[8]:


def run_mult(X,range_E):
    '''
    For running the solutions for a range of Energies
    '''
    data=[]
    for e in range_E:
        psi,f,q=Initialize(X,e)
        run_eq(X,e,psi,f,q)
        data.append([e,psi])
    return data


# In[9]:


def Eg_Desnd(X,E_init,num_prec,dE=.1):
    '''
    Energy descend function.
    E_init->initial guess, num_prec->precision needed, 
    Thisd function searches for eigen values by moving itself
    to the Energy region with minimum Explosion-factor.
    Seraches 3 nearby values of E_init moves itselves to the 
    energy region with minimum explosion-factor
    '''
    V=Poten_H(X)
    if min(V)>E_init:#if the energy is higer than the minimum of our poternial
        return Eg_Desnd(X,min(V)+dE,10,dE)
    else:
        act_range=np.where(V<=E_init)#active range is the likely region where we can expect a spike in probability
        ind_lw=act_range[0][0]# lower index of active range
        ind_up=act_range[0][-1]#upper index of actieve range
        j=0
        while j<=num_prec:#for interating till needed precision
            E_range=[E_init-dE,E_init,E_init+dE]
            d=run_mult(X,E_range)
            Explo_F=[]
            for  i in range(len(E_range)):
                Y=d[i][1]#solution of ith energy in E_range
                Expec_max=max(Y[ind_lw:ind_up])#maximum psi in the expected region
                Expec_min=min(Y[ind_lw:ind_up])#minimum psi in the expected region
                if Expec_min>Expec_max:#if minimum is higer in value
                    Expec_max=Expec_min
                explo=abs(Y[-1]/Expec_max)
                Explo_F.append(explo)
            Explo_F=np.copy(Explo_F)
            if Explo_F.argmin()==1:#if the middle energy has minimum explosion. Then zoom into that range
                dE/=2
                j+=1
            else:
                E_init=E_range[Explo_F.argmin()]
        return E_init


# In[114]:


def Explo_min_Finder(X,E_range):
    '''
    Finds the local minimum in explosion factor and returs
    the corresponding energy
    '''
    V=Poten_H(X)
    if min(E_range)<min(V):
        #if the energy is below the minimum potential then real solutions are not possible
        E_range[E_range.argmin()]=min(V)+h*w/10
        return Explo_min_Finder(X,E_range)
    else:
        #active range is defined based on the maximum energy
        active_range=np.where(V<=max(E_range))
        ind_lw=active_range[0][0]
        ind_up=active_range[0][-1]
        Sol=run_mult(X,E_range)#getting the solutions ofd the diff equation
        Explo_Farray=[]#explosion-factor array
        Optim_E=[]#the energies near to eigen values
        for s in Sol:
            y=s[1]
            Expec_max=max(y[ind_lw:ind_up])#maximum in the expected region
            Expec_min=abs(min(y[ind_lw:ind_up]))
            if Expec_max<Expec_min:
                Expec_max=Expec_min
            Explo_Farray.append(abs(y[-1]/Expec_max))
        for j in range(len(Sol)-2):
            if Explo_Farray[j]>Explo_Farray[j+1]:
                if Explo_Farray[j+1]<Explo_Farray[j+2]:
                    Optim_E.append(Sol[j+1][0])
        return Optim_E


# In[118]:


def Harmonic_eigen_finder(X,min_e=0,max_e=6,dE=.1):
    '''
    This function finds all eigen energy values for the harmonic
    oscilator when a range is specified along with the intervels to be checked
    The the X range also has to be specified.
    dE is the difference between 2 adjacent E we take
    '''
    Eigen_E=[]#the possible eigen enrgies
    E_array=np.arange(min_e,max_e,dE)#the rough energies to checked for eignality
    EG=Explo_min_Finder(X,E_array)
    for e in EG:
        print(f"Input for Energy optimization:{e}")
        e_optimized=Eg_Desnd(X,e,10,dE/10)
        Eigen_E.append(e_optimized)
        print(f"Optimized Energy:{e_optimized}")
    return Eigen_E


# In[124]:


def Normalize(x,y,norml_Val=1):#function to normalize the function
    '''
    This function normalizes the y value, such that nintegral
    of over entire X will be one
    the process is simple just finds out the area under the graph
    Absolute of areas are taken otherwise the symetric functions will
    be multiplied by a large number since their are is nearly zero
    here we need to normalise psi square so taking the absolute areas dosen't hurt
    '''
    A=0
    for i in range(len(x)-1):
        dx=(x[i+1]-x[i])
        a=abs(dx*(y[i]+y[i+1])/2)
        A=A+a
    norm_y=y/A
    return norm_y


# In[175]:


def Run_both(X,e=.5,psi_0=1e-10,psi_1=1e-10):
    '''
    Runs the solutions in both directions and the use merger
    to mergs both the functions.
    '''
    X_b=X[::-1]
    psi,f,q=Initialize(X,e,psi_0,psi_1)
    psi_b,f_b,q_b=Initialize(X_b,e,psi_0,psi_1)
    run_eq(X,e,psi,f,q)
    run_eq(X_b,e,psi_b,f_b,q_b)
    N_psi=Merger(X,psi,X_b,psi_b)
    return N_psi


# In[176]:


def Merger(X,psi,X_b,psi_b):
    '''
    Merges 2 opppositiely run solutions of the equations 
    and merges them both. The merging is not perfect but the effect
    is imperfections in not significant when looked at ovewrall
    function
    '''
    psi_b=np.copy(psi_b[::-1])
    psi_m=np.copy(psi)# the final merged psi
    last_min=0#stores the postion of last minimum in the psi array this indicates the minimum before the explotion
    max_psi=max(psi)
    min_psi=min(psi)
    if abs(min_psi)>max_psi:
        max_psi=min(psi)
    if max_psi==psi[-1]:#detecting an explosion near the far end
        for p in range(len(psi)-2):
            if abs(psi_m[p])>abs(psi_m[p+1]):
                if abs(psi_m[p+1])<abs(psi_m[p+2]):
                    last_min=p+1
        psi_m[last_min:]=psi_b[last_min:]#merging both of them
        psi_n=Normalize(X,psi_m)
    else:#if no explosion the function is fine
        psi_n=Normalize(X,psi_m)
    return psi_n


# In[182]:


def Plot_Eq(X,E_range):
    '''
    Plots all the solutions for the given energies
    '''
    print(f"Plottings for {E_range}")
    i=1
    for e in E_range:
        Psi=Run_both(X,e)
        ax.plot(X,Psi+e,label=f'Eigen State {i}')
        i+=1


# In[189]:


print("This programs objectieve is find the acceptable solutions for the Quantum Harmonic Oscilator")
print("By deafult the value of m,w,h-bar is set to 1,for easy analysis.")
K=input("Do you want to use default parameters?\n(Y/N):")
d=0
while d==0:
    if K=='N' or K=='n':
        m=float(input("Enter the value of mass(of the particle):"))
        w=float(input("Enter the value of angular frequency:"))
        h=float(input("Enter the value of h-bar:"))
        d=1
    elif K=='Y' or K=='y':
        h=1
        m=1
        w=1
        d=1
    else:
        print("Enter Answer:")
        d=0
print("Usually lower bound of E is zero. Giving any lower value is useless. You can raise this value.")
print("Usually better to give the upper boung in the range of w*h.")        
e_min=float(input("Enter the lower bound for searching energy:"))
e_max=float(input("Enter the upper bound for searching energy:"))


# In[200]:


X_min=input("Enter the lower bound of X to be covered(Enter ! To set this automatically):")
if X_min=='!':
    E_V_intersec=(2*e_max/(m*w*w))**(0.5)#intersection point of E an potential
    X_max=E_V_intersec*(1.5)
    X_min=-E_V_intersec*(1.5)
else:
    Xmax=float(input("Enter the upper bound of X to be covered"))


# In[201]:


X=np.linspace(X_min,X_max,10**3)


# In[202]:


E=Harmonic_eigen_finder(X,e_min,e_max)


# In[210]:


fig=plt.figure()
ax=plt.axes(xlabel='X',ylabel='Psi',ylim=(0,E[-1]*1.1))
ax.plot(X,Poten_H(X))
Plot_Eq(X,E)
plt.legend()
plt.show()

# In[204]:





# In[ ]:





# In[ ]:





# In[ ]:




