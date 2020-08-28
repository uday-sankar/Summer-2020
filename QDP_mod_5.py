
import matplotlib.pyplot as plt
import numpy as np
from math import exp
import sys
import time



# In[2]:


def Poten(x):
    if C_4!=0:
        C_0=C_2**2/(4*C_4)#term to make the potential positieve always
    else:
        C_0=C0
    V=K*(C_4*(x**4)-C_2*(x**2)+C_0)
    return V
def F(x,E):
    v=Poten(x)
    return 2*m*(v-E)/h**2


# In[3]:


def Numrov(x,q0,q1,psi1,f1,dx,E):
    q2=(dx**2)*f1*psi1+2*q1-q0
    f2=F(x,E)
    psi2=q2/(1-f2*dx**2)
    return q2,f2,psi2


# In[17]:


def initials(E=3/2,Xmin=-5,Xmax=5,psi_0=10**(-30),psi_1=10**(-29),div=10**4):
    '''
    Xmin,Xmax=minimum and maximum of the range
    div denotes the number of divisions for X
    '''
    X=np.linspace(Xmin,Xmax,div)
    dx=X[2]-X[1]
    f_0=F(X[0],E)
    f_1=F(X[1],E)
    q_0=psi_0*(1-dx**2*f_0/12)
    q_1=psi_1*(1-dx**2*f_1/12)
    psi=[psi_0,psi_1]
    f=[f_0,f_1]
    q=[q_0,q_1]
    return X,psi,f,q,dx


# In[26]:


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
    return X,psi,f,q,dx


# In[5]:


def run_eq(X,q,f,psi,dx,E):
    #print(eps)
    for i in range(len(X)-2):
# q2,f2,psi2=Numrov(X[i+1],q[-2],q[-1],psi[-1],f[-1],dx,eps)
        x=X[i+1]
        f1=f[-1]
        psi1=psi[-1]
        q1=q[-1]
        q0=q[-2]
        q2,f2,psi2=Numrov(x,q0,q1,psi1,f1,dx,E)
        q.append(q2)
        f.append(f2)
        psi.append(psi2)
    psi_n=Normalize(X,psi)
    return X,psi_n


# In[6]:


def run_mult(range_eps,Ra):
    data=[]
    i=0
    for eps in range_eps:
        X,psi,f,q,dx=initials(eps,Ra[0],Ra[1])
        X,n_psi=run_eq(X,q,f,psi,dx,eps)
        data.append([X,n_psi])
        if len(range_eps)>50:
            sys.stdout.flush()
            sys.stdout.write("\r{0}>".format('#'*(i+1)))
        i+=1
    return data


# In[7]:


def Eg_optim(E_init,num_prec,peak=[-1,1],Ra=[-5,5],div=.1):
    '''
    Energy optimizer function
    this function recieves an enegry assumption and optimizes this to an energy value which 
    will satisfy the boundary condition best
    this code follows similar logic the previously defined eigen finder
    
    E_init:-> initial assumption of E
    num_prec:-> the precision we require for of the eigen energy
    {num_pec denotes the number of decimal places till the eigen value is correct}
    higer the num_prec more accurate the eigen energy is 
    higer the bum_prec more time it takes to compute
    '''
    print('\n')
    Dyn_E=E_init#We consider 2 values of E near to this Dynamic E
    dE=div#stores the range in which we check for an optimum(the values are:-> E-dE,E,E+dE)
    Dummy_data=run_mult([Dyn_E],Ra)#X is same for all
    x=Dummy_data[0][0]
    x_0=peak[0]
    x_l=peak[1]
    imin=np.where((x>x_0) & (x<(x_0+.1)))[0][0]# getting the range of indices near the peak region
    imax=np.where((x>x_l)&(x<(x_l+.1)))[0][0]#here our range is the peak range given
    opt_count=0#denotes the number of precision accuired, initialized to be zero
    print(f'Input value for optimization:{E_init}')
    #A loop is run until the needed optimization is achived 
    i=0
    while opt_count<=num_prec:
        '''
        Method: We run the different values of E and accquire an array called explo
        (explotion factor),the explo array keeps the ratio of last value of psi calculated 
        to the maxium found in the expected maximum range. the values in explo will be v
        ery large when the energy is not eigen.when a local minimum is aqcuired in Explo 
        that range of E is magnified to find a more precise value of Eigen Energy.
        '''
        E_range=[Dyn_E-dE,Dyn_E,Dyn_E+dE]#Our epsilon range 3 psi values for which we try the plotting
        q_s=run_mult(E_range,Ra)#getting the plot of quantum states for different energy value
        E_optim=Explo_ary(E_range,[imin,imax],Ra)[0]#running for 3 energies
        if E_optim==E_range[0] or E_optim==E_range[2] :
            Dyn_E=E_optim
        else:
            dE=dE/10.0
            opt_count+=1
        sys.stdout.flush()
        sys.stdout.write("\r{0}".format(f"\tEnergy Value after {1+i} optimizations:->{Dyn_E}"))
        i+=1
    if Dyn_E<0:
        print('Optimization failed')
    else:
        print(f'\nValue  after optimization =>{Dyn_E}')
        return Dyn_E


# In[8]:


def Explo_ary(E_array,peak_ind,Ra=[-5,5]):#function which will return the explotion array
    '''
    This function computes the explotion factor i,e the ration between last point in psi
    to the maximum near the expected region
    '''
    optim_eng=[]#stores the energies which are at closer to eigen enrgies
    imin=peak_ind[0]#the index marking the begining of the region where we ecpect a max
    imax=peak_ind[1]#the index marking the end of the region where we expect a maximum
    q_s=run_mult(E_array,Ra)#getting the plot of quantum states for different energy value
    Explo=[]#array to store the fractional difference
    lg=len(E_array)
    for j in range(lg):#since we already know there are only 3 plots we calculated
            Y=q_s[j][1]#psi values are diffrent for different epsilons
            expec_max=0# The maximum near the expected reagion
            for y in Y[imin:imax]: #finding the maximum in our expected peak region
                if abs(y)>expec_max:
                    expec_max=abs(y)
            exp_f=abs(Y[-1]/expec_max)# the explotion factor for each energy
            Explo.append(exp_f)
            Xplo_e.append([exp_f,E_array[j]])
    Explo=np.copy(Explo)
    if lg<4:#in case there are only 3 values we need the minimum finding the local minimum is only fruitful when we have at least 6 values
        ind=np.where(Explo==Explo.min())[0][0]
        optim_eng=[E_array[ind]]
    else:# in this case we find the local minimums
        for i in range(lg-2):
            if Explo[i+1]<Explo[i]:
                if Explo[i+1]<Explo[i+2]:
                    optim_eng.append(E_array[i+1])
    return optim_eng


# In[9]:


def Energy_finder(E_est,num_prec=5,peak=[-1,1],Ra=[-5,5],div=.1):
    '''
    Returns the maximum possible accurate value of eigen energy based on the 
    prececuion given
    '''
    E_eigen=[]
    E_array=np.linspace(E_est-div,E_est+div,20)# an array with .01 differences
    dE=div#stores the range in which we check for an optimum(the values are:-> E-dE,E,E+dE)
    Dummy_data=run_mult([E_array[0]],Ra)#X is same for all
    x=Dummy_data[0][0]
    x_0=peak[0]
    x_l=peak[1]
    imin=np.where((x>x_0) & (x<(x_0+div)))[0][0]# getting the range of indices near the peak region
    imax=np.where((x>x_l)&(x<(x_l+div)))[0][0]#here our range is the peak range given
    Eg_ar=[]
    E_est=round(E_est,2)
    print(f"\nSuspected eigen energy value lies between {E_est-div}-->{E_est+div}\n")
    for i in range(num_prec):
        #print(E_array[0],'\t',E_est,'\t',E_array[-1],'\t',dE)
        Eg_ar=Explo_ary(E_array,[imin,imax],Ra)
        dE=dE/10
        if len(Eg_ar)>1:
            for e in Eg_ar:
                E_eigen.append(Eg_optim(e,(num_prec-i-1),peak,Ra,dE))
                #E=np.copy(E_eigen)
            return E_eigen#E.flatten()
        else:
            E_est=Eg_ar[0]
            E_array=np.linspace(E_est-1*dE,E_est+1*dE,20)
        sys.stdout.flush()
        sys.stdout.write("\r{0}".format(f'\tEnergy search range {E_array[0]}->{E_array[-1]}'))
    return Eg_ar


# In[10]:


def Energy_loc(E_range,peak_r,X_range=[-5,5],div=.1):
    E_r=np.arange(E_range[0],E_range[1],div)
    d=run_mult(E_r,X_range)
    X=d[0][0]
    x_0=peak_r[0]
    x_l=peak_r[1]
    Eigen_E=[]#array to store eigen energies
    n=len(E_r)#number of energy values we are considering
    imin=np.where((X>x_0) & (X<(x_0+.1)))[0][0]# getting the range of indices near the peak region
    imax=np.where((X>x_l)&(X<(x_l+.1)))[0][0]#here our range is the peak range given
    Eng=Explo_ary(E_r,[imin,imax])
    for E in Eng:         
        Eigen_E.append(Energy_finder(E,5,peak_r,X_range))
    return Eigen_E


# In[11]:


def Normalize(x,y,norml_Val=1):#function to normalize the function
    A=0
    norm_y=[]
    for i in range(len(x)-1):
        a=abs((x[i+1]-x[i])*(y[i]+y[i+1])/2)
        A=A+a
    for i in range(len(x)):
        norm_y.append(y[i]/A)
    return norm_y


# In[13]:


def Merger(X,psi,X_b,psi_b):
    '''
    Given the both forward and backward solution this function merges both of them together.
    An explotion near the end of the forward. othewise this tep is not needed 
    X_b starts from 10 till -10(towards negatieve direction)
    '''
    psi_b=np.copy(psi_b[::-1])
    psi_m=np.copy(psi)# the final merged psi 
    last_min=0#stores the postion of last minimum in the psi array this indicates the minimum before the explotion
    for p in range(len(psi)-2):
        if abs(psi_m[p])>abs(psi_m[p+1]):  
            if abs(psi_m[p+1])<abs(psi_m[p+2]):
                last_min=p+1   
    psi_m[last_min:]=psi_b[last_min:]#merging both of them
    psi_n=Normalize(X,psi_m)
    return psi_n





def Run_both(eps=1,Xmin=-10,Xmax=10,psi_0=1e-30,psi_1=1e-30,div=10**5):
    '''
    This function splirts the x axis into two and runs two function one from negative till x=0
    another from positive.
    xmin has to be a negatieve number and Xmax a positieve number, othewise this step donot work
    This gives a functiojn which satisfies boundary condition irrespective of the eigen value
    So FOR SOLVING EIGEN VALUES THIS FUNCTION SHOULD NOT BE USED
    '''
    X,psi,f,q,dx=initials(eps,Xmin,Xmax,psi_0,psi_1,div)
    X_b,psi_b,f_b,q_b,dx=initialsBk(eps,Xmin,Xmax,psi_0,psi_1,div)
    run_eq(X,q,f,psi,dx,eps)
    run_eq(X_b,q_b,f_b,psi_b,dx,eps)
    N_psi=Merger(X,psi,X_b,psi_b)
    return X,N_psi
    





def Plot_Eq(E_range,Xmin=-10,Xmax=10):
    """
    Recieves the energies for which we need the plots. 
    And the range of x axis to be covered
    plots the respective plots after running using the
    Run_both function
    """
   
    for e in E_range:
        x,Psi=Run_both(e,Xmin,Xmax)
        ax.plot(x,Psi+e,label=e)
    ax.plot(X,Poten(X),label='Potential function (V)')

h=1
m=1
w=1
n=0.0
C_4=4
C_2=40
C0=10
K=1/10
exp_max_r=[-2.5,-1.5]# the range between which we are more likely yo find a probability peak

Xplo_e=[]
b=time.time()
E=Energy_loc([0,10],exp_max_r)
TT=time.time()-b
print('\n Time taken for computing eigen values= ',TT)


# In[40]:


Eigen_Eg=[]
i=1
for e in E:
    for erg in e:
        Eigen_Eg.append(erg)
        print(f"Eigen energy {i}={erg}")
        i+=1


# In[51]:


X=np.linspace(-10,10,10**3)
fig=plt.figure()
ax=plt.axes(xlabel='x',ylabel='Psi',xlim=(-10,10),ylim=(-0,12))
ax.legend()
ax.set_title("Solutions of Schrodinger equations \n for Quantum Double Potential Well")
Plot_Eq(Eigen_Eg,-10,10)
plt.show()