
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
#approximate time needed for running = 11min

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

def Numrov(x,q0,q1,psi1,f1,dx,E):
    q2=(dx**2)*f1*psi1+2*q1-q0
    f2=F(x,E)
    psi2=q2/(1-f2*dx**2)
    return q2,f2,psi2

def initials(E=3/2,Xmin=-5,Xmax=5,psi_0=1e-30,psi_1=1e-29,div=int(1e5)):
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

def run_eq(X,q,f,psi,dx,E):
    for i in range(len(X)-2):
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

def run_mult(range_eps,Ra):
    data=[]
    i=0
    for eps in range_eps:
        X,psi,f,q,dx=initials(eps,Ra[0],Ra[1])
        X,n_psi=run_eq(X,q,f,psi,dx,eps)
        data.append([X,n_psi])
        if len(range_eps)>50:
            sys.stdout.flush()
            sys.stdout.write("\r{0}>>".format('#'*int((i+1)*20/len(range_eps))))
        i+=1
    return data

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
    Dyn_E=E_init#We consider 2 values of E near to this Dynamic E
    dE=div#stores the range in which we check for an optimum(the values are:-> E-dE,E,E+dE)
    Dummy_data=run_mult([Dyn_E],Ra)#X is same for all
    x=Dummy_data[0][0]
    x_0=peak[0]
    x_l=peak[1]
    imin=np.where((x>x_0) & (x<(x_0+.1)))[0][0]# getting the range of indices near the peak region
    imax=np.where((x>x_l)&(x<(x_l+.1)))[0][0]#here our range is the peak range given
    opt_count=0#denotes the number of precision accuired, initialized to be zero
    print(f'\n\tInput value for optimization:{E_init}')
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
        sys.stdout.write("\r{0}".format(f"\t \tEnergy Value after {1+i} optimizations:->{Dyn_E}"))
        i+=1
    if Dyn_E<0:
        print('\nOptimization failed')
    else:
        print(f'\n\tValue  after optimization =>{Dyn_E}')
        return Dyn_E


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
        ind=np.where(Explo==Explo.min())[0][0]# this section code is exclusievely used by Eg_optim function
        optim_eng=[E_array[ind]]
    else:# in this case we find the local minimums
        for i in range(lg-2):
            if Explo[i+1]<Explo[i]:
                if Explo[i+1]<Explo[i+2]:
                    optim_eng.append(E_array[i+1])
    return optim_eng



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
	print(f"\nSuspected eigen energy value lies between {E_est-div}-->{E_est+div}")
	for i in range(num_prec):
		print(f'\tEnergy search range after {i+1} runs {E_array[0]}->{E_array[-1]}')
		Eg_ar=Explo_ary(E_array,[imin,imax],Ra)
		dE=dE/10
		if len(Eg_ar)>1:
			for e in Eg_ar:
				E_eigen.append(Eg_optim(e,(num_prec-i-1),peak,Ra,dE))
			print('\n')
			return E_eigen
		else:
			E_est=Eg_ar[0]
			E_array=np.linspace(E_est-1*dE,E_est+1*dE,20)
	return Eg_ar


def Energy_loc(E_range,peak_r,X_range=[-5,5],div=.1):
	E_r=np.arange(E_range[0],E_range[1],div)
	d=run_mult(E_r,X_range)
	X=d[0][0]
	x_0=peak_r[0]
	x_l=peak_r[1]
	Eigen_E=[]#array to store eigen energies
	#n=len(E_r)#number of energy values we are considering
	imin=np.where((X>x_0) & (X<(x_0+.1)))[0][0]# getting the range of indices near the peak region
	imax=np.where((X>x_l)&(X<(x_l+.1)))[0][0]#here our range is the peak range given
	Eng=Explo_ary(E_r,[imin,imax])
	i=0
	num=len(Eng)
	for E in Eng:
		com=round((i+1)*100/(num+.1),2)
		print(f'\t{com}% done')         
		Eigen_E.append(Energy_finder(E,5,peak_r,X_range))
		i=i+1
	return Eigen_E


def Normalize(x,y,norml_Val=1):#function to normalize the function
    A=0
    norm_y=[]
    for i in range(len(x)-1):
        a=abs((x[i+1]-x[i])*(y[i]+y[i+1])/2)
        A=A+a
        norm_y=y/A
    return norm_y

def plot_mult(Epsrange):
    pl=run_mult(Epsrange)
    i=0
    for eps in Epsrange:
        plt.plot(pl[i][0],pl[i][1],label=eps)
        i+=1
#changing w,h,m can shift the peaks from their currrent position and the program might fail
h=1
m=1
w=1
n=0.0
C_4=4
C_2=40
C0=10
K=1/10
exp_max_r=[-2.5,-1.5]# the range where we expect to find a probability peak

Xplo_e=[]
b=time.time()
E=Energy_loc([0,10],exp_max_r)
TT=time.time()-b
print('\nTime taken for calculation of 6 eigen energies=',TT)

Eigen_Eg=[]
for e in E:# to make the energy array single dimensional
    for erg in e:
        Eigen_Eg.append(erg)
Eigen_Eg
plot_mult(Eigen_Eg)
X=np.linspace(-5,5,10**5)#D[0][0]
plt.plot(X,Poten(X))
plt.show()






