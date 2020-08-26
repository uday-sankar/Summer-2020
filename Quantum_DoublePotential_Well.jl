using Pkg
using Plots

function Potential(x)
    V=K*(C_4*x^4-C_2*x^2+C_0)
    return V
end
function F(x,E)
    v=Potential(x)
    return 2m*(v-E)/h^2
end

function Initialize(En=1,X=LinRange(-5,5,10^5),psi_0=1e-10,psi_1=1e-10)
    dx=X[2]-X[1]
    f_0=X[1]^2-En
    f_1=X[2]^2-En
    q_0=psi_0*(1-dx^2*f_0/12)
    q_1=psi_1*(1-dx^2*f_1/12)
    psi=[psi_0,psi_1]
    f=[f_0,f_1]
    q=[q_0,q_1]
    return psi,f,q,dx
end

function Numerov(x,q0,q1,psi1,f1,dx,En)
    q2=dx^2*f1*psi1+2q1-q0
    f2=F(x,En)
    psi2=q2/(1-f2*dx^2/12)
    return q2,f2,psi2
end

function Normalize(X,Y)
    A=0
    for i in 1:(size(X,1)-1)
        a=(X[i+1]-X[i])*(Y[i]+Y[i+1])/2
        A=A+abs(a)
    end
    norm_y=Y./A
    return norm_y
end

function run_eq(X,q,f,psi,dx,En)
    for i in 1:size(X,1)-2
        x=X[i+1]
        f1=f[i+1]
        psi1=psi[i+1]
        q1=q[i+1]
        q0=q[i]
        q2,f2,psi2=Numerov(x,q0,q1,psi1,f1,dx,En)
        append!(q,q2)
        append!(f,f2)
        append!(psi,psi2)
    end
    psi_norm=Normalize(X,psi)
    return psi_norm
end

function run_mult(En_rang,X=Linrange(-5,5,10^5),psi_0=1e-10,psi_1=1e-10)
    Pl=[]#D plot array to store all the psi values storing
    for e in En_rang
        psi,f,q,dx=Initialize(e,X,psi_0,psi_1)
        psi_n=run_eq(X,q,f,psi,dx,e)
        append!(Pl,[[psi_n,e]])
    end
    return Pl
end

function EgOptim(En_ini,X,num_prec=10,spike=[-1,1],div=.1)
    #=
    eps_init->The initial epsilon value we are giving, which is to be
    optimized
    num_step->the number of times the algorithm to be run
    div->denoting the other values to be tested
    In this algo we find the most appropriate espilon among
    eps_ini-div,eps_ini and eps_ini+div
    if we
    =#
    ind=findall(a->a>=spike[1]&&a<=spike[2],X)#indexes between the range -2to2
    str_ind=ind[1]
    stp_ind=ind[end]
    i=0
    while i<=num_prec
        E_range=[En_ini-div,En_ini,En_ini+div]# the 3 values foe which we find the explotion values
        d=run_mult(E_range,X)#solving the equation for the 3 values
        Explo_F=zeros(3)
        for i in 1:3
            Y=d[i][1]
            spike_max=maximum(Y[str_ind:stp_ind])#The maximum at the spike region
            spike_min=abs(minimum(Y[str_ind:stp_ind]))#The absolute of minimum at the spike region
            if spike_max<spike_min#if the wave has maximum below x axis
                spike_max=spike_min
            end
            explo_f=abs(Y[end]/spike_max)#the explo factor
            Explo_F[i]=explo_f
        end
        pmin=argmin(Explo_F)#position of mimimum
        if pmin==2
            div=div/2
            i+=1
        else
            En_ini=E_range[pmin]
        end
    end
    return En_ini
end

function Explo_facFinder(E_array,str_ind,stp_ind,X=LinRange(-10,10,10^5))
    #=
    This functions return the energy corresponding to a local
    minimum in the Explotion factor
    =#
    Optim_en=[]
    Sol=run_mult(E_array,X)
    l=size(E_array)[1]
    Explo_F=zeros(l)
    for i in 1:l
        Y=Sol[i][1]
        spike_max=maximum(Y[str_ind:stp_ind])#The maximum at the spike region
        spike_min=abs(minimum(Y[str_ind:stp_ind]))#The absolute of minimum at the spike region
        if spike_max<spike_min#if the wave has maximum below x axis
            spike_max=spike_min
        end
        explo_f=abs(Y[end]/spike_max)#the explo factor
        Explo_F[i]=explo_f
    end
    for i in 1:(l-2)
        if Explo_F[i]>Explo_F[i+1] && Explo_F[i+1]<Explo_F[i+2]
            append!(Optim_en,E_array[i+1])
        end
    end
    return Optim_en
end

function Eg_Isolater(En_init,X,num_prec=5,spike=[-1,1],div=.1)
    #=
    Energy isolater function tries to find eigen enrgies eperated
    by small energy gap. till the range of 1e-5
    tries to find the minimum acroos a range of values and
    optimizes the result
    =#
    ind=findall(a->a>=spike[1]&&a<=spike[2],X)#indexes between the range -2to2
    str_ind=ind[1]
    stp_ind=ind[end]
    dE=div::Float64
    Eigen_E=[]
    i=0
    while i<=num_prec
        E_array=[En_init+i*dE for i in -9:9]
        Optim_en=Explo_facFinder(E_array,str_ind,stp_ind,X)
        if size(Optim_en)[1]>1#if multiple enrgies are present
            for e in Optim_en# then each of those is optimized
                OptimE=EgOptim(e,X,5,spike,div)#by EgOptim
                append!(Eigen_E,OptimE)
            end
            return Eigen_E
        else
            En_init=Optim_en[1]
            dE=dE/10
            i=i+1
        end
    end
    append!(Eigen_E,En_init)
    return Eigen_E
end

function EigenLoc(EMin,EMax,X,spike,div=.1)
    #=
    Eigen enrgry locator function divides the Energy gap into discreate
    units and try to find possible eigen enrgies
    =#
    EG_range=EMin:div:EMax#the enrgy range in which we try to find Eigen energies
    ind=findall(a->a>=spike[1]&&a<=spike[2],X)#indexes between the range -2to2
    str_ind=ind[1]
    stp_ind=ind[end]
    Eigen_Energy_array=[]#Array to store the eigen enrgies
    Eg_approx=Explo_facFinder(EG_range,str_ind,stp_ind,X)#the approximate position of eigen value
    for E in Eg_approx
        Eng=Eg_Isolater(E,X,10,spike,0.01)
        append!(Eigen_Energy_array,Eng)
    end
    return Eigen_Energy_array
end

function plot_mult(X,Plot_data)
    for y in Plot_data
        plot!(X,y[1].+y[2],label=y[2])
    end
end

h=1
m=1
w=1
K=.1
C_4=4
C_2=40
C_0=C_2^2/(4*C_4)

X_test=LinRange(-10,10,10^4)
X_run=LinRange(-4,4,10^5)
plotly()

spike=[-2.5,-1.5]
E=EigenLoc(1,10,X_test,spike)
print(E)
pl=run_mult(E,X_run)
plot()
plot_mult(X_run,pl)
display(plot!())
