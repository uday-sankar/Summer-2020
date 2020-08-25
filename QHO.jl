using Pkg
using Plots

F(x,e)=x^2-e

function Initialize(eps=1,X=LinRange(-5,5,10^5),psi_0=1e-10,psi_1=1e-10)
    dx=X[2]-X[1]
    f_0=X[1]^2-eps
    f_1=X[2]^2-eps
    q_0=psi_0*(1-dx^2*f_0/12)
    q_1=psi_1*(1-dx^2*f_1/12)
    psi=[psi_0,psi_1]
    f=[f_0,f_1]
    q=[q_0,q_1]
    return psi,f,q,dx
end

function Numerov(x,q0,q1,psi1,f1,dx,eps)
    q2=dx^2*f1*psi1+2q1-q0
    f2=F(x,eps)
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

function run_eq(X,q,f,psi,dx,eps)
    for i in 1:size(X,1)-2
        x=X[i+1]
        f1=f[i+1]
        psi1=psi[i+1]
        q1=q[i+1]
        q0=q[i]
        q2,f2,psi2=Numerov(x,q0,q1,psi1,f1,dx,eps)
        append!(q,q2)
        append!(f,f2)
        append!(psi,psi2)
    end
    psi_norm=Normalize(X,psi)
    return psi_norm
end

function run_mult(Ep_rang,X=Linrange(-5,5,10^5),psi_0=1e-10,psi_1=1e-10)
    Pl=[]#D plot array to store all the psi values storing
    for e in Ep_rang
        psi,f,q,dx=Initialize(e,X,psi_0,psi_1)
        psi_n=run_eq(X,q,f,psi,dx,e)
        append!(Pl,[[psi_n,e]])
    end
    return Pl
end

function plot_mult(X,Plot_data)
    for y in Plot_data
        plot!(X,y[1],label=y[2])
    end

end

function EigenFinder01(eps_ini,X,num_step,spike=[-1,1],div=.1)
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
    for i in 1:num_step
        E_range=[eps_ini-div,eps_ini,eps_ini+div]# the 3 values foe which we find the explotion values
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
        else
            eps_ini=E_range[pmin]
        end
    end
    return eps_ini
end

function EigenFinder02(min_e,max_e,num_exp,X,spike=[-1,1])
    #=
    min_e-> the lower bound for energy/epsilon in our searching function
    max_e->the upper bound for the same
    num_exp-> the expected number of eigen value between the range
    We divide our intervel into many points and try to optimize each value
    =#
    Eps=[]#array storing the eigen epsilon/energy value
    M_eRange=LinRange(min_e,max_e,num_exp)#main energy/epsilon range divided into many parts
    for i in 1:num_exp
        eps=EigenFinder01(M_eRange[i],X,15,spike)#optimizind each of the values
        eps=round(eps,digits=2)#rounding to 2 decimal places
        if size(Eps,1)>=1
            pos=findall(a->a==eps,Eps)#findinf position of eps
            if size(pos,1)==0#if eps is already present in the array this will not be zero
                append!(Eps,eps)
            end
        else
            append!(Eps,eps)
        end
    end
    return Eps
end

X=LinRange(-7,7,10^5)

EPS=EigenFinder02(0,10,10,X)
pl=run_mult(EPS,X)
gr()
plot_mult(X,pl)
display(plot!())
#=E=EigenFinder01(1.5,X,20)
psi,f,q,dx=Initialize(E,X)
run_eq(X,q,f,psi,dx,E)
display(plot!(X,psi))
pl=run_mult([1,3,5],X)

gr()
plot_mult(X,pl)
display(plot!())
=#
