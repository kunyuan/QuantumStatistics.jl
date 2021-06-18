"""
Power method, damp interation and implicit renormalization
"""
#module eigensolver
using LinearAlgebra

"""
Function Separation

Separate Gap function to low and high energy parts with given rule
"""
function Separation(mom_grid, freq_grid)
    cut=0.25
    low=zeros(size(mom_grid)[1],size(freq_grid)[1])
    high=zeros(size(mom_grid)[1],size(freq_grid)[1])
    for i in 1:size(mom_grid)[1]
        for j in 1:size(freq_grid)[1]
            if(freq_grid[j]<cut)
                low[i,j]=1
            else
                high[i,j]=1
            end
        end
    end
    return low,high
end

"""
Function delta_to_F

Convert delta to F function by doing:
F=delta*GG
"""
function delta_to_F(delta, freq_in, tau_in)
    
end

"""
Function Implicit_Renorm

    For given kernal, use implicit renormalization method to solve the eigenvalue

"""


function Implicit_Renorm(delta ,kernal , mom_in, freq_in, tau_in, NN, rtol )
    Looptype=0
    n=0
    err=1.0 
    accm=0
    shift=2.0
    lamu0=-2.0
    n_change=20  "steps of damp iteration in on complete loop"
    n_change2=20+10 "total steps of one complete loop "
    d_accm=zeros(size(mom_in)[1],size(freq_in)[1])
    to_low, to_high=Separation(mom_in, freq_in)
    
    while(n<NN && err>rtol)
        F=delta_to_F(delta, mom_in , freq_in, tau_in)
        n=n+1
        delta_new=Integral(F, kernal, mom_diff, freq_grid)
        delta_low=delta_new.*to_low
        delta_high=delta_new.*to_high
        if(Looptype==0)
            accm=accm+1
            d_accm=d_accm+delta_high
            delta=(delta.*to_low+d_accm./accm)
        else
            lamu=dot(delta.*to_low, delta_low)
            delta_low=delta_low+shift*delta.*to_low
            modulus=sqrt(dot(delta_low, delta_low))
            delta=(delta_low/modulus+delta.*to_high)
        end
        if(n%n_change2==n_change)
            Looptype=(Looptype+1)%2
        elseif(n%n_change2==0)
            accm=0
            d_accm0=0*d_accm
            err=abs(lamu-lamu0)
            lamu0=lamu
            println(lamu)
            Looptype=(Looptype+1)%2
        end
        
    end
end


