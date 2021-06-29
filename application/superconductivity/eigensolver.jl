"""
Power method, damp interation and implicit renormalization
"""
#module eigensolver
using QuantumStatistics
using LinearAlgebra
using Printf
#using Gaston
using Plots
include("parameter.jl")
include("eigen.jl")
include("grid.jl")

"""
Function Separation

Separate Gap function to low and high energy parts with given rule
"""
function Separation(delta0, delta, k::CompositeGrid,  fdlr)
    # cut=0.25
    # low=zeros(size(mom_grid)[1],size(freq_grid)[1])
    # high=zeros(size(mom_grid)[1],size(freq_grid)[1])
    # for i in 1:size(mom_grid)[1]
    #     for j in 1:size(freq_grid)[1]
    #         if(freq_grid[j]<cut)
    #             low[i,j]=1
    #         else
    #             high[i,j]=1
    #         end
    #     end
    # end
    #mom_sep = 0.1
    low=zeros(Float64, ( length(k.grid), fdlr.size))
    high=zeros(Float64, ( length(k.grid), fdlr.size))
    low0=zeros(Float64,  length(k.grid))
    high0=zeros(Float64,  length(k.grid))
    for i in 1:fdlr.size
        for (pi,p) in enumerate(k.grid)
            if(abs(p-kF)<mom_sep)
                low[pi,i]=1
                low0[pi]=1
            else
                high[pi,i]=1
                high[pi]=1
            end
        end
    end
    delta_0_low = low0 .* delta0
    delta_0_high = high0 .* delta0
    delta_low = low .* delta
    delta_high = high .* delta

    # for j in 1:length(mom_grid)
    #     for (ωi, ω) in enumerate(fdlr.ω)
    #         if(abs.(ω)<cut)
    #             low[ωi,j]=1
    #         else
    #             high[ωi,j]=1
    #         end
    #     end
    # end
    # coeff_low = coeff .* low
    # coeff_high = coeff .* high
    # delta_low =  DLR.dlr2matfreq(:acorr, coeff_low, fdlr, fdlr.n, axis=1)
    # delta_high = DLR.dlr2matfreq(:acorr, coeff_high, fdlr, fdlr.n, axis=1)

    # delta0_dum = zeros(ComplexF64, (fdlr.size, length(mom_grid)))
    # for j in 1:length(mom_grid)
    #     for (ωi, ω) in enumerate(fdlr.ω)
    #         delta0_dum[ωi,j]=delta0[j]
    #     end
    # end


    # coeff = DLR.matfreq2dlr(:acorr, delta0_dum, fdlr, axis=1)
    # println("coeff=",coeff[:,1])
    # coeff_low = coeff .* low
    # coeff_high = coeff .* high
    # delta_low = delta_low .+ DLR.dlr2matfreq(:acorr, coeff_low, fdlr, fdlr.n, axis=1)
    # delta_high = delta_high .+ DLR.dlr2matfreq(:acorr, coeff_high, fdlr, fdlr.n, axis=1)
    
    return delta_0_low, delta_0_high, delta_low, delta_high
end

function Separation_F(F_in,  k::CompositeGrid,  fdlr)
    # cut=0.25
    # low=zeros(size(mom_grid)[1],size(freq_grid)[1])
    # high=zeros(size(mom_grid)[1],size(freq_grid)[1])
    # for i in 1:size(mom_grid)[1]
    #     for j in 1:size(freq_grid)[1]
    #         if(freq_grid[j]<cut)
    #             low[i,j]=1
    #         else
    #             high[i,j]=1
    #         end
    #     end
    # end
    F=F_in
    cut = 0.25
    coeff = DLR.tau2dlr(:acorr, F, fdlr, axis=2)
    #println(coeff[1:10,20])
    low=zeros(Float64, ( k.Np * k.order, fdlr.size))
    high=zeros(Float64, ( k.Np * k.order, fdlr.size))
    
    for (ωi, ω) in enumerate(fdlr.ω)
        for j in 1:(k.Np * k.order)
            if(abs.(ω)<cut)
                low[j,ωi]=1
            else
                high[j,ωi]=1
            end
        end
    end
    coeff_low = coeff .* low
    coeff_high = coeff .* high
    F_low =  DLR.dlr2tau(:acorr, coeff_low, fdlr, fdlr.τ, axis=2)
    F_high = DLR.dlr2tau(:acorr, coeff_high, fdlr, fdlr.τ, axis=2)

    return real.(F_low), real.(F_high)
end



"""
Function Implicit_Renorm

    For given kernal, use implicit renormalization method to solve the eigenvalue

"""


function Implicit_Renorm(kernal, kgrid, qgrids, fdlr )
    NN=1000
    rtol=1e-6
    Looptype=1
    n=0
    err=1.0 
    accm=0
    shift=4.0
    lamu0=-2.0
    lamu=0.0
    n_change=10  #steps of power method iteration in one complete loop
    n_change2=10+10 #total steps of one complete loop

    delta = zeros(Float64, (length(kgrid.grid), fdlr.size))
    delta_0 = zeros(Float64, length(kgrid.grid)) .+ 1.0
    #Separate Delta
    d_0_accm=zeros(Float64, length(kgrid.grid))
    d_accm=zeros(Float64, (length(kgrid.grid), fdlr.size))
    delta_0_low, delta_0_high, delta_low, delta_high = Separation(delta_0, delta, kgrid, fdlr)
    while(n<NN && err>rtol)
        F=calcF(delta_0, delta, fdlr, kgrid)
        n=n+1
        delta_0_new, delta_new =  calcΔ(F, kernal, fdlr , kgrid, qgrids)./(-4*π*π)
        delta_0_low_new, delta_0_high_new, delta_low_new, delta_high_new = Separation(delta_0_new, delta_new, kgrid, fdlr)
        if(Looptype==0)
            accm=accm+1
            d_0_accm = d_0_accm + delta_0_high_new 
            d_accm = d_accm + delta_high_new
            delta_0_high = d_0_accm ./ accm
            delta_high = d_accm ./ accm
        else
            lamu = dot(delta_0_low_new, delta_0_low)
            delta_0_low_new = delta_0_low_new+shift*delta_0_low
            delta_low_new = delta_low_new+shift*delta_low
            modulus = sqrt(dot(delta_0_low_new, delta_0_low_new))
            delta_0_low = delta_0_low_new ./ modulus
            delta_low = delta_low_new ./ modulus
            #println(lamu)
        end
        delta_0 = delta_0_low + delta_0_high
        delta = delta_low + delta_high
        if(n%n_change2==n_change)
            Looptype=(Looptype+1)%2
        elseif(n%n_change2==0)
            accm = 0
            d_accm = d_accm .* 0
            d_0_accm = d_0_accm .* 0
            err=abs(lamu-lamu0)
            lamu0=lamu
            println(lamu)
            Looptype=(Looptype+1)%2
        end
        
    end

    #Separate F
    # F_accm=zeros(Float64, (length(kgrid.grid), fdlr.size))
    # F=calcF(delta_0, delta, fdlr, kgrid)
    # F_low, F_high=Separation_F(F, kgrid, fdlr)
    # while(n<NN && err>rtol)
    #     n=n+1
    #     delta_0_new, delta_new = calcΔ(F, kernal, fdlr , kgrid, qgrids)./(-4*π*π)
    #     F_new=calcF(delta_0_new, delta_new, fdlr, kgrid)
    #     F_low_new, F_high_new = Separation_F(F_new, kgrid, fdlr)
    #     println(Looptype)
    #     if(Looptype==0)
    #         accm=accm+1
    #         F_accm=F_accm+F_high_new
    #         F_high=F_accm./accm
    #     else
    #         lamu=dot(F_low, F_low_new)
    #         F_low_new=F_low_new+shift*F_low
    #         modulus=sqrt(dot(F_low_new, F_low_new))
    #         F_low=F_low_new/modulus
    #         println(lamu)
    #     end
    #     F = F_low+F_high
    #     if(n%n_change2==n_change)
    #         Looptype=(Looptype+1)%2
    #     elseif(n%n_change2==0)
    #         accm=0
    #         F_accm=0*F_accm
    #         err=abs(lamu-lamu0)
    #         lamu0=lamu
    #         println(lamu)
    #         Looptype=(Looptype+1)%2
    #     end
    # end
    filename = "../data/flow_$(WID).dat"
    open(filename, "w") do io
        @printf(io, "%32.17g\n", lamu)
    end
    return delta_0, delta
end

function Explicit_Solver(kgrid, qgrids, fdlr,fdlr2, bdlr )
    NN=1000
    rtol=1e-6
    n=0
    modulus= 1.0
    err=1.0 
    shift=4.0
    lamu0=-2.0
    lamu0_2= -2.0
    lamu=0.0
    lamu2=0.0
    #kernal = dH1_freq(kgrid, qgrids, bdlr, fdlr)
    kernal = dH1_tau(kgrid, qgrids, fdlr)
    #kernal_double = dH1_tau(kgrid_double, qgrids_double, fdlr)
    kernal_2 = dH1_tau(kgrid, qgrids, fdlr2)
    delta = zeros(Float64, (length(kgrid.grid), fdlr.size))
    for (ki, k) in enumerate(kgrid.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            delta[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
        end
    end
    delta = real(DLR.matfreq2tau(:acorr, delta, fdlr, fdlr.τ, axis=2))
    delta_0 = zeros(Float64, length(kgrid.grid)) #.+ 1.0
    delta_0_new=zeros(Float64, length(kgrid.grid))
    delta_new=zeros(Float64, (length(kgrid.grid), fdlr.size))

    Δ_2 = zeros(Float64, (length(kgrid.grid), fdlr2.size))
    Δ0_2 = zeros(Float64, length(kgrid.grid)) #.+ 1.0
    Δ0_2_new=zeros(Float64, length(kgrid.grid))
    Δ_2_new=zeros(Float64, (length(kgrid.grid), fdlr2.size))
    Δ_2_int = zeros(Float64, (length(kgrid.grid), fdlr.size))

    for (ki, k) in enumerate(kgrid.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr2.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            Δ_2[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)

        end
    end
    Δ_2 = real(DLR.matfreq2tau(:acorr, Δ_2, fdlr2, fdlr2.τ, axis=2))

    for (ki, k) in enumerate(kgrid.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            # F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            Δ_2_int[ki, ni] =  Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
        end
    end
    Δ_2_int = real(DLR.matfreq2tau(:acorr, Δ_2_int, fdlr, fdlr.τ, axis=2))
    Δ_init = DLR.tau2dlr(:acorr, Δ_2, fdlr2, axis=2)
    Δ_init = DLR.dlr2tau(:acorr, Δ_init, fdlr2, [1.0e-12, β-1.0e-12] , axis=2)

    println("Δ_init=$(Δ_init[1,:] )")
    #Separate Delta
   
    while(n<NN)
        n=n+1
       
        # F_2 = calcF(Δ0_2, Δ_2, fdlr2, kgrid)

        # F_2_int = DLR.tau2dlr(:acorr, F_2, fdlr2, axis=2)
        # F = DLR.dlr2tau(:acorr, F_2_int, fdlr2, fdlr.τ , axis=2)


        # # test dlr err
        # # F_test =  DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # # F_test =  DLR.matfreq2tau(:acorr, F_test, fdlr, fdlr.τ, axis=2)
       
        # # println("err_F_real=",maximum(abs.(real.(F_test) - F)))
        # # println("err_F_imag=", maximum(abs.(imag.(F_test) )))
        # # println("err_F_ratio=",maximum(abs.(real.(F_test) - F)./abs.(F)))
        # delta_0_new, delta_new = calcΔ(F, kernal, fdlr, kgrid, qgrids)./(-4*π*π)
        
        # Δ0_2_new, Δ_2_new = calcΔ(F_2, kernal_2, fdlr2 ,kgrid, qgrids)./(-4*π*π)

        # Δ_freq = DLR.tau2matfreq(:acorr, delta_new, fdlr, fdlr.n, axis=2)
        # Δ_freq_2 = DLR.tau2matfreq(:acorr, Δ_2_new, fdlr2, fdlr.n, axis=2)
        # Δ_2_int = DLR.tau2dlr(:acorr, Δ_2_new, fdlr2, axis=2)
        # Δ_2_int = DLR.dlr2tau(:acorr, Δ_2_int, fdlr2, fdlr.τ , axis=2)
        # err0 = maximum(abs.(Δ0_2_new-delta_0_new))
        # err = maximum(abs.(Δ_2_int-delta_new))
        # err_freq = maximum(abs.(Δ_freq_2-Δ_freq))
        # ind=findmax(abs.(Δ0_2_new-delta_0_new))[2]

        # println("err0: $(err0), err: $(err),err_freq: $(err_freq) ,mom: $(kgrid.grid[ind[1]])")

        #compare F
        F=calcF(Δ0_2, Δ_2_int, fdlr, kgrid)
        F_2 = calcF(Δ0_2, Δ_2, fdlr2, kgrid)
        
        F_2_int = DLR.tau2dlr(:acorr, F_2, fdlr2, axis=2)
        F_2_int = DLR.dlr2tau(:acorr, F_2_int, fdlr2, fdlr.τ , axis=2)

        F_freq = DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        F_2_freq = DLR.tau2matfreq(:acorr, F_2, fdlr2, fdlr.n, axis=2)
        err0 = maximum(abs.(F_2_int-F))
        err = maximum(abs.(F_2_freq-F_freq))
        ind=findmax(abs.(F_2_int-F))[2]

        println("tau_err: $(err0), freq_err: $(err) ,mom: $(kgrid.grid[ind[1]]), freq: $(fdlr.n[ind[2]])")
        # p = plot(fdlr.τ, F_2_int[ind[1],:] )
        # p = plot!(fdlr.τ, F[ind[1],:])
        # # test dlr err
        # # F_test =  DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
        # # F_test =  DLR.matfreq2tau(:acorr, F_test, fdlr, fdlr.τ, axis=2)
        # display(p)
        # readline()
        # println("err_F_real=",maximum(abs.(real.(F_test) - F)))
        # println("err_F_imag=", maximum(abs.(imag.(F_test) )))
        # println("err_F_ratio=",maximum(abs.(real.(F_test) - F)./abs.(F)))
        Δ0_2_new, Δ_2_new = calcΔ(F_2, kernal_2, fdlr2 ,kgrid, qgrids)./(-4*π*π)  
       
 


        #test interpolation err
        #lamu = F_new[kf_label,1]/F[kf_label,1]
        # if(n%10==0)
        #     F_double =  zeros(Float64, (length(kgrid_double.grid), fdlr.size))
        #     for τi in 1:fdlr.size
        #         F_double[:, τi] = interpolate(F[:, τi], kgrid, kgrid_double.grid)
        #     end
        #     delta_0_double, delta_double = calcΔ(F_double, kernal_double, fdlr, kgrid_double, qgrids_double)./(-4*π*π)
        #     delta_0_fine = interpolate(delta_0_double, kgrid_double, kgrid.grid)
        #     printstyled("Max Err for Δ0 interpolation: ", maximum(abs.(delta_0_new - delta_0_fine)), "\n", color=:red)
        # end
        

        # lamu = dot(delta, delta_new)
        # delta_0_new=delta_0_new+shift*delta_0
        # delta_new=delta_new+shift*delta
        # modulus=sqrt(dot(delta_new, delta_new))
        # #modulus = abs(F_new[kf_label,1])
        # delta_0 = delta_0_new/modulus
        # delta = delta_new/modulus
        # println("$(modulus), $(maximum(abs.(Δ_2)))")
        # err=abs(lamu-lamu0)
        # lamu0=lamu
        # println(lamu)




        lamu2 = dot(Δ_2, Δ_2_new)
        Δ0_2_new=Δ0_2_new+shift*Δ0_2
        Δ_2_new=Δ_2_new+shift*Δ_2
       
        #modulus = abs(F_new[kf_label,1])
        err0 = maximum(abs.(abs.(Δ_2_new)-abs.(Δ_2*modulus)))
        err = maximum(abs.(abs.(Δ0_2_new)-abs.(Δ0_2*modulus)))
        ind=findmax(abs.(abs.(Δ_2_new)-abs.(Δ_2*modulus)))[2]
        ind0=findmax(abs.(abs.(Δ0_2_new)-abs.(Δ0_2*modulus)))[2]
        
        modulus=sqrt(dot(Δ_2_new, Δ_2_new))
        println("Δ change dynamic: $(err0), static: $(err) ,mom_dy: $(kgrid.grid[ind[1]]), τ_dy: $(fdlr2.τ[ind[2]]), mom_st: $(kgrid.grid[ind0[1]]), value_st:$(Δ0_2_new[ind0[1]]), value_st_old:$(Δ0_2[ind0[1]]*modulus)")
        Δ0_2 = Δ0_2_new/modulus
        Δ_2 = Δ_2_new/modulus
        #println("$(modulus), $(maximum(abs.(Δ_2)))")
        lamu0_2=lamu2
        println(lamu2)
        
        # Δ_2_int_2 = DLR.tau2dlr(:acorr, Δ_2, fdlr2, axis=2)
        # Δ_2_int_2 = DLR.dlr2tau(:acorr, Δ_2_int_2, fdlr2, fdlr.τ , axis=2)

        # err0 = maximum(abs.(Δ0_2-delta_0))
        # err = maximum(abs.(Δ_2_int_2-delta))
        # #err_freq = maximum(abs.(Δ_freq_2-Δ_freq))
        # ind=findmax(abs.(Δ0_2-delta_0))[2]

        # println("err0: $(err0), err: $(err),err_freq: $(err_freq) ,mom: $(kgrid.grid[ind[1]])")
        
        Δ_2_int = DLR.tau2dlr(:acorr, Δ_2, fdlr2, axis=2)
        Δ_2_int = DLR.dlr2tau(:acorr, Δ_2_int, fdlr2, fdlr.τ , axis=2)


    end

    #Separate F
    # F=calcF(delta_0, delta, fdlr, kgrid)
    # while(n<NN)# && err>rtol)
    #     n=n+1
    #     delta_0_new, delta_new = calcΔ(F, kernal, fdlr, kgrid, qgrids)./(-4*π*π)
    #     F_new=calcF(delta_0_new, delta_new, fdlr, kgrid)
    #     #lamu = F_new[kf_label,1]/F[kf_label,1]
    #     lamu = dot(F, F_new)
    #     F_new=F_new+shift*F
    #     modulus =sqrt(dot(F_new, F_new))
    #     #modulus = abs(F_new[kf_label,1])
    #     F=F_new/modulus
    #     err=abs(lamu-lamu0)
    #     lamu0=lamu
    #     println(lamu)
    # end
    return delta_0_new, delta_new
end









if abspath(PROGRAM_FILE) == @__FILE__
    fdlr = DLR.DLRGrid(:acorr, 100EF, β, 1e-10)
    bdlr = DLR.DLRGrid(:corr, 100EF, β, 1e-10) 
    ########## non-uniform kgrid #############
    Nk = 16
    order = 8
    maxK = 10.0 * kF
    minK = 0.00001 / (β * kF)
    #minK = 0.0000001    
    kpanel = KPanel(Nk, kF, maxK, minK)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    println("kgrid number: $(length(kgrid.grid))")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    #kernal = dH1_freq(kgrid, qgrids, bdlr, fdlr)
    kernal = dH1_tau(kgrid, qgrids, fdlr)

    #err test section
    #kgrid_double = CompositeGrid(kpanel, 2 * order, :cheb)
    #qgrids_double = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), 2order, :gaussian) for k in kgrid_double.grid]
    #fdlr2 = DLR.DLRGrid(:acorr, 100EF, β, 1e-10) 



    Δ0_final, Δ_final = Implicit_Renorm(kernal , kgrid, qgrids, fdlr)
    #Δ0_final, Δ_final = Explicit_Solver( kgrid, qgrids, fdlr, fdlr2, bdlr)
    #Δ0_final, Δ_final = Explicit_Solver_inherit( kgrid, qgrids, fdlr, fdlr2, bdlr)
    Δ_freq = DLR.tau2matfreq(:acorr, Δ_final, fdlr, fdlr.n, axis=2)
    filename = "./delta_$(WID).dat"



    #println(fdlr.n, fdlr.n[fdlr.size ÷ 2 + 1])
    open(filename, "w") do io
        for r in 1:length(kgrid.grid)
            @printf(io, "%32.17g  %32.17g  %32.17g\n",kgrid.grid[r] ,Δ0_final[r] ,real(Δ_freq[r, fdlr.size ÷ 2 + 1]))
        end
    end

    cut_f = 1
    for r in 1:length(kgrid.grid)
        if kgrid.grid[r] < kF
            global cut_f = r
        end
    end
    filename = "./delta_$(WID)_freq.dat"
    open(filename, "w") do io
        for r in 1:fdlr.size
            @printf(io, "%32.17g  %32.17g\n",fdlr.n[r],real(Δ_freq[cut_f,r]))
        end
    end
    step = fdlr.size÷16
    p = plot(kgrid.grid, Δ0_final[:])
    for i in 1:6
        global p = plot!(p, kgrid.grid, real(Δ_freq[:, fdlr.size ÷ 2 + 1 + i*step]))
    end
    #p = plot!(p, kgrid.grid, F_fine[:, 10])
    display(p)
    readline()

    # F = calcF(Δ0, Δ, fdlr, kgrid)
    # F_freq = DLR.tau2matfreq(:acorr, F, fdlr, fdlr.n, axis=2)
    # Δ_low, Δ_high = Separation_F(F, kgrid, fdlr)
    # q1=20
    # n1=fdlr.size÷2-5
    # n2=fdlr.size÷2+5
    # println(Δ_low[q1,n1:n2])
    # println(Δ_high[q1,n1:n2])
    # println(Δ_low[q1,n1:n2]+Δ_high[q1,n1:n2])
    # println(F_freq[q1,n1:n2])

    # F_test1 =  DLR.matfreq2tau(:acorr, Δ_low, fdlr, fdlr.τ, axis=2)
    # F_test1 =  DLR.tau2matfreq(:acorr, F_test1, fdlr, fdlr.n, axis=2)
    # println(F_test1[q1,n1:n2])
    # println(maximum(abs.(real(F_test1)-Δ_low)),",", maximum(abs.(imag(F_test1))))
end
