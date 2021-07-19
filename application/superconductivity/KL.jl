using QuantumStatistics
using LinearAlgebra
using DelimitedFiles
using Printf
#using Gaston
#using Plots
using Statistics

srcdir = "."
rundir = isempty(ARGS) ? "." : (pwd()*"/"*ARGS[1])

if !(@isdefined β)
    include(rundir*"/parameter.jl")
    using .parameter
end
include("eigensolver.jl")
include("eigen.jl")
include("grid.jl")





if abspath(PROGRAM_FILE) == @__FILE__
    fdlr = DLR.DLRGrid(:acorr, 100EF, β, 1e-10)
    bdlr = DLR.DLRGrid(:corr, 100EF, β, 1e-10) 
    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid]
    #nfermi_grid = [1]

    #kernal_bare, kernal = legendre_dc_2D(nfermi_grid, kgrid, qgrids, kpanel_bose, order_int)
    #kernal_bare, kernal_freq = dH1_freq(kgrid, qgrids, bdlr, fdlr)
    kernal_bare, kernal_freq = legendre_dc(bdlr, kgrid, qgrids, kpanel_bose, order_int)
    
    kernal = real(DLR.matfreq2tau(:corr, kernal_freq, bdlr, fdlr.τ, axis=3))

    dataFileName = rundir*"/delta_$(WID).dat"

    f = open(dataFileName, "r")
    Δ_data = readdlm(f)
    raw_0low = Δ_data[:,1]
    raw_0high = Δ_data[:,2]
    raw_low = Δ_data[:,3]
    raw_high = Δ_data[:,4]
    Δ0_low  = (reshape(raw_0low,(fdlr.size,length(kgrid.grid))))[1,:]
    Δ0_high  = (reshape(raw_0high,(fdlr.size,length(kgrid.grid))))[1,:]
    Δ_low  = transpose(reshape(raw_0low,(fdlr.size,length(kgrid.grid))))
    Δ_high  = transpose(reshape(raw_0high,(fdlr.size,length(kgrid.grid))))
    F_low = calcF(Δ0_low, Δ_low, fdlr, kgrid)
   
    F_ins_low = DLR.tau2dlr(:acorr, F_low, fdlr, axis=2)
    F_ins_low = DLR.dlr2tau(:acorr, F_ins_low, fdlr, [1.0e-12,] , axis=2)[:,1]
    F_high = calcF(Δ0_high, Δ_high, fdlr, kgrid)
    F_ins_high = DLR.tau2dlr(:acorr, F_high, fdlr, axis=2)
    F_ins_high = DLR.dlr2tau(:acorr, F_ins_high, fdlr, [1.0e-12,] , axis=2)[:,1]
    F = F_low + F_high
    F_ins = F_ins_low +F_ins_high
    Δ0 = Δ0_low +Δ0_high
    Δ = Δ_low + Δ_high
    denorm_dy = dlr_dot(F_low, Δ_low, kgrid,qgrids,fdlr)
    denorm = Normalization(F_ins_low, Δ0_low, kgrid,qgrids) + denorm_dy
    const kF_label = searchsortedfirst(kgrid.grid, kF)
    # F_freq = real(DLR.tau2matfreq(:acorr, F, fdlr, nfermi_grid, axis=2))
    # println(fdlr.n)
    # Δ_new = calcΔ_freq(F_freq, kernal,kernal_bare, nfermi_grid, kgrid, qgrids)./(-4*π*π)
    # numer_dy = dlr_dot_freq(F_freq, Δ_new, kgrid, qgrids,nfermi_grid)

    Δ0_new, Δ_new = calcΔ(F, kernal,kernal_bare, fdlr, kgrid, qgrids)./(-4*π*π)
    numer_dy = dlr_dot(F, Δ_new, kgrid, qgrids,fdlr)
    # pic = plot( kgrid.grid,  F_freq[:,1])
    # pic = plot(kgrid.grid,  Δ_new[:,1])

    # pic = plot( kgrid.grid,  Δ_low[:,1])
    # pic = plot( kgrid.grid,  Δ_high[:,1] .+  Δ_low[:,1])


    # display(pic)
    # readline()
    numer = numer_dy



    println(numer/denorm)
    outFileName = rundir*"/flow_$(WID).dat"
    f = open(outFileName, "a")
    @printf(f, "%32.17g\n", numer/denorm)
end
