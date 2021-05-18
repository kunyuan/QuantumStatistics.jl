# This example demonstrated how to calculate one-loop diagram of free electrons using the Monte Carlo module
# Observable is normalized: Γ₄*N_F where N_F is the free electron density of states

using QuantumStatistics, LinearAlgebra, Random, Printf,  BenchmarkTools, InteractiveUtils, Parameters

const Step = 1e7 # MC steps of each block

include("RPA.jl") # dW0 will be only calculated once in the master, then distributed to other workers. Therefore, there is no need to import RPA.jl for all workers.

include("parameter.jl")
include("interaction.jl")

# claim all globals to be constant, otherwise, global variables could impact the efficiency
########################### parameters ##################################
const IsF = false # calculate quasiparticle interaction F or not
const AngSize = 16

########################## variables for MC integration ##################
const KInL = [kF, 0.0, 0.0] # incoming momentum of the left particle
const Qd = [0.0, 0.0, 0.0] # transfer momentum is zero in the forward scattering channel

struct Para{Q,T}
    extAngle::Vector{Float64}
    dW0::Matrix{Float64}
    qgrid::Q
    τgrid::T
    function Para(AngSize)
        extAngle = collect(LinRange(0.0, π, AngSize)) # external angle grid
        qgrid = Grid.boseK(kF, 6kF, 0.2kF, 256) 
        τgrid = Grid.tau(β, EF / 20, 128)

        vqinv = [(q^2 + mass2) / (4π * e0^2) for q in qgrid.grid]
        dW0 = dWRPA(vqinv, qgrid.grid, τgrid.grid, kF, β, spin, me) # dynamic part of the effective interaction
        return new{typeof(qgrid),typeof(τgrid)}(extAngle, dW0, qgrid, τgrid)
    end
end

function phase(tInL, tOutL, tInR, tOutR)
    if (IsF)
        return cos(π * ((tInL + tOutL) - (tInR + tOutR)));
    else
        return cos(π * ((tInL - tOutL) + (tInR - tOutR)))
    end
end

function integrand(config)
    @assert config.curr == 1

    T, K, Ang = config.var[1], config.var[2], config.var[3]
    k1, k2 = K[1], K[1] - Qd
    t1, t2 = T[1], T[2] # t1, t2 both have two tau variables
    θ = config.para.extAngle[Ang[1]] # angle of the external momentum on the right
    KInR = [kF * cos(θ), kF * sin(θ), 0.0]

    vld, wld, vle, wle = vertexDynamic(config, Qd, KInL - k1, t1[1], t1[2])
    vrd, wrd, vre, wre = vertexDynamic(config, Qd, KInR - k2, t2[1], t2[2])

    ϵ1, ϵ2 = (dot(k1, k1) - kF^2) / (2me), (dot(k2, k2) - kF^2) / (2me) 
    wd, we = 0.0, 0.0
    # possible green's functions on the top
    gt1 = Spectral.kernelFermiT(t2[1] - t1[1], ϵ1, β)


    # gt2 = Spectral.kernelFermiT(t1[1] - t2[1], ϵ2, β)
    # wd += 1.0 / β * 1.0 / β * gt1 * gt2 / (2π)^3 * phase(t1[1], t1[1], t2[1], t2[1])

    # gt3 = Spectral.kernelFermiT(t1[1] - t2[2], ϵ2, β)
    # G = gt1 * gt3 / (2π)^3 * phase(t1[1], t1[1], t2[2], t2[1])
    # wd += G * (vld * wre)

    # wd += spin * (vld + wld) * (vrd + wrd) * gt1 * gt2 / (2π)^3 * phase(t1[1], t1[1], t2[1], t2[1])
    # println(vld, ", ", wld, "; ", vrd, ", ", wld)

    ############## Diagram v x v ######################
    """
      KInL                      KInR
       |                         | 
  t1.L ↑     t1.L       t2.L     ↑ t2.L
       |-------------->----------|
       |       |    k1    |      |
       |   ve  |          |  ve  |
       |       |    k2    |      |
       |--------------<----------|
  t1.L ↑    t1.L        t2.L     ↑ t2.L
       |                         | 
      KInL                      KInR
"""
    gd1 = Spectral.kernelFermiT(t1[1] - t2[1], ϵ2, β)
    G = gt1 * gd1 / (2π)^3 * phase(t1[1], t1[1], t2[1], t2[1])
    we += G * (vle * vre)
    ##################################################

    ############## Diagram w x v ######################
    """
      KInL                      KInR
       |                         | 
  t1.R ↑     t1.L       t2.L     ↑ t2.L
       |-------------->----------|
       |       |    k1    |      |
       |   we  |          |  ve  |
       |       |    k2    |      |
       |--------------<----------|
  t1.L ↑    t1.R        t2.L     ↑ t2.L
       |                         | 
      KInL                      KInR
    """
    gd2 = Spectral.kernelFermiT(t1[2] - t2[1], ϵ2, β)
    G = gt1 * gd2 / (2π)^3 * phase(t1[1], t1[2], t2[1], t2[1])
    we += G * (wle * vre) 
    ##################################################

    ############## Diagram v x w ######################
    """
      KInL                      KInR
       |                         | 
  t1.L ↑     t1.L       t2.L     ↑ t2.L
       |-------------->----------|
       |       |    k1    |      |
       |   ve  |          |  we  |
       |       |    k2    |      |
       |--------------<----------|
  t1.L ↑    t1.L        t2.R     ↑ t2.R
       |                         | 
      KInL                      KInR
    """
    gd3 = Spectral.kernelFermiT(t1[1] - t2[2], ϵ2, β)
    G = gt1 * gd3 / (2π)^3 * phase(t1[1], t1[1], t2[2], t2[1])
    we += G * (vle * wre)
    ##################################################

    ############## Diagram w x w ######################
    """
      KInL                      KInR
       |                         | 
  t1.R ↑     t1.L       t2.L     ↑ t2.L
       |-------------->----------|
       |       |    k1    |      |
       |   we  |          |  we  |
       |       |    k2    |      |
       |--------------<----------|
  t1.L ↑    t1.R        t2.R     ↑ t2.R
       |                         | 
      KInL                      KInR
"""
    gd4 = Spectral.kernelFermiT(t1[2] - t2[2], ϵ2, β)
    G = gt1 * gd4 / (2π)^3 * phase(t1[1], t1[2], t2[2], t2[1])
    we += G * (wle * wre)
    ##################################################

    return Weight(wd, we)
end

function measure(config)
    @assert config.curr == 1

    angidx = config.var[3][1]
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    config.observable[:, angidx] .+= weight / abs(weight) * factor
end

function run(steps)
    T = MonteCarlo.TauPair(β, β / 2.0)
    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    Ext = MonteCarlo.Discrete(1, AngSize) # external variable is specified

    dof = [[2, 1, 1],]
    obs = zeros(Float64, (2, AngSize))

    para = Para(AngSize)

    config = MonteCarlo.Configuration(steps, (T, K, Ext), dof, obs; para=para)
    avg, std = MonteCarlo.sample(config, integrand, measure; print=10, Nblock=16)

    if isnothing(avg) == false
        NF = TwoPoint.LindhardΩnFiniteTemperature(dim, 0.0, 0, kF, β, me, spin)[1]
        println("NF = $NF")

        avg .*= NF / β
        std .*= NF / β

        for (idx, angle) in enumerate(para.extAngle)
            @printf("%10.6f   %10.6f ± %10.6f  %10.6f ± %10.6f\n", angle, avg[1, idx], std[1,idx], avg[2,idx], std[2,idx])
        end
    end
end

# @btime run(1, 10)
run(Step)
# @time run(Repeat, totalStep)